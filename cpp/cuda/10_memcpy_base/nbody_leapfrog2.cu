///
/// @file nbody_leapfrog2.cu
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief sample implementation of direct N-body simulation (orbit integration: 2nd-order leapfrog scheme)
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///

#include <unistd.h>

#include <boost/filesystem.hpp>                // boost::filesystem
#include <boost/math/constants/constants.hpp>  // boost::math::constants::two_pi
#include <boost/program_options.hpp>           // boost::program_options
#include <cmath>                               // std::fma
#include <cstdint>                             // int32_t
#include <iostream>                            // std::cout
#include <string>                              // std::string
#include <type_traits>                         // std::remove_const_t

#include "common/cfg.hpp"
#include "common/conservatives.hpp"
#include "common/init.hpp"
#include "common/io.hpp"
#include "common/type.hpp"
#include "util/hdf5.hpp"
#include "util/macro.hpp"
#include "util/timer.hpp"

constexpr type::flt_acc newton = AS_FLT_ACC(1.0);  // gravitational constant

#ifndef NTHREADS
constexpr type::int_idx NTHREADS = 128U;
#endif  // NTHREADS

///
/// @brief required block size for the given problem size and number of threads per thread-block
///
constexpr auto BLOCKSIZE(const type::int_idx num, const type::int_idx thread) { return (1U + ((num - 1U) / thread)); }

///
/// @brief calculate gravitational acceleration on accelerator device
///
/// @param[in] ipos position of i-particles
/// @param[out] iacc acceleration of i-particles
/// @param[in] Nj number of j-particles
/// @param[in] jpos position of j-particles
/// @param[in] eps2 square of softening length
/// @param[in] ivel velocity of i-particles
///
__global__ void calc_acc_device(const type::position *const ipos, type::acceleration *__restrict iacc, const type::int_idx Nj, const type::position *const jpos, const type::flt_pos eps2) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  // initialization
  const auto pi = ipos[ii];
  std::remove_reference_t<decltype(*iacc)> ai = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};

  // force evaluation
  for (std::remove_const_t<decltype(Nj)> jj = 0U; jj < Nj; jj++) {
    // load j-particle
    const auto pj = jpos[jj];

    // calculate particle-particle interaction
    const auto dx = pj.x - pi.x;
    const auto dy = pj.y - pi.y;
    const auto dz = pj.z - pi.z;
    const auto r2 = eps2 + dx * dx + dy * dy + dz * dz;
    const auto r_inv = AS_FP_L(1.0) / std::sqrt(type::cast2fp_l(r2));
    const auto r2_inv = r_inv * r_inv;
    const auto alp = pj.w * CAST2ACC(r_inv * r2_inv);

    // force accumulation
    ai.x += alp * dx;
    ai.y += alp * dy;
    ai.z += alp * dz;

#ifdef CALCULATE_POTENTIAL
    // gravitational potential
    ai.w += alp * r2;
#endif  // CALCULATE_POTENTIAL
  }
  iacc[ii] = ai;
}

///
/// @brief calculate gravitational acceleration on accelerator device
///
/// @param[in] Ni number of i-particles
/// @param[in] ipos position of i-particles
/// @param[out] iacc acceleration of i-particles
/// @param[in] Nj number of j-particles
/// @param[in] jpos position of j-particles
/// @param[in] eps2 square of softening length
/// @param[in] ivel velocity of i-particles
///
static inline void calc_acc(const type::int_idx Ni, const type::position *const ipos, type::acceleration *__restrict iacc, const type::int_idx Nj, const type::position *const jpos, const type::flt_pos eps2) {
  calc_acc_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS>>>(ipos, iacc, Nj, jpos, eps2);
}

#ifndef BENCHMARK_MODE
///
/// @brief finalize calculation of gravitational acceleration on accelerator device
///
/// @param[in,out] acc acceleration of i-particles
/// @param[in] pos position of i-particles
/// @param[in] eps_inv inverse of softening length
///
__global__ void trim_acc_device(type::acceleration *__restrict acc
#ifdef CALCULATE_POTENTIAL
                                ,
                                const type::position *const pos, const type::flt_acc eps_inv
#endif  // CALCULATE_POTENTIAL
) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  // initialization
  auto ai = acc[ii];

  ai.x *= newton;
  ai.y *= newton;
  ai.z *= newton;

#ifdef CALCULATE_POTENTIAL
  ai.w = (-ai.w + eps_inv * CAST2ACC(pos[ii].w)) * newton;
#endif  // CALCULATE_POTENTIAL

  acc[ii] = ai;
}

///
/// @brief finalize calculation of gravitational acceleration
///
/// @param[in] Ni number of i-particles
/// @param[in,out] acc acceleration of i-particles
/// @param[in] pos position of i-particles
/// @param[in] eps_inv inverse of softening length
///
static inline void trim_acc(const type::int_idx Ni, type::acceleration *__restrict acc
#ifdef CALCULATE_POTENTIAL
                            ,
                            const type::position *const pos, const type::flt_acc eps_inv
#endif  // CALCULATE_POTENTIAL
) {
  trim_acc_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS>>>(acc
#ifdef CALCULATE_POTENTIAL
                                                         ,
                                                         pos, eps_inv
#endif  // CALCULATE_POTENTIAL
  );
}

///
/// @brief integrate velocity on accelerator device
///
/// @param[in,out] vel velocity of N-body particles
/// @param[in] acc acceleration of N-body particles
/// @param[in] dt time step
///
__global__ void kick_device(type::velocity *__restrict vel, const type::acceleration *const acc, const type::flt_vel dt) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  // initialization
  auto vi = vel[ii];
  const auto ai = acc[ii];

  vi.x += dt * CAST2VEL(ai.x);
  vi.y += dt * CAST2VEL(ai.y);
  vi.z += dt * CAST2VEL(ai.z);

  vel[ii] = vi;
}

///
/// @brief integrate velocity
///
/// @param[in] num number of N-body particles
/// @param[in,out] vel velocity of N-body particles
/// @param[in] acc acceleration of N-body particles
/// @param[in] dt time step
///
static inline void kick(const type::int_idx num, type::velocity *__restrict vel, const type::acceleration *const acc, const type::flt_vel dt) {
  kick_device<<<BLOCKSIZE(num, NTHREADS), NTHREADS>>>(vel, acc, dt);
}

///
/// @brief integrate position on accelerator device
///
/// @param[in] num number of N-body particles
/// @param[in,out] pos position of N-body particles
/// @param[in] vel velocity of N-body particles
/// @param[in] dt time step
///
__global__ void drift_device(type::position *__restrict pos, const type::velocity *const vel, const type::flt_pos dt) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  // initialization
  auto pi = pos[ii];
  const auto vi = vel[ii];

  pi.x += dt * CAST2POS(vi.x);
  pi.y += dt * CAST2POS(vi.y);
  pi.z += dt * CAST2POS(vi.z);

  pos[ii] = pi;
}

///
/// @brief integrate position
///
/// @param[in] num number of N-body particles
/// @param[in,out] pos position of N-body particles
/// @param[in] vel velocity of N-body particles
/// @param[in] dt time step
///
static inline void drift(const type::int_idx num, type::position *__restrict pos, const type::velocity *const vel, const type::flt_pos dt) {
  drift_device<<<BLOCKSIZE(num, NTHREADS), NTHREADS>>>(pos, vel, dt);
}

///
/// @brief integrate velocity in half-step backward on accelerator device
///
/// @param[in] vel_src velocity of N-body particles
/// @param[in] acc acceleration of N-body particles
/// @param[out] vel velocity of N-body particles
/// @param[in] dt time step
///
__global__ void kick_backward_half_device(const type::velocity *const vel_src, const type::acceleration *const acc, type::velocity *__restrict vel, const type::flt_vel dt) {
  const auto dt_2 = AS_FLT_VEL(0.5) * dt;
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  // initialization
  auto vi = vel_src[ii];
  const auto ai = acc[ii];

  vi.x -= dt_2 * CAST2VEL(ai.x);
  vi.y -= dt_2 * CAST2VEL(ai.y);
  vi.z -= dt_2 * CAST2VEL(ai.z);

  vel[ii] = vi;
}

///
/// @brief integrate velocity in half-step backward
///
/// @param[in] num number of N-body particles
/// @param[in] vel_src velocity of N-body particles
/// @param[in] acc acceleration of N-body particles
/// @param[out] vel velocity of N-body particles
/// @param[in] dt time step
///
static inline void kick_backward_half(const type::int_idx num, const type::velocity *const vel_src, const type::acceleration *const acc, type::velocity *__restrict vel, const type::flt_vel dt) {
  kick_backward_half_device<<<BLOCKSIZE(num, NTHREADS), NTHREADS>>>(vel_src, acc, vel, dt);
}
#endif  // BENCHMARK_MODE

///
/// @brief allocate memory for N-body particles
///
/// @param[out] pos_hst position of N-body particles (host memory)
/// @param[out] vel_hst velocity of N-body particles (host memory)
/// @param[out] acc_hst acceleration of N-body particles (host memory)
/// @param[out] pos_dev position of N-body particles (device memory)
/// @param[out] vel_dev velocity of N-body particles (device memory)
/// @param[out] acc_dev acceleration of N-body particles (device memory)
/// @param[out] vel_tmp tentative array for velocity of N-body particles (device memory)
/// @param[in] num number of N-body particles
///
static inline void allocate_Nbody_particles(
    type::position **pos_hst, type::velocity **vel_hst, type::acceleration **acc_hst,
    type::position **pos_dev, type::velocity **vel_dev, type::acceleration **acc_dev, type::velocity **vel_tmp, const type::int_idx num) {
  auto size = static_cast<size_t>(num);
  if ((num % NTHREADS) != 0U) {
    size += static_cast<decltype(size)>(NTHREADS - (num % NTHREADS));
  }
  cudaMallocHost((void **)pos_hst, size * sizeof(std::remove_reference_t<decltype(**pos_hst)>));
  cudaMallocHost((void **)vel_hst, size * sizeof(std::remove_reference_t<decltype(**vel_hst)>));
  cudaMallocHost((void **)acc_hst, size * sizeof(std::remove_reference_t<decltype(**acc_hst)>));
  cudaMalloc((void **)pos_dev, size * sizeof(std::remove_reference_t<decltype(**pos_dev)>));
  cudaMalloc((void **)vel_dev, size * sizeof(std::remove_reference_t<decltype(**vel_dev)>));
  cudaMalloc((void **)vel_tmp, size * sizeof(std::remove_reference_t<decltype(**vel_tmp)>));
  cudaMalloc((void **)acc_dev, size * sizeof(std::remove_reference_t<decltype(**acc_dev)>));

  // zero-clear arrays (for safety of massless particles)
  constexpr std::remove_reference_t<decltype(**pos_hst)> p_zero = {AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0)};
  constexpr std::remove_reference_t<decltype(**vel_hst)> v_zero = {AS_FLT_VEL(0.0), AS_FLT_VEL(0.0), AS_FLT_VEL(0.0)};
  constexpr std::remove_reference_t<decltype(**acc_hst)> a_zero = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};
#pragma omp parallel for
  for (std::remove_const_t<decltype(size)> ii = 0U; ii < size; ii++) {
    (*pos_hst)[ii] = p_zero;
    (*vel_hst)[ii] = v_zero;
    (*acc_hst)[ii] = a_zero;
  }

  // compy massless particles to device (just for safety)
  cudaMemcpy(*pos_dev, *pos_hst, size * sizeof(std::remove_reference_t<decltype(**pos_dev)>), cudaMemcpyHostToDevice);
  cudaMemcpy(*vel_dev, *vel_hst, size * sizeof(std::remove_reference_t<decltype(**vel_dev)>), cudaMemcpyHostToDevice);
  cudaMemcpy(*vel_tmp, *vel_hst, size * sizeof(std::remove_reference_t<decltype(**vel_tmp)>), cudaMemcpyHostToDevice);
  cudaMemcpy(*acc_dev, *acc_hst, size * sizeof(std::remove_reference_t<decltype(**acc_dev)>), cudaMemcpyHostToDevice);
}

///
/// @brief deallocate memory for N-body particles
///
/// @param[out] pos_hst position of N-body particles (host memory)
/// @param[out] vel_hst velocity of N-body particles (host memory)
/// @param[out] acc_hst acceleration of N-body particles (host memory)
/// @param[out] pos_dev position of N-body particles (device memory)
/// @param[out] vel_dev velocity of N-body particles (device memory)
/// @param[out] acc_dev acceleration of N-body particles (device memory)
/// @param[out] vel_tmp tentative array for velocity of N-body particles (device memory)
///
static inline void release_Nbody_particles(type::position *pos_hst, type::velocity *vel_hst, type::acceleration *acc_hst, type::position *pos_dev, type::velocity *vel_dev, type::acceleration *acc_dev, type::velocity *vel_tmp) {
  cudaFreeHost(pos_hst);
  cudaFreeHost(vel_hst);
  cudaFreeHost(acc_hst);
  cudaFree(pos_dev);
  cudaFree(vel_dev);
  cudaFree(vel_tmp);
  cudaFree(acc_dev);
}

///
/// @brief configure the target GPU device if necessary
///
static inline void configure_gpu() {
  // this sample code does not utilize the shared memory, so maximize the capacity of the L1 cache (0% for shared memory)
  cudaFuncSetAttribute(calc_acc_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
#ifndef BENCHMARK_MODE
  cudaFuncSetAttribute(trim_acc_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(kick_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(drift_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
#endif  // BENCHMARK_MODE
}

auto main([[maybe_unused]] const int32_t argc, [[maybe_unused]] const char *const *const argv) -> int32_t {
  // record time-to-solution
  auto time_to_solution = util::timer();
  time_to_solution.start();

  // initialize the simulation
  auto cfg = config();
  cfg.configure(argc, argv);

  // read input parameters
  const auto file = cfg.get_file();
  const auto eps = cfg.get_eps();
  const auto M_tot = cfg.get_mass();
  const auto rad = cfg.get_radius();
  const auto virial = cfg.get_virial();
#ifdef BENCHMARK_MODE
  const auto [num_min, num_max, num_bin] = cfg.get_num();
#else   // BENCHMARK_MODE
  const auto num = cfg.get_num();
  const auto [ft, snapshot_interval] = cfg.get_time();
  const auto dt = cfg.get_dt();
#endif  // BENCHMARK_MODE
  const auto eps2 = eps * eps;

#ifndef BENCHMARK_MODE
  // initialize N-body simulation
  type::fp_m time = AS_FP_M(0.0);
  type::fp_m time_from_snapshot = AS_FP_M(0.0);
  int32_t step = 0;
  int32_t present = 0;
  auto previous = present;
  const auto snp_fin = static_cast<int32_t>(std::ceil(ft / snapshot_interval));
#ifdef CALCULATE_POTENTIAL
  const auto eps_inv = AS_FLT_ACC(1.0) / CAST2ACC(eps);
#endif  // CALCULATE_POTENTIAL
  util::hdf5::commit_datatype_vec3();
  util::hdf5::commit_datatype_vec4();
#endif  // BENCHMARK_MODE

  // configure accelerator device
  configure_gpu();

#ifdef BENCHMARK_MODE
  const auto num_logbin = std::log2(num_max / num_min) / static_cast<double>((num_bin > 1) ? (num_bin - 1) : (num_bin));
  for (std::remove_const_t<decltype(num_bin)> ii = 0; ii < num_bin; ii++) {
    const auto num = static_cast<type::int_idx>(std::nearbyint(num_min * std::exp2(static_cast<double>(ii) * num_logbin)));
#endif  // BENCHMARK_MODE

    // memory allocation
    type::position *pos = nullptr;
    type::velocity *vel = nullptr;
    type::velocity *vel_tmp = nullptr;
    type::acceleration *acc = nullptr;
    type::position *pos_dev = nullptr;
    type::velocity *vel_dev = nullptr;
    type::acceleration *acc_dev = nullptr;
    allocate_Nbody_particles(&pos, &vel, &acc, &pos_dev, &vel_dev, &acc_dev, &vel_tmp, num);

    // generate initial-condition
    init::set_uniform_sphere(num, pos, vel, M_tot, rad, virial, CAST2VEL(newton));
    cudaMemcpy(pos_dev, pos, num * sizeof(std::remove_reference_t<decltype(*pos_dev)>), cudaMemcpyHostToDevice);
    cudaMemcpy(vel_dev, vel, num * sizeof(std::remove_reference_t<decltype(*vel_dev)>), cudaMemcpyHostToDevice);

#ifndef BENCHMARK_MODE
    // write the first snapshot
    calc_acc(num, pos_dev, acc_dev, num, pos_dev, eps2);
    trim_acc(num, acc_dev
#ifdef CALCULATE_POTENTIAL
             ,
             pos_dev, eps_inv
#endif  // CALCULATE_POTENTIAL
    );
    // write snapshot
    auto error = conservatives();
    cudaMemcpy(acc, acc_dev, num * sizeof(type::acceleration), cudaMemcpyDeviceToHost);
    io::write_snapshot(num, pos, vel, acc, file.c_str(), present, time, error);

    // half-step integration for velocity
    kick(num, vel_dev, acc_dev, AS_FP_M(0.5) * dt);

    while (present < snp_fin) {
      step++;
      time_from_snapshot += dt;
      if (time_from_snapshot >= snapshot_interval) {
        present++;
      }
      DEBUG_PRINT(std::cout, "t = " << time + time_from_snapshot << ", step = " << step)

      // orbit integration (2nd-order leapfrog scheme)
      drift(num, pos_dev, vel_dev, dt);
      calc_acc(num, pos_dev, acc_dev, num, pos_dev, eps2);
      trim_acc(num, acc_dev
#ifdef CALCULATE_POTENTIAL
               ,
               pos, eps_inv
#endif  // CALCULATE_POTENTIAL
      );
      kick(num, vel_dev, acc_dev, dt);

      // write snapshot
      if (present > previous) {
        previous = present;
        time_from_snapshot = AS_FP_M(0.0);
        time += snapshot_interval;
        kick_backward_half(num, vel_dev, acc_dev, vel_tmp, dt);
        cudaMemcpy(pos, pos_dev, num * sizeof(std::remove_reference_t<decltype(*pos)>), cudaMemcpyDeviceToHost);
        cudaMemcpy(vel, vel_tmp, num * sizeof(std::remove_reference_t<decltype(*vel)>), cudaMemcpyDeviceToHost);
        cudaMemcpy(acc, acc_dev, num * sizeof(std::remove_reference_t<decltype(*acc)>), cudaMemcpyDeviceToHost);
        io::write_snapshot(num, pos, vel, acc, file.c_str(), present, time, error);
      }
    }
#else   // BENCHMARK_MODE
  // launch benchmark
  auto timer = util::timer();
  cudaDeviceSynchronize();  // complete the calculation on GPU before reading results from CPU
  timer.start();
  calc_acc(num, pos_dev, acc_dev, num, pos_dev, eps2);
  cudaDeviceSynchronize();  // complete the calculation on GPU before reading results from CPU
  timer.stop();
  auto elapsed = timer.get_elapsed_wall();
  int32_t iter = 1;

  // increase iteration counts if the measured time is too short
  const auto minimum_elapsed = cfg.get_minimum_elapsed_time();
  while (elapsed < minimum_elapsed) {
    // predict the iteration counts
    constexpr double booster = 1.25;  // additional safety-parameter to reduce rejection rate
    iter = static_cast<decltype(iter)>(std::exp2(std::ceil(std::log2(static_cast<double>(iter) * booster * minimum_elapsed / elapsed))));

    // re-execute the benchmark
    timer.clear();
    cudaDeviceSynchronize();  // complete the calculation on GPU before reading results from CPU
    timer.start();
    for (int32_t loop = 0; loop < iter; loop++) {
      calc_acc(num, pos_dev, acc_dev, num, pos_dev, eps2);
    }
    cudaDeviceSynchronize();  // complete the calculation on GPU before reading results from CPU
    timer.stop();
    elapsed = timer.get_elapsed_wall();
  }

  // finalize benchmark
  io::write_log(argv[0], elapsed, iter, num, file.c_str());
#endif  // BENCHMARK_MODE

    // memory deallocation
    release_Nbody_particles(pos, vel, acc, pos_dev, vel_dev, acc_dev, vel_tmp);

#ifdef BENCHMARK_MODE
  }
#else  // BENCHMARK_MODE
  util::hdf5::remove_datatype_vec3();
  util::hdf5::remove_datatype_vec4();

  time_to_solution.stop();
  io::write_log(argv[0], time_to_solution.get_elapsed_wall(), step, num, file.c_str()
#if defined(CALCULATE_POTENTIAL) && !defined(BENCHMARK_MODE)
                                                                             ,
                dt, error
#endif  // defined(CALCULATE_POTENTIAL) && !defined(BENCHMARK_MODE)
  );
#endif  // BENCHMARK_MODE

  return (0);
}
