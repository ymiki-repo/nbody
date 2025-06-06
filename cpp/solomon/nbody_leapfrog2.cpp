///
/// @file nbody_leapfrog2.cpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief sample implementation of direct N-body simulation (orbit integration: 2nd-order leapfrog scheme)
///
/// @copyright Copyright (c) 2025 Yohei MIKI
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
#include <solomon.hpp>
#include <string>       // std::string
#include <type_traits>  // std::remove_const_t

#include "common/cfg.hpp"
#include "common/conservatives.hpp"
#include "common/init.hpp"
#include "common/io.hpp"
#include "common/type.hpp"
#include "util/hdf5.hpp"
#include "util/macro.hpp"
#include "util/timer.hpp"

// #define _OPENMP 201307
// #define OFFLOAD_BY_OPENMP_TARGET
// #include "solomon/solomon/solomon.hpp"

constexpr type::flt_acc newton = AS_FLT_ACC(1.0);  // gravitational constant

#ifndef NTHREADS
constexpr type::int_idx NTHREADS = 1024U;
#endif  // NTHREADS

///
/// @brief calculate gravitational acceleration
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
  OFFLOAD(AS_INDEPENDENT, NUM_THREADS(NTHREADS))
  for (std::remove_const_t<decltype(Ni)> ii = 0U; ii < Ni; ii++) {
    // initialization
    const auto pi = ipos[ii];
    std::remove_reference_t<decltype(*iacc)> ai = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};

    // force evaluation
    PRAGMA_ACC_LOOP(ACC_CLAUSE_SEQ)
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
}

#ifndef BENCHMARK_MODE
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
  OFFLOAD(AS_INDEPENDENT)
  for (std::remove_const_t<decltype(Ni)> ii = 0U; ii < Ni; ii++) {
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
  OFFLOAD(AS_INDEPENDENT)
  for (std::remove_const_t<decltype(num)> ii = 0U; ii < num; ii++) {
    // initialization
    auto vi = vel[ii];
    const auto ai = acc[ii];

    vi.x += dt * CAST2VEL(ai.x);
    vi.y += dt * CAST2VEL(ai.y);
    vi.z += dt * CAST2VEL(ai.z);

    vel[ii] = vi;
  }
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
  OFFLOAD(AS_INDEPENDENT)
  for (std::remove_const_t<decltype(num)> ii = 0U; ii < num; ii++) {
    // initialization
    auto pi = pos[ii];
    const auto vi = vel[ii];

    pi.x += dt * CAST2POS(vi.x);
    pi.y += dt * CAST2POS(vi.y);
    pi.z += dt * CAST2POS(vi.z);

    pos[ii] = pi;
  }
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
  const auto dt_2 = AS_FLT_VEL(0.5) * dt;
  OFFLOAD(AS_INDEPENDENT)
  for (std::remove_const_t<decltype(num)> ii = 0U; ii < num; ii++) {
    // initialization
    auto vi = vel_src[ii];
    const auto ai = acc[ii];

    vi.x -= dt_2 * CAST2VEL(ai.x);
    vi.y -= dt_2 * CAST2VEL(ai.y);
    vi.z -= dt_2 * CAST2VEL(ai.z);

    vel[ii] = vi;
  }
}
#endif  // BENCHMARK_MODE

///
/// @brief allocate memory for N-body particles
///
/// @param[out] pos position of N-body particles
/// @param[out] vel velocity of N-body particles
/// @param[out] vel_tmp tentative array for velocity of N-body particles
/// @param[out] acc acceleration of N-body particles
/// @param[in] num number of N-body particles
///
static inline void allocate_Nbody_particles(type::position **pos, type::velocity **vel, type::velocity **vel_tmp, type::acceleration **acc, const type::int_idx num) {
  const auto size = static_cast<size_t>(num);
  *pos = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**pos)>[size];
  *vel = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**vel)>[size];
  *vel_tmp = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**vel_tmp)>[size];
  *acc = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**acc)>[size];

  // zero-clear arrays (for safety of massless particles)
  constexpr std::remove_reference_t<decltype(**pos)> p_zero = {AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0)};
  constexpr std::remove_reference_t<decltype(**vel)> v_zero = {AS_FLT_VEL(0.0), AS_FLT_VEL(0.0), AS_FLT_VEL(0.0)};
  constexpr std::remove_reference_t<decltype(**acc)> a_zero = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};
#pragma omp parallel for
  for (std::remove_const_t<decltype(size)> ii = 0U; ii < size; ii++) {
    (*pos)[ii] = p_zero;
    (*vel)[ii] = v_zero;
    (*vel_tmp)[ii] = v_zero;
    (*acc)[ii] = a_zero;
  }
}

///
/// @brief deallocate memory for N-body particles
///
/// @param[out] pos position of N-body particles
/// @param[out] vel velocity of N-body particles
/// @param[out] vel_tmp tentative array for velocity of N-body particles
/// @param[out] acc acceleration of N-body particles
///
static inline void release_Nbody_particles(type::position *pos, type::velocity *vel, type::velocity *vel_tmp, type::acceleration *acc) {
  delete[] pos;
  delete[] vel;
  delete[] vel_tmp;
  delete[] acc;
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
    allocate_Nbody_particles(&pos, &vel, &vel_tmp, &acc, num);
    MALLOC_ON_DEVICE(pos [0:num], vel [0:num], vel_tmp [0:num], acc [0:num])

    // generate initial-condition
    init::set_uniform_sphere(num, pos, vel, M_tot, rad, virial, CAST2VEL(newton));
    MEMCPY_H2D(pos [0:num], vel [0:num])

#ifndef BENCHMARK_MODE
    // write the first snapshot
    calc_acc(num, pos, acc, num, pos, eps2);
    trim_acc(num, acc
#ifdef CALCULATE_POTENTIAL
             ,
             pos, eps_inv
#endif  // CALCULATE_POTENTIAL
    );
    MEMCPY_D2H(acc [0:num])
    auto error = conservatives();
    io::write_snapshot(num, pos, vel, acc, file.c_str(), present, time, error);

    // half-step integration for velocity
    kick(num, vel, acc, AS_FP_M(0.5) * dt);

    while (present < snp_fin) {
      step++;
      time_from_snapshot += dt;
      if (time_from_snapshot >= snapshot_interval) {
        present++;
      }
      DEBUG_PRINT(std::cout, "t = " << time + time_from_snapshot << ", step = " << step)

      // orbit integration (2nd-order leapfrog scheme)
      drift(num, pos, vel, dt);
      calc_acc(num, pos, acc, num, pos, eps2);
      trim_acc(num, acc
#ifdef CALCULATE_POTENTIAL
               ,
               pos, eps_inv
#endif  // CALCULATE_POTENTIAL
      );
      kick(num, vel, acc, dt);

      // write snapshot
      if (present > previous) {
        previous = present;
        time_from_snapshot = AS_FP_M(0.0);
        time += snapshot_interval;
        kick_backward_half(num, vel, acc, vel_tmp, dt);
        MEMCPY_D2H(pos [0:num], vel_tmp [0:num], acc [0:num])
        io::write_snapshot(num, pos, vel_tmp, acc, file.c_str(), present, time, error);
      }
    }
#else   // BENCHMARK_MODE
  // launch benchmark
  auto timer = util::timer();
  timer.start();
  calc_acc(num, pos, acc, num, pos, eps2);
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
    timer.start();
    for (decltype(iter) loop = 0; loop < iter; loop++) {
      calc_acc(num, pos, acc, num, pos, eps2);
    }
    timer.stop();
    elapsed = timer.get_elapsed_wall();
  }

  // finalize benchmark
  io::write_log(argv[0], elapsed, iter, num, file.c_str());
#endif  // BENCHMARK_MODE

    // memory deallocation
    FREE_FROM_DEVICE(pos [0:num], vel [0:num], vel_tmp [0:num], acc [0:num])
    release_Nbody_particles(pos, vel, vel_tmp, acc);

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
