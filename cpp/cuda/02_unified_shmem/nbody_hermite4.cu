///
/// @file nbody_hermite4.cu
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief sample implementation of direct N-body simulation (orbit integration: 4th-order Hermite scheme)
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///

#include <unistd.h>

#include <boost/filesystem.hpp>                ///< boost::filesystem
#include <boost/math/constants/constants.hpp>  ///< boost::math::constants::two_pi
#include <boost/program_options.hpp>           ///< boost::program_options
#include <cmath>                               ///< std::fma
#include <cstdint>                             ///< int32_t
#include <cub/device/device_radix_sort.cuh>
#include <iostream>
#include <memory>
#include <string>

#include "common/cfg.hpp"
#include "common/conservatives.hpp"
#include "common/init.hpp"
#include "common/io.hpp"
#include "common/type.hpp"
#include "util/hdf5.hpp"
#include "util/macro.hpp"
#include "util/timer.hpp"

constexpr type::flt_acc newton = AS_FLT_ACC(1.0);  ///< gravitational constant

#ifndef NTHREADS
constexpr type::int_idx NTHREADS = 512U;
#endif  // NTHREADS

#ifndef NUNROLL
#define NUNROLL (128)
#endif  // NUNROLL

#if NUNROLL == 1
#define PRAGMA_UNROLL _Pragma("unroll 1")
#elif NUNROLL == 2
#define PRAGMA_UNROLL _Pragma("unroll 2")
#elif NUNROLL == 4
#define PRAGMA_UNROLL _Pragma("unroll 4")
#elif NUNROLL == 8
#define PRAGMA_UNROLL _Pragma("unroll 8")
#elif NUNROLL == 16
#define PRAGMA_UNROLL _Pragma("unroll 16")
#elif NUNROLL == 32
#define PRAGMA_UNROLL _Pragma("unroll 32")
#elif NUNROLL == 64
#define PRAGMA_UNROLL _Pragma("unroll 64")
#elif NUNROLL == 128
#define PRAGMA_UNROLL _Pragma("unroll 128")
#elif NUNROLL == 256
#define PRAGMA_UNROLL _Pragma("unroll 256")
#elif NUNROLL == 512
#define PRAGMA_UNROLL _Pragma("unroll 512")
#elif NUNROLL == 1024
#define PRAGMA_UNROLL _Pragma("unroll 1024")
#endif

///
/// @brief required block size for the given problem size and number of threads per thread-block
///
constexpr auto BLOCKSIZE(const type::int_idx num, const type::int_idx thread) { return (1U + ((num - 1U) / thread)); }

extern __shared__ type::fp_m dynamic_shmem[];  ///< dynamically allocated shared memory

///
/// @brief calculate gravitational acceleration on accelerator device
///
/// @param[in] ipos position of i-particles
/// @param[in] ivel velocity of i-particles
/// @param[out] iacc acceleration of i-particles
/// @param[out] ijrk jerk of i-particles
/// @param[in] Nj number of j-particles
/// @param[in] jpos position of j-particles
/// @param[in] jvel velocity of j-particles
/// @param[in] eps2 square of softening length
///
__global__ void calc_acc_device(
    const type::position *const ipos, const type::velocity *const ivel, type::acceleration *__restrict iacc, type::jerk *__restrict ijrk,
    const type::int_idx Nj, const type::position *const jpos, const type::velocity *const jvel, const type::fp_l eps2) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  // initialization
  const auto pi = ipos[ii];
  const auto vi = ivel[ii];
  type::acceleration ai = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};
  type::jerk ji = {AS_FLT_JRK(0.0), AS_FLT_JRK(0.0), AS_FLT_JRK(0.0)};

  //   __shared__ type::position jpos_shmem[NTHREADS];
  //   __shared__ type::velocity jvel_shmem[NTHREADS];
  type::position *jpos_shmem = (type::position *)dynamic_shmem;
  type::velocity *jvel_shmem = (type::velocity *)&jpos_shmem[NTHREADS];

  // force evaluation
  for (type::int_idx jh = 0U; jh < Nj; jh += NTHREADS) {
    // load j-particle
    __syncthreads();
    jpos_shmem[threadIdx.x] = jpos[jh + threadIdx.x];
    jvel_shmem[threadIdx.x] = jvel[jh + threadIdx.x];
    __syncthreads();

    PRAGMA_UNROLL
    for (type::int_idx jj = 0U; jj < NTHREADS; jj++) {
      const auto pj = jpos_shmem[jj];
      const auto vj = jvel_shmem[jj];

      // calculate acceleration
      const auto dx = pj.x - pi.x;
      const auto dy = pj.y - pi.y;
      const auto dz = pj.z - pi.z;
      const auto r2 = eps2 + dx * dx + dy * dy + dz * dz;
#if FP_L <= 32
      const auto r_inv = rsqrtf(type::cast2fp_l(r2));
#else   // FP_L <= 32
      // obtain reciprocal square root by using Newton--Raphson method instead of 1.0 / sqrt(r2) in FP64
      const auto seed = type::cast2fp_l(rsqrtf(static_cast<float>(r2)));
      const auto r_inv = AS_FP_L(0.5) * (AS_FP_L(3.0) - r2 * seed * seed) * seed;
#endif  // FP_L <= 32
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

      // calculate jerk
      const auto dvx = vj.x - vi.x;
      const auto dvy = vj.y - vi.y;
      const auto dvz = vj.z - vi.z;
      const auto bet = AS_FP_M(-3.0) * (dx * dvx + dy * dvy + dz * dvz) * CAST2JRK(r2_inv);  ///< 3 * beta
      // jerk accumulation
      ji.x += alp * (dvx + bet * dx);
      ji.y += alp * (dvy + bet * dy);
      ji.z += alp * (dvz + bet * dz);
    }
  }
  iacc[ii] = ai;
  ijrk[ii] = ji;
}

///
/// @brief calculate gravitational acceleration
///
/// @param[in] Ni number of i-particles
/// @param[in] ipos position of i-particles
/// @param[in] ivel velocity of i-particles
/// @param[out] iacc acceleration of i-particles
/// @param[out] ijrk jerk of i-particles
/// @param[in] Nj number of j-particles
/// @param[in] jpos position of j-particles
/// @param[in] jvel velocity of j-particles
/// @param[in] eps2 square of softening length
///
static inline void calc_acc(
    const type::int_idx Ni, const type::position *const ipos, const type::velocity *const ivel, type::acceleration *__restrict iacc, type::jerk *__restrict ijrk,
    const type::int_idx Nj, const type::position *const jpos, const type::velocity *const jvel, const type::fp_l eps2) {
  //   calc_acc_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS>>>(ipos, ivel, iacc, ijrk, Nj, jpos, jvel, eps2);
  calc_acc_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS, (sizeof(type::position) + sizeof(type::velocity)) * NTHREADS>>>(ipos, ivel, iacc, ijrk, Nj, jpos, jvel, eps2);
}

///
/// @brief correct particle position and velocity on accelerator device
///
/// @param[out] tag tag to sort SoA
///
__global__ void reset_tag_device(type::int_idx *__restrict tag) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;
  tag[ii] = ii;
}

#ifndef BENCHMARK_MODE
///
/// @brief finalize calculation of gravitational acceleration on accelerator device
///
/// @param[in,out] acc acceleration of i-particles
/// @param[in,out] jerk jerk of i-particles
/// @param[in] pos position of i-particles
/// @param[in] eps_inv inverse of softening length
///
__global__ void trim_acc_device(type::acceleration *__restrict acc, type::jerk *__restrict jerk
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

  auto ji = jerk[ii];
  ji.x *= newton;
  ji.y *= newton;
  ji.z *= newton;
  jerk[ii] = ji;
}

///
/// @brief finalize calculation of gravitational acceleration
///
/// @param[in] Ni number of i-particles
/// @param[in,out] acc acceleration of i-particles
/// @param[in,out] jerk jerk of i-particles
/// @param[in] pos position of i-particles
/// @param[in] eps_inv inverse of softening length
///
static inline void trim_acc(const type::int_idx Ni, type::acceleration *__restrict acc, type::jerk *__restrict jerk
#ifdef CALCULATE_POTENTIAL
                            ,
                            const type::position *const pos, const type::flt_acc eps_inv
#endif  // CALCULATE_POTENTIAL
) {
  trim_acc_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS>>>(acc, jerk
#ifdef CALCULATE_POTENTIAL
                                                         ,
                                                         pos, eps_inv
#endif  // CALCULATE_POTENTIAL
  );
}

///
/// @brief guess initial time step by calculating snap and crackle on accelerator device
///
/// @param[in] ipos position of i-particles
/// @param[in] ivel velocity of i-particles
/// @param[in] iacc acceleration of i-particles
/// @param[in] ijrk jerk of i-particles
/// @param[in] Nj number of j-particles
/// @param[in] jpos position of j-particles
/// @param[in] jvel velocity of j-particles
/// @param[in] jacc acceleration of j-particles
/// @param[in] jjrk jerk of j-particles
/// @param[in] eps2 square of softening length
/// @param[in] eta safety parameter to determine time step
/// @param[out] dt estimated time step
///
__global__ void guess_initial_dt_device(
    const type::position *const ipos, const type::velocity *const ivel, const type::acceleration *const iacc, const type::jerk *const ijrk,
    const type::int_idx Nj, const type::position *const jpos, const type::velocity *const jvel, const type::acceleration *const jacc, const type::jerk *const jjrk,
    const type::fp_l eps2, const type::fp_m eta, type::fp_m *__restrict dt) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  // initialization
  const auto p_i = ipos[ii];
  const auto v_i = ivel[ii];
  const auto a_i = iacc[ii];
  const auto j_i = ijrk[ii];
  auto sx = AS_FP_M(0.0);
  auto sy = AS_FP_M(0.0);
  auto sz = AS_FP_M(0.0);
  auto cx = AS_FP_M(0.0);
  auto cy = AS_FP_M(0.0);
  auto cz = AS_FP_M(0.0);

  //   __shared__ type::position jpos_shmem[NTHREADS];
  //   __shared__ type::velocity jvel_shmem[NTHREADS];
  //   __shared__ type::acceleration jacc_shmem[NTHREADS];
  //   __shared__ type::jerk jjrk_shmem[NTHREADS];
  type::position *jpos_shmem = (type::position *)dynamic_shmem;
  type::velocity *jvel_shmem = (type::velocity *)&jpos_shmem[NTHREADS];
  type::acceleration *jacc_shmem = (type::acceleration *)&jvel_shmem[NTHREADS];
  type::jerk *jjrk_shmem = (type::jerk *)&jacc_shmem[NTHREADS];

  // force evaluation
  for (type::int_idx jh = 0U; jh < Nj; jh += NTHREADS) {
    // load j-particle
    __syncthreads();
    jpos_shmem[threadIdx.x] = jpos[jh + threadIdx.x];
    jvel_shmem[threadIdx.x] = jvel[jh + threadIdx.x];
    jacc_shmem[threadIdx.x] = jacc[jh + threadIdx.x];
    jjrk_shmem[threadIdx.x] = jjrk[jh + threadIdx.x];
    __syncthreads();

    PRAGMA_UNROLL
    for (type::int_idx jj = 0U; jj < NTHREADS; jj++) {
      const auto p_j = jpos_shmem[jj];
      const auto v_j = jvel_shmem[jj];
      const auto a_j = jacc_shmem[jj];
      const auto j_j = jjrk_shmem[jj];

      // calculate acceleration
      const auto dx = p_j.x - p_i.x;
      const auto dy = p_j.y - p_i.y;
      const auto dz = p_j.z - p_i.z;
      const auto r2 = eps2 + dx * dx + dy * dy + dz * dz;
#if FP_L <= 32
      const auto r_inv = rsqrtf(type::cast2fp_l(r2));
#else   // FP_L <= 32
      // obtain reciprocal square root by using Newton--Raphson method instead of 1.0 / sqrt(r2) in FP64
      const auto seed = type::cast2fp_l(rsqrtf(static_cast<float>(r2)));
      const auto r_inv = AS_FP_L(0.5) * (AS_FP_L(3.0) - r2 * seed * seed) * seed;
#endif  // FP_L <= 32
      const auto r2_inv = type::cast2fp_m(r_inv * r_inv);
      const auto alp = p_j.w * type::cast2fp_m(r_inv) * r2_inv;

      // calculate jerk
      const auto dvx = v_j.x - v_i.x;
      const auto dvy = v_j.y - v_i.y;
      const auto dvz = v_j.z - v_i.z;
      const auto bet = -(dx * dvx + dy * dvy + dz * dvz) * r2_inv;

      // calculate snap
      const auto dax = a_j.x - a_i.x;
      const auto day = a_j.y - a_i.y;
      const auto daz = a_j.z - a_i.z;
      const auto bet2 = bet * bet;
      const auto gam = AS_FP_M(5.0) * bet2 - (dvx * dvx + dvy * dvy + dvz * dvz + dx * dax + dy * day + dz * daz) * r2_inv;
      const auto s0 = AS_FP_M(6.0) * bet;
      const auto s1 = AS_FP_M(3.0) * gam;
      sx += alp * (dax + s0 * dvx + s1 * dx);
      sy += alp * (day + s0 * dvy + s1 * dy);
      sz += alp * (daz + s0 * dvz + s1 * dz);

      // calculate crackle
      const auto djx = j_j.x - j_i.x;
      const auto djy = j_j.y - j_i.y;
      const auto djz = j_j.z - j_i.z;
      const auto del = AS_FP_M(5.0) * bet * (s1 - AS_FP_M(8.0) * bet2) - (dx * djx + dy * djy + dz * djz + AS_FP_M(3.0) * (dvx * dax + dvy * day + dvz * daz)) * r2_inv;
      const auto c0 = AS_FP_M(9.0) * bet;
      const auto c1 = AS_FP_M(9.0) * gam;
      const auto c2 = AS_FP_M(3.0) * del;
      cx += alp * (djx + c0 * dax + c1 * dvx + c2 * dx);
      cy += alp * (djy + c0 * day + c1 * dvy + c2 * dy);
      cz += alp * (djz + c0 * daz + c1 * dvz + c2 * dz);
    }
  }

  // guess initial dt at t = 0
  const auto s0x = newton * sx;
  const auto s0y = newton * sy;
  const auto s0z = newton * sz;
  const auto c0x = newton * cx;
  const auto c0y = newton * cy;
  const auto c0z = newton * cz;
  const auto j2 = j_i.x * j_i.x + j_i.y * j_i.y + j_i.z * j_i.z;
  const auto s2 = s0x * s0x + s0y * s0y + s0z * s0z;
  dt[ii] = eta * std::sqrt((j2 + std::sqrt(s2 * (a_i.x * a_i.x + a_i.y * a_i.y + a_i.z * a_i.z))) / (s2 + std::sqrt(j2 * (c0x * c0x + c0y * c0y + c0z * c0z))));
}

///
/// @brief guess initial time step by calculating snap and crackle
///
/// @param[in] Ni number of i-particles
/// @param[in] ipos position of i-particles
/// @param[in] ivel velocity of i-particles
/// @param[in] iacc acceleration of i-particles
/// @param[in] ijrk jerk of i-particles
/// @param[in] Nj number of j-particles
/// @param[in] jpos position of j-particles
/// @param[in] jvel velocity of j-particles
/// @param[in] jacc acceleration of j-particles
/// @param[in] jjrk jerk of j-particles
/// @param[in] eps2 square of softening length
/// @param[in] eta safety parameter to determine time step
/// @param[out] dt estimated time step
///
static inline void guess_initial_dt(
    const type::int_idx Ni, const type::position *const ipos, const type::velocity *const ivel, const type::acceleration *const iacc, const type::jerk *const ijrk,
    const type::int_idx Nj, const type::position *const jpos, const type::velocity *const jvel, const type::acceleration *const jacc, const type::jerk *const jjrk,
    const type::fp_l eps2, const type::fp_m eta, type::fp_m *__restrict dt) {
  //   guess_initial_dt_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS>>>(ipos, ivel, iacc, ijrk, Nj, jpos, jvel, jacc, jjrk, eps2, eta, dt);
  guess_initial_dt_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS, (sizeof(type::position) + sizeof(type::velocity) + sizeof(type::acceleration) + sizeof(type::jerk)) * NTHREADS>>>(ipos, ivel, iacc, ijrk, Nj, jpos, jvel, jacc, jjrk, eps2, eta, dt);
}

///
/// @brief predict particle position and velocity on accelerator device
///
/// @param[in] t_pres current time of j-particles
/// @param[in] pos0 current position of j-particles
/// @param[in] vel0 current velocity of j-particles
/// @param[in] acc0 current acceleration of j-particles
/// @param[in] jrk0 current jerk of j-particles
/// @param[out] pos1 predicted position of j-particles
/// @param[out] vel1 predicted velocity of j-particles
/// @param[in] t_next next time
///
__global__ void predict_device(
    const type::fp_m *const t_pres, const type::position *const pos0, const type::velocity *const vel0, const type::acceleration *const acc0, const type::jerk *const jrk0,
    type::position *__restrict pos1, type::velocity *__restrict vel1, const type::fp_m t_next) {
  const type::int_idx jj = blockIdx.x * blockDim.x + threadIdx.x;

  // set time step for this particle
  const auto dt = t_next - t_pres[jj];
  const auto dt_2 = AS_FP_M(0.5) * dt;
  const auto dt_3 = dt * boost::math::constants::third<type::fp_m>();

  // load j-particles
  const auto v0 = vel0[jj];
  const auto a0 = acc0[jj];
  const auto j0 = jrk0[jj];

  // predict particle position and velocity
  type::velocity v1;
  v1.x = v0.x + dt * (a0.x + dt_2 * j0.x);
  v1.y = v0.y + dt * (a0.y + dt_2 * j0.y);
  v1.z = v0.z + dt * (a0.z + dt_2 * j0.z);
  auto p1 = pos0[jj];
  p1.x += dt * (v0.x + dt_2 * (a0.x + dt_3 * j0.x));
  p1.y += dt * (v0.y + dt_2 * (a0.y + dt_3 * j0.y));
  p1.z += dt * (v0.z + dt_2 * (a0.z + dt_3 * j0.z));

  // store the predicted position and velocity
  pos1[jj] = p1;
  vel1[jj] = v1;
}

///
/// @brief predict particle position and velocity
///
/// @param[in] jnum number of j-particles
/// @param[in] t_pres current time of j-particles
/// @param[in] pos0 current position of j-particles
/// @param[in] vel0 current velocity of j-particles
/// @param[in] acc0 current acceleration of j-particles
/// @param[in] jrk0 current jerk of j-particles
/// @param[out] pos1 predicted position of j-particles
/// @param[out] vel1 predicted velocity of j-particles
/// @param[in] t_next next time
///
static inline void predict(
    const type::int_idx jnum, const type::fp_m *const t_pres, const type::position *const pos0, const type::velocity *const vel0, const type::acceleration *const acc0, const type::jerk *const jrk0,
    type::position *__restrict pos1, type::velocity *__restrict vel1, const type::fp_m t_next) {
  predict_device<<<BLOCKSIZE(jnum, NTHREADS), NTHREADS>>>(t_pres, pos0, vel0, acc0, jrk0, pos1, vel1, t_next);
}

///
/// @brief correct particle position and velocity on accelerator device
///
/// @param[in,out] t_pres current time of i-particles (to be corrected)
/// @param[in,out] pos0 current position of i-particles (to be corrected)
/// @param[in,out] vel0 current velocity of i-particles (to be corrected)
/// @param[in,out] acc0 current acceleration of i-particles (to be corrected)
/// @param[in,out] jrk0 current jerk of i-particles (to be corrected)
/// @param[in,out] t_next next time of i-particles (to be corrected)
/// @param[in] pos1 predicted position of i-particles
/// @param[in] vel1 predicted velocity of i-particles
/// @param[in] acc1 predicted acceleration of i-particles
/// @param[in] jrk1 predicted jerk of i-particles
/// @param[in] t1 next time
/// @param[in] eta safety parameter to determine time step
///
__global__ void correct_device(
    type::fp_m *__restrict t_pres, type::position *__restrict pos0, type::velocity *__restrict vel0, type::acceleration *__restrict acc0, type::jerk *__restrict jrk0, type::fp_m *__restrict t_next,
    const type::position *const pos1, const type::velocity *const vel1, const type::acceleration *const acc1, const type::jerk *const jrk1, const type::fp_m t1, const type::fp_m eta) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  // set time step for this particle
  const auto dt = t1 - t_pres[ii];
  t_pres[ii] = t1;
  const auto dt_inv = AS_FP_M(1.0) / dt;
  const auto dt_2 = AS_FP_M(0.5) * dt;
  const auto dt_3 = dt * boost::math::constants::third<type::fp_m>();
  const auto dt_4 = AS_FP_M(0.25) * dt;
  const auto dt_5 = AS_FP_M(0.2) * dt;
  const auto dt3_6 = dt * dt_2 * dt_3;
  const auto dt4_24 = dt3_6 * dt_4;
  const auto six_dt2 = AS_FP_M(6.0) * dt_inv * dt_inv;
  const auto twelve_dt3 = AS_FP_M(2.0) * six_dt2 * dt_inv;

  // load acceleration and jerk
  const auto a0 = acc0[ii];
  const auto a1 = acc1[ii];
  acc0[ii] = a1;
  const auto j0 = jrk0[ii];
  const auto j1 = jrk1[ii];
  jrk0[ii] = j1;

  // calculate snap
  const auto s0x = six_dt2 * ((a1.x - a0.x) - dt_3 * (j1.x + AS_FLT_JRK(2.0) * j0.x));
  const auto s0y = six_dt2 * ((a1.y - a0.y) - dt_3 * (j1.y + AS_FLT_JRK(2.0) * j0.y));
  const auto s0z = six_dt2 * ((a1.z - a0.z) - dt_3 * (j1.z + AS_FLT_JRK(2.0) * j0.z));

  // calculate crackle
  const auto c0x = twelve_dt3 * ((a0.x - a1.x) + dt_2 * (j1.x + j0.x));
  const auto c0y = twelve_dt3 * ((a0.y - a1.y) + dt_2 * (j1.y + j0.y));
  const auto c0z = twelve_dt3 * ((a0.z - a1.z) + dt_2 * (j1.z + j0.z));

  // correct velocity
  auto v1 = vel1[ii];
  v1.x += dt3_6 * (s0x + dt_4 * c0x);
  v1.y += dt3_6 * (s0y + dt_4 * c0y);
  v1.z += dt3_6 * (s0z + dt_4 * c0z);
  vel0[ii] = v1;

  // correct position
  auto p1 = pos1[ii];
  p1.x += dt4_24 * (s0x + dt_5 * c0x);
  p1.y += dt4_24 * (s0y + dt_5 * c0y);
  p1.z += dt4_24 * (s0z + dt_5 * c0z);
  pos0[ii] = p1;

  // interpolate snap
  const auto s1x = s0x + dt * c0x;
  const auto s1y = s0y + dt * c0y;
  const auto s1z = s0z + dt * c0z;

  // update time step (c1x = d0x under 3rd-order Hermite interpolation)
  const auto j2 = j1.x * j1.x + j1.y * j1.y + j1.z * j1.z;
  const auto s2 = s1x * s1x + s1y * s1y + s1z * s1z;
  t_next[ii] = t1 + eta * std::sqrt((j2 + std::sqrt(s2 * (a1.x * a1.x + a1.y * a1.y + a1.z * a1.z))) / (s2 + std::sqrt(j2 * (c0x * c0x + c0y * c0y + c0z * c0z))));
}

///
/// @brief correct particle position and velocity
///
/// @param[in] inum number of i-particles
/// @param[in,out] t_pres current time of i-particles (to be corrected)
/// @param[in,out] pos0 current position of i-particles (to be corrected)
/// @param[in,out] vel0 current velocity of i-particles (to be corrected)
/// @param[in,out] acc0 current acceleration of i-particles (to be corrected)
/// @param[in,out] jrk0 current jerk of i-particles (to be corrected)
/// @param[in,out] t_next next time of i-particles (to be corrected)
/// @param[in] pos1 predicted position of i-particles
/// @param[in] vel1 predicted velocity of i-particles
/// @param[in] acc1 predicted acceleration of i-particles
/// @param[in] jrk1 predicted jerk of i-particles
/// @param[in] t1 next time
/// @param[in] eta safety parameter to determine time step
///
static inline void correct(
    const type::int_idx inum, type::fp_m *__restrict t_pres, type::position *__restrict pos0, type::velocity *__restrict vel0, type::acceleration *__restrict acc0, type::jerk *__restrict jrk0, type::fp_m *__restrict t_next,
    const type::position *const pos1, const type::velocity *const vel1, const type::acceleration *const acc1, const type::jerk *const jrk1, const type::fp_m t1, const type::fp_m eta) {
  correct_device<<<BLOCKSIZE(inum, NTHREADS), NTHREADS>>>(t_pres, pos0, vel0, acc0, jrk0, t_next, pos1, vel1, acc1, jrk1, t1, eta);
}

///
/// @brief sort N-body particles on accelerator device
///
/// @param[in] src_pos position of N-body particles (source array)
/// @param[in] src_vel velocity of N-body particles (source array)
/// @param[in] src_acc acceleration of N-body particles (source array)
/// @param[in] src_jrk jerk of N-body particles (source array)
/// @param[in] src_prs present time of N-body particles (source array)
/// @param[in] src_idx index of N-body particles (source array)
/// @param[out] dst_pos position of N-body particles (destination array)
/// @param[out] dst_vel velocity of N-body particles (destination array)
/// @param[out] dst_acc acceleration of N-body particles (destination array)
/// @param[out] dst_jrk jerk of N-body particles (destination array)
/// @param[out] dst_prs present time of N-body particles (destination array)
/// @param[out] dst_idx index of N-body particles (destination array)
/// @param[in] tag tentative array to sort N-body particles
///
__global__ void sort_particles_device(
    const type::position *const src_pos, const type::velocity *const src_vel, const type::acceleration *const src_acc, const type::jerk *const src_jrk, const type::fp_m *const src_prs, const type::int_idx *const src_idx,
    type::position *__restrict dst_pos, type::velocity *__restrict dst_vel, type::acceleration *__restrict dst_acc, type::jerk *__restrict dst_jrk, type::fp_m *__restrict dst_prs, type::int_idx *__restrict dst_idx,
    const type::int_idx *const tag) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;
  const auto jj = tag[ii];
  dst_pos[ii] = src_pos[jj];
  dst_vel[ii] = src_vel[jj];
  dst_acc[ii] = src_acc[jj];
  dst_jrk[ii] = src_jrk[jj];
  dst_prs[ii] = src_prs[jj];
  // dst_nxt[ii] = src_nxt[jj];
  dst_idx[ii] = src_idx[jj];
}

///
/// @brief sort N-body particles
///
/// @param[in] num total number of N-body particles
/// @param[out] tag0_dev tentative array to sort N-body particles
/// @param[out] tag1_dev tentative array to sort N-body particles
/// @param[in] iacc acceleration of i-particles
/// @param[in,out] src input N-body particles (to be sorted)
/// @param[in] dst tentative SoA of N-body particles
/// @param[in] temp_storage tentative storage for CUB
/// @param[in] temp_storage_size size of tentative storage for CUB
///
static inline void sort_particles(const type::int_idx num, type::int_idx *__restrict tag0_dev, type::int_idx *__restrict tag1_dev, type::nbody *__restrict src, type::nbody *__restrict dst, void *temp_storage, size_t &temp_storage_size) {
  // sort particle time
  reset_tag_device<<<BLOCKSIZE(num, NTHREADS), NTHREADS>>>(tag0_dev);
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, (*src).nxt, (*dst).nxt, tag0_dev, tag1_dev, num);

  // sort N-body particles
  sort_particles_device<<<BLOCKSIZE(num, NTHREADS), NTHREADS>>>((*src).pos, (*src).vel, (*src).acc, (*src).jrk, (*src).prs, (*src).idx, (*dst).pos, (*dst).vel, (*dst).acc, (*dst).jrk, (*dst).prs, (*dst).idx, tag1_dev);

  // swap SoAs
  const auto _tmp = *src;
  *src = *dst;
  *dst = _tmp;
}

///
/// @brief set next time of N-body particles on accelerator device
///
/// @param[out] next next time of N-body particles
/// @param[in] time_next updated next time
///
__global__ void set_next_time_device(type::fp_m *__restrict next, const type::fp_m time_next) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;
  next[ii] = time_next;
}

///
/// @brief set block time step
///
/// @param[in] num total number of N-body particles
/// @param[in,out] body_dev N-body particles (device memory)
/// @param[in] time_pres current time
/// @param[in] time_sync time when all N-body particles to be synchronized
/// @param[in,out] body N-body particles (host memory)
///
/// @return number of i-particles and time after the orbit integration
///
static inline auto set_time_step(const type::int_idx num, type::nbody &body, const type::fp_m time_pres, const type::fp_m time_sync) {
  cudaDeviceSynchronize();  // complete the computation on GPU before reading results from CPU
  auto Ni = num;
  const auto time_next = std::min(time_sync, time_pres + std::exp2(std::floor(std::log2(body.nxt[0] - time_pres))));
  if (time_next < time_sync) {
    // adopt block time step
    const auto next_time_next = time_pres + std::exp2(AS_FP_M(1.0) + std::floor(std::log2(time_next - time_pres)));
    for (type::int_idx ii = NTHREADS; ii < num; ii += NTHREADS) {
      if (body.nxt[ii] >= next_time_next) {
        Ni = ii;
        break;
      }
    }
  }

  set_next_time_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS>>>(body.nxt, time_next);

  return (std::make_pair(Ni, time_next));
}

///
/// @brief reset particle time on accelerator device
///
/// @param[out] pres present time of N-body particles
/// @param[in,out] next next time of N-body particles
/// @param[in] snapshot_interval current time
///
__global__ void reset_particle_time_device(type::fp_m *__restrict pres, type::fp_m *__restrict next, const type::fp_m snapshot_interval) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;
  pres[ii] = AS_FP_M(0.0);
  next[ii] -= snapshot_interval;
}

///
/// @brief reset particle time
///
/// @param[in] num total number of N-body particles
/// @param[in,out] body N-body particles
/// @param[in] snapshot_interval current time
///
static inline void reset_particle_time(const type::int_idx num, type::nbody &body, const type::fp_m snapshot_interval) {
  reset_particle_time_device<<<BLOCKSIZE(num, NTHREADS), NTHREADS>>>(body.prs, body.nxt, snapshot_interval);
}
#endif  // BENCHMARK_MODE

///
/// @brief allocate memory for N-body particles
///
/// @param[out] body0 SoA for N-body particles
/// @param[out] pos0 position of N-body particles
/// @param[out] vel0 velocity of N-body particles
/// @param[out] acc0 acceleration of N-body particles
/// @param[out] jrk0 jerk of N-body particles
/// @param[out] prs0 present time of N-body particles
/// @param[out] nxt0 next time of N-body particles
/// @param[out] idx0 ID of N-body particles
/// @param[out] body1 SoA for N-body particles
/// @param[out] pos1 position of N-body particles
/// @param[out] vel1 velocity of N-body particles
/// @param[out] acc1 acceleration of N-body particles
/// @param[out] jrk1 jerk of N-body particles
/// @param[out] prs1 present time of N-body particles
/// @param[out] nxt1 next time of N-body particles
/// @param[out] idx1 ID of N-body particles
/// @param[out] tag0 tag to sort N-body particles
/// @param[out] tag1 tag to sort N-body particles
/// @param[out] temp_storage temporary storage for sorting on accelerator device (device memory)
/// @param[out] temp_storage_size size of temporary storage for sorting on accelerator device
/// @param[in] num number of N-body particles
///
static inline void allocate_Nbody_particles(
    type::nbody &body0, type::position **pos0, type::velocity **vel0, type::acceleration **acc0, type::jerk **jrk0, type::fp_m **prs0, type::fp_m **nxt0, type::int_idx **idx0,
    type::nbody &body1, type::position **pos1, type::velocity **vel1, type::acceleration **acc1, type::jerk **jrk1, type::fp_m **prs1, type::fp_m **nxt1, type::int_idx **idx1,
    type::int_idx **tag0, type::int_idx **tag1, void **temp_storage, size_t &temp_storage_size, const type::int_idx num) {
  auto size = static_cast<size_t>(num);
  if ((num % NTHREADS) != 0U) {
    size += static_cast<size_t>(NTHREADS - (num % NTHREADS));
  }
  cudaMallocManaged((void **)pos0, size * sizeof(type::position));
  cudaMallocManaged((void **)pos1, size * sizeof(type::position));
  cudaMallocManaged((void **)vel0, size * sizeof(type::velocity));
  cudaMallocManaged((void **)vel1, size * sizeof(type::velocity));
  cudaMallocManaged((void **)acc0, size * sizeof(type::acceleration));
  cudaMallocManaged((void **)acc1, size * sizeof(type::acceleration));
  cudaMallocManaged((void **)jrk0, size * sizeof(type::jerk));
  cudaMallocManaged((void **)jrk1, size * sizeof(type::jerk));
  cudaMallocManaged((void **)prs0, size * sizeof(type::fp_m));
  cudaMallocManaged((void **)prs1, size * sizeof(type::fp_m));
  cudaMallocManaged((void **)nxt0, size * sizeof(type::fp_m));
  cudaMallocManaged((void **)nxt1, size * sizeof(type::fp_m));
  cudaMallocManaged((void **)idx0, size * sizeof(type::int_idx));
  cudaMallocManaged((void **)idx1, size * sizeof(type::int_idx));

  // initialize arrays (for safety of massless particles and first touch)
  constexpr type::position p_zero = {AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0)};
  constexpr type::velocity v_zero = {AS_FLT_VEL(0.0), AS_FLT_VEL(0.0), AS_FLT_VEL(0.0)};
  constexpr type::acceleration a_zero = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};
  constexpr type::jerk j_zero = {AS_FLT_JRK(0.0), AS_FLT_JRK(0.0), AS_FLT_JRK(0.0)};
#pragma omp parallel for
  for (size_t ii = 0U; ii < size; ii++) {
    (*pos0)[ii] = p_zero;
    (*pos1)[ii] = p_zero;
    (*vel0)[ii] = v_zero;
    (*vel1)[ii] = v_zero;
    (*acc0)[ii] = a_zero;
    (*acc1)[ii] = a_zero;
    (*jrk0)[ii] = j_zero;
    (*jrk1)[ii] = j_zero;
    (*prs0)[ii] = AS_FP_M(0.0);
    (*prs1)[ii] = AS_FP_M(0.0);
    (*nxt0)[ii] = std::numeric_limits<type::fp_m>::max();
    (*nxt1)[ii] = std::numeric_limits<type::fp_m>::max();
    (*idx0)[ii] = std::numeric_limits<type::int_idx>::max();
    (*idx1)[ii] = std::numeric_limits<type::int_idx>::max();
  }

  // memory allocation for sorting
  cudaMalloc((void **)tag0, size * sizeof(type::int_idx));
  cudaMalloc((void **)tag1, size * sizeof(type::int_idx));
  temp_storage_size = 0U;
  *temp_storage = NULL;
  reset_tag_device<<<BLOCKSIZE(size, NTHREADS), NTHREADS>>>(*tag0);
  reset_tag_device<<<BLOCKSIZE(size, NTHREADS), NTHREADS>>>(*tag1);
  cub::DeviceRadixSort::SortPairs(*temp_storage, temp_storage_size, *nxt0, *nxt1, *tag0, *tag1, size);
  cudaMalloc(temp_storage, temp_storage_size);

  // assign SoA
  body0.pos = *pos0;
  body0.vel = *vel0;
  body0.acc = *acc0;
  body0.jrk = *jrk0;
  body0.prs = *prs0;
  body0.nxt = *nxt0;
  body0.idx = *idx0;
  body1.pos = *pos1;
  body1.vel = *vel1;
  body1.acc = *acc1;
  body1.jrk = *jrk1;
  body1.prs = *prs1;
  body1.nxt = *nxt1;
  body1.idx = *idx1;
}

///
/// @brief deallocate memory for N-body particles
///
/// @param[out] pos0 position of N-body particles
/// @param[out] vel0 velocity of N-body particles
/// @param[out] acc0 acceleration of N-body particles
/// @param[out] jrk0 jerk of N-body particles
/// @param[out] prs0 present time of N-body particles
/// @param[out] nxt0 next time of N-body particles
/// @param[out] idx0 ID of N-body particles
/// @param[out] pos1 position of N-body particles
/// @param[out] vel1 velocity of N-body particles
/// @param[out] acc1 acceleration of N-body particles
/// @param[out] jrk1 jerk of N-body particles
/// @param[out] prs1 present time of N-body particles
/// @param[out] nxt1 next time of N-body particles
/// @param[out] idx1 ID of N-body particles
/// @param[out] tag0 tag to sort N-body particles
/// @param[out] tag1 tag to sort N-body particles
/// @param[out] temp_storage temporary storage for sorting on accelerator device
///
static inline void release_Nbody_particles(
    type::position *pos0, type::velocity *vel0, type::acceleration *acc0, type::jerk *jrk0, type::fp_m *prs0, type::fp_m *nxt0, type::int_idx *idx0,
    type::position *pos1, type::velocity *vel1, type::acceleration *acc1, type::jerk *jrk1, type::fp_m *prs1, type::fp_m *nxt1, type::int_idx *idx1,
    type::int_idx *tag0, type::int_idx *tag1, void *temp_storage) {
  cudaFree(pos0);
  cudaFree(pos1);
  cudaFree(vel0);
  cudaFree(vel1);
  cudaFree(acc0);
  cudaFree(acc1);
  cudaFree(jrk0);
  cudaFree(jrk1);
  cudaFree(prs0);
  cudaFree(prs1);
  cudaFree(nxt0);
  cudaFree(nxt1);
  cudaFree(idx0);
  cudaFree(idx1);
  cudaFree(tag0);
  cudaFree(tag1);
  cudaFree(temp_storage);
}

///
/// @brief configure the target GPU device if necessary
///
static inline void configure_gpu() {
  // obtain available capacity of shared memory
  cudaDeviceProp prop;
  int32_t id;
  cudaGetDevice(&id);
  cudaGetDeviceProperties(&prop, id);

  // estimate required capacity of shared memory
  int32_t N_block;
  //   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&N_block, calc_acc_device, NTHREADS, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&N_block, calc_acc_device, NTHREADS, (sizeof(type::position) + sizeof(type::velocity)) * NTHREADS);
  cudaFuncSetAttribute(calc_acc_device, cudaFuncAttributePreferredSharedMemoryCarveout, static_cast<int32_t>(std::ceil(static_cast<float>((sizeof(type::position) + sizeof(type::velocity)) * NTHREADS * N_block) / static_cast<float>(prop.sharedMemPerBlock))));
#ifndef BENCHMARK_MODE
  //   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&N_block, guess_initial_dt_device, NTHREADS, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&N_block, guess_initial_dt_device, NTHREADS, (sizeof(type::position) + sizeof(type::velocity) + sizeof(type::acceleration) + sizeof(type::jerk)) * NTHREADS);
  cudaFuncSetAttribute(guess_initial_dt_device, cudaFuncAttributePreferredSharedMemoryCarveout, static_cast<int32_t>(std::ceil(static_cast<float>((sizeof(type::position) + sizeof(type::velocity) + sizeof(type::acceleration) + sizeof(type::jerk)) * NTHREADS * N_block) / static_cast<float>(prop.sharedMemPerBlock))));
#endif  // BENCHMARK_MODE

  // below functions do not utilize the shared memory, so maximize the capacity of the L1 cache (0% for shared memory)
  cudaFuncSetAttribute(reset_tag_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
#ifndef BENCHMARK_MODE
  cudaFuncSetAttribute(trim_acc_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(predict_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(correct_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(sort_particles_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(set_next_time_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(reset_particle_time_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
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
  const auto eta = cfg.get_eta();
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
  const auto num_logbin = std::log2(num_max / num_min) / static_cast<double>(num_bin - 1);
  for (int32_t ii = 0; ii < num_bin; ii++) {
    const auto num = static_cast<type::int_idx>(std::nearbyint(num_min * std::exp2(static_cast<double>(ii) * num_logbin)));
#endif  // BENCHMARK_MODE

    // memory allocation
    alignas(MEMORY_ALIGNMENT) type::position *pos0, *pos1;
    alignas(MEMORY_ALIGNMENT) type::velocity *vel0, *vel1;
    alignas(MEMORY_ALIGNMENT) type::acceleration *acc0, *acc1;
    alignas(MEMORY_ALIGNMENT) type::jerk *jerk0, *jerk1;
    alignas(MEMORY_ALIGNMENT) type::fp_m *pres0, *pres1;
    alignas(MEMORY_ALIGNMENT) type::fp_m *next0, *next1;
    alignas(MEMORY_ALIGNMENT) type::int_idx *id0, *id1;
    alignas(MEMORY_ALIGNMENT) type::int_idx *tag0, *tag1;
    alignas(MEMORY_ALIGNMENT) void *temp_storage;
    size_t temp_storage_size = 0U;
    auto body0 = type::nbody();
    auto body1 = type::nbody();
    allocate_Nbody_particles(
        body0, &pos0, &vel0, &acc0, &jerk0, &pres0, &next0, &id0,
        body1, &pos1, &vel1, &acc1, &jerk1, &pres1, &next1, &id1, &tag0, &tag1, &temp_storage, temp_storage_size, num);

    // generate initial-condition
    init::set_uniform_sphere(num, body0.pos, body0.vel, M_tot, rad, virial, CAST2VEL(newton));
#pragma omp parallel for
    for (type::int_idx ii = 0U; ii < num; ii++) {
      body0.idx[ii] = ii;
    }

#ifndef BENCHMARK_MODE
    // write the first snapshot
    auto Ni = num;
    auto N_pairs = static_cast<double>(Ni) * static_cast<double>(num);
    calc_acc(Ni, body0.pos, body0.vel, body0.acc, body0.jrk, num, body0.pos, body0.vel, eps2);
    trim_acc(Ni, body0.acc, body0.jrk
#ifdef CALCULATE_POTENTIAL
             ,
             body0.pos, eps_inv
#endif  // CALCULATE_POTENTIAL
    );
    cudaDeviceSynchronize();  // complete the computation on GPU before reading results from CPU
    auto error = conservatives();
    io::write_snapshot(num, body0.pos, body0.vel, body0.acc, body0.jrk, body0.idx, file.c_str(), present, time, error);
    guess_initial_dt(num, body0.pos, body0.vel, body0.acc, body0.jrk, num, body0.pos, body0.vel, body0.acc, body0.jrk, eps2, eta, body0.nxt);

    while (present < snp_fin) {
      step++;

      sort_particles(num, tag0, tag1, &body0, &body1, temp_storage, temp_storage_size);
      std::tie(Ni, time_from_snapshot) = set_time_step(num, body0, time_from_snapshot, snapshot_interval);
      if (time_from_snapshot >= snapshot_interval) {
        present++;
        time_from_snapshot = snapshot_interval;
      }
      DEBUG_PRINT(std::cout, "t = " << time + time_from_snapshot << ", step = " << step)

      // orbit integration (4th-order Hermite scheme)
      N_pairs += static_cast<double>(Ni) * static_cast<double>(num);
      predict(num, body0.prs, body0.pos, body0.vel, body0.acc, body0.jrk, body1.pos, body1.vel, time_from_snapshot);
      calc_acc(Ni, body1.pos, body1.vel, body1.acc, body1.jrk, num, body1.pos, body1.vel, eps2);
      trim_acc(Ni, body1.acc, body1.jrk
#ifdef CALCULATE_POTENTIAL
               ,
               body1.pos, eps_inv
#endif  // CALCULATE_POTENTIAL
      );
      correct(Ni, body0.prs, body0.pos, body0.vel, body0.acc, body0.jrk, body0.nxt, body1.pos, body1.vel, body1.acc, body1.jrk, time_from_snapshot, eta);

      // write snapshot
      if (present > previous) {
        previous = present;
        time_from_snapshot = AS_FP_M(0.0);
        time += snapshot_interval;
        cudaDeviceSynchronize();  // complete the computation on GPU before reading results from CPU
        io::write_snapshot(num, body0.pos, body0.vel, body0.acc, body0.jrk, body0.idx, file.c_str(), present, time, error);
        reset_particle_time(num, body0, snapshot_interval);
      }
    }
#else   // BENCHMARK_MODE
  // launch benchmark
  auto timer = util::timer();
  cudaDeviceSynchronize();  // complete the computation on GPU before reading results from CPU
  timer.start();
  calc_acc(num, body0.pos, body0.vel, body0.acc, body0.jrk, num, body0.pos, body0.vel, eps2);
  cudaDeviceSynchronize();  // complete the computation on GPU before reading results from CPU
  timer.stop();
  auto elapsed = timer.get_elapsed_wall();
  int32_t iter = 1;

  // increase iteration counts if the measured time is too short
  const auto minimum_elapsed = cfg.get_minimum_elapsed_time();
  while (elapsed < minimum_elapsed) {
    // predict the iteration counts
    constexpr double booster = 1.25;  ///< additional safety-parameter to reduce rejection rate
    iter = static_cast<int32_t>(std::exp2(std::ceil(std::log2(static_cast<double>(iter) * booster * minimum_elapsed / elapsed))));

    // re-execute the benchmark
    timer.clear();
    cudaDeviceSynchronize();  // complete the computation on GPU before reading results from CPU
    timer.start();
    for (int32_t loop = 0; loop < iter; loop++) {
      calc_acc(num, body0.pos, body0.vel, body0.acc, body0.jrk, num, body0.pos, body0.vel, eps2);
    }
    cudaDeviceSynchronize();  // complete the computation on GPU before reading results from CPU
    timer.stop();
    elapsed = timer.get_elapsed_wall();
  }

  // finalize benchmark
  io::write_log(argv[0], elapsed, iter, num, file.c_str());
#endif  // BENCHMARK_MODE

    // memory deallocation
    release_Nbody_particles(
        pos0, vel0, acc0, jerk0, pres0, next0, id0,
        pos1, vel1, acc1, jerk1, pres1, next1, id1, tag0, tag1, temp_storage);

#ifdef BENCHMARK_MODE
  }
#else  // BENCHMARK_MODE
  util::hdf5::remove_datatype_vec3();
  util::hdf5::remove_datatype_vec4();

  time_to_solution.stop();
  io::write_log(argv[0], time_to_solution.get_elapsed_wall(), step, num, file.c_str()
#ifdef CALCULATE_POTENTIAL
                                                                             ,
                eta, error
#endif  // CALCULATE_POTENTIAL
                ,
                N_pairs);
#endif  // BENCHMARK_MODE

  return (0);
}
