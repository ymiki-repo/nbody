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
    const type::int_idx Nj, const type::position *const jpos, const type::velocity *const jvel, const type::flt_pos eps2) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  // initialization
  const auto pi = ipos[ii];
  const auto vi = ivel[ii];
  type::acceleration ai = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};
  type::jerk ji = {AS_FLT_JRK(0.0), AS_FLT_JRK(0.0), AS_FLT_JRK(0.0)};

  // force evaluation
  for (type::int_idx jj = 0U; jj < Nj; jj++) {
    // load j-particle
    const auto pj = jpos[jj];
    const auto vj = jvel[jj];

    // calculate acceleration
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
    const type::int_idx Nj, const type::position *const jpos, const type::velocity *const jvel, const type::flt_pos eps2) {
  calc_acc_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS>>>(ipos, ivel, iacc, ijrk, Nj, jpos, jvel, eps2);
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
    const type::flt_pos eps2, const type::fp_m eta, type::fp_m *__restrict dt) {
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

  // force evaluation
  for (type::int_idx jj = 0U; jj < Nj; jj++) {
    // load j-particle
    const auto p_j = jpos[jj];
    const auto v_j = jvel[jj];
    const auto a_j = jacc[jj];
    const auto j_j = jjrk[jj];

    // calculate acceleration
    const auto dx = p_j.x - p_i.x;
    const auto dy = p_j.y - p_i.y;
    const auto dz = p_j.z - p_i.z;
    const auto r2 = eps2 + dx * dx + dy * dy + dz * dz;
    const auto r_inv = AS_FP_L(1.0) / std::sqrt(type::cast2fp_l(r2));
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
    const type::flt_pos eps2, const type::fp_m eta, type::fp_m *__restrict dt) {
  guess_initial_dt_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS>>>(ipos, ivel, iacc, ijrk, Nj, jpos, jvel, jacc, jjrk, eps2, eta, dt);
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
static inline auto set_time_step(const type::int_idx num, type::nbody &body_dev, const type::fp_m time_pres, const type::fp_m time_sync, type::nbody &body) {
  // NOTE: porting below for-loop to GPU and then memcpy time_next and Ni from device to host will be faster (and simple); below is an example using cudaMemcpy2D()
  // copy nxt[0], nxt[NTHREADS], nxt[2 * NTHREADS], nxt[3 * NTHREADS], ...
  const type::int_idx N_grp = BLOCKSIZE(num, NTHREADS);
  cudaMemcpy2D(body.nxt, sizeof(type::fp_m), body_dev.nxt, sizeof(type::fp_m) * NTHREADS, sizeof(type::fp_m), N_grp, cudaMemcpyDeviceToHost);
  auto Ni = num;
  const auto time_next = std::min(time_sync, time_pres + std::exp2(std::floor(std::log2(body.nxt[0] - time_pres))));
  if (time_next < time_sync) {
    // adopt block time step
    const auto next_time_next = time_pres + std::exp2(AS_FP_M(1.0) + std::floor(std::log2(time_next - time_pres)));
    for (type::int_idx ii = 1U; ii < N_grp; ii++) {
      if (body.nxt[ii] >= next_time_next) {
        Ni = ii * NTHREADS;
        break;
      }
    }
  }

  set_next_time_device<<<BLOCKSIZE(Ni, NTHREADS), NTHREADS>>>(body_dev.nxt, time_next);

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

__global__ void clear_particles_device(type::position *__restrict pos, type::velocity *__restrict vel, type::acceleration *__restrict acc, type::jerk *__restrict jrk, type::fp_m *__restrict prs, type::fp_m *__restrict nxt, type::int_idx *__restrict idx) {
  const type::int_idx ii = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr type::position p_zero = {AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0)};
  constexpr type::velocity v_zero = {AS_FLT_VEL(0.0), AS_FLT_VEL(0.0), AS_FLT_VEL(0.0)};
  constexpr type::acceleration a_zero = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};
  constexpr type::jerk j_zero = {AS_FLT_JRK(0.0), AS_FLT_JRK(0.0), AS_FLT_JRK(0.0)};

  pos[ii] = p_zero;
  vel[ii] = v_zero;
  acc[ii] = a_zero;
  jrk[ii] = j_zero;
  prs[ii] = AS_FP_M(0.0);
  nxt[ii] = std::numeric_limits<type::fp_m>::max();
  idx[ii] = std::numeric_limits<type::int_idx>::max();
}

///
/// @brief allocate memory for N-body particles
///
/// @param[out] body_hst SoA for N-body particles (host memory)
/// @param[out] pos_hst position of N-body particles (host memory)
/// @param[out] vel_hst velocity of N-body particles (host memory)
/// @param[out] acc_hst acceleration of N-body particles (host memory)
/// @param[out] jrk_hst jerk of N-body particles (host memory)
/// @param[out] nxt_hst next time of N-body particles (host memory)
/// @param[out] idx_hst ID of N-body particles (host memory)
/// @param[out] body0_dev SoA for N-body particles (device memory)
/// @param[out] pos0_dev position of N-body particles (device memory)
/// @param[out] vel0_dev velocity of N-body particles (device memory)
/// @param[out] acc0_dev acceleration of N-body particles (device memory)
/// @param[out] jrk0_dev jerk of N-body particles (device memory)
/// @param[out] prs0_dev present time of N-body particles (device memory)
/// @param[out] nxt0_dev next time of N-body particles (device memory)
/// @param[out] idx0_dev ID of N-body particles (device memory)
/// @param[out] body1_dev SoA for N-body particles (device memory)
/// @param[out] pos1_dev position of N-body particles (device memory)
/// @param[out] vel1_dev velocity of N-body particles (device memory)
/// @param[out] acc1_dev acceleration of N-body particles (device memory)
/// @param[out] jrk1_dev jerk of N-body particles (device memory)
/// @param[out] prs1_dev present time of N-body particles (device memory)
/// @param[out] nxt1_dev next time of N-body particles (device memory)
/// @param[out] idx1_dev ID of N-body particles (device memory)
/// @param[out] tag0_dev tag to sort N-body particles (device memory)
/// @param[out] tag0_dev tag to sort N-body particles (device memory)
/// @param[out] temp_storage temporary storage for sorting on accelerator device (device memory)
/// @param[out] temp_storage_size size of temporary storage for sorting on accelerator device
/// @param[in] num number of N-body particles
///
static inline void allocate_Nbody_particles(
    type::nbody &body_hst, type::position **pos_hst, type::velocity **vel_hst, type::acceleration **acc_hst, type::jerk **jrk_hst, type::fp_m **nxt_hst, type::int_idx **idx_hst,
    type::nbody &body0_dev, type::position **pos0_dev, type::velocity **vel0_dev, type::acceleration **acc0_dev, type::jerk **jrk0_dev, type::fp_m **prs0_dev, type::fp_m **nxt0_dev, type::int_idx **idx0_dev,
    type::nbody &body1_dev, type::position **pos1_dev, type::velocity **vel1_dev, type::acceleration **acc1_dev, type::jerk **jrk1_dev, type::fp_m **prs1_dev, type::fp_m **nxt1_dev, type::int_idx **idx1_dev,
    type::int_idx **tag0_dev, type::int_idx **tag1_dev, void **temp_storage, size_t &temp_storage_size, const type::int_idx num) {
  auto size = static_cast<size_t>(num);
  if ((num % NTHREADS) != 0U) {
    size += static_cast<size_t>(NTHREADS - (num % NTHREADS));
  }
  cudaMallocHost((void **)pos_hst, size * sizeof(type::position));
  cudaMallocHost((void **)vel_hst, size * sizeof(type::velocity));
  cudaMallocHost((void **)acc_hst, size * sizeof(type::acceleration));
  cudaMallocHost((void **)jrk_hst, size * sizeof(type::jerk));
  cudaMallocHost((void **)nxt_hst, (size / NTHREADS) * sizeof(type::fp_m));
  cudaMallocHost((void **)idx_hst, size * sizeof(type::int_idx));
  cudaMalloc((void **)pos0_dev, size * sizeof(type::position));
  cudaMalloc((void **)vel0_dev, size * sizeof(type::velocity));
  cudaMalloc((void **)acc0_dev, size * sizeof(type::acceleration));
  cudaMalloc((void **)jrk0_dev, size * sizeof(type::jerk));
  cudaMalloc((void **)prs0_dev, size * sizeof(type::fp_m));
  cudaMalloc((void **)nxt0_dev, size * sizeof(type::fp_m));
  cudaMalloc((void **)idx0_dev, size * sizeof(type::int_idx));
  cudaMalloc((void **)pos1_dev, size * sizeof(type::position));
  cudaMalloc((void **)vel1_dev, size * sizeof(type::velocity));
  cudaMalloc((void **)acc1_dev, size * sizeof(type::acceleration));
  cudaMalloc((void **)jrk1_dev, size * sizeof(type::jerk));
  cudaMalloc((void **)prs1_dev, size * sizeof(type::fp_m));
  cudaMalloc((void **)nxt1_dev, size * sizeof(type::fp_m));
  cudaMalloc((void **)idx1_dev, size * sizeof(type::int_idx));

  // initialize arrays (for safety of massless particles and first touch)
  constexpr type::position p_zero = {AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0)};
  constexpr type::velocity v_zero = {AS_FLT_VEL(0.0), AS_FLT_VEL(0.0), AS_FLT_VEL(0.0)};
  constexpr type::acceleration a_zero = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};
  constexpr type::jerk j_zero = {AS_FLT_JRK(0.0), AS_FLT_JRK(0.0), AS_FLT_JRK(0.0)};
#pragma omp parallel for
  for (size_t ii = 0U; ii < size; ii++) {
    (*pos_hst)[ii] = p_zero;
    (*vel_hst)[ii] = v_zero;
    (*acc_hst)[ii] = a_zero;
    (*jrk_hst)[ii] = j_zero;
    (*idx_hst)[ii] = std::numeric_limits<type::int_idx>::max();
  }
#pragma omp parallel for
  for (size_t ii = 0U; ii < (size / NTHREADS); ii++) {
    (*nxt_hst)[ii] = std::numeric_limits<type::fp_m>::max();
  }
  clear_particles_device<<<BLOCKSIZE(size, NTHREADS), NTHREADS>>>(*pos0_dev, *vel0_dev, *acc0_dev, *jrk0_dev, *prs0_dev, *nxt0_dev, *idx0_dev);
  clear_particles_device<<<BLOCKSIZE(size, NTHREADS), NTHREADS>>>(*pos1_dev, *vel1_dev, *acc1_dev, *jrk1_dev, *prs1_dev, *nxt1_dev, *idx1_dev);

  // memory allocation for sorting
  cudaMalloc((void **)tag0_dev, size * sizeof(type::int_idx));
  cudaMalloc((void **)tag1_dev, size * sizeof(type::int_idx));
  temp_storage_size = 0U;
  *temp_storage = NULL;
  reset_tag_device<<<BLOCKSIZE(size, NTHREADS), NTHREADS>>>(*tag0_dev);
  reset_tag_device<<<BLOCKSIZE(size, NTHREADS), NTHREADS>>>(*tag1_dev);
  cub::DeviceRadixSort::SortPairs(*temp_storage, temp_storage_size, *nxt0_dev, *nxt1_dev, *tag0_dev, *tag1_dev, size);
  cudaMalloc(temp_storage, temp_storage_size);

  // assign SoA
  body_hst.pos = *pos_hst;
  body_hst.vel = *vel_hst;
  body_hst.acc = *acc_hst;
  body_hst.jrk = *jrk_hst;
  body_hst.nxt = *nxt_hst;
  body_hst.idx = *idx_hst;
  body0_dev.pos = *pos0_dev;
  body0_dev.vel = *vel0_dev;
  body0_dev.acc = *acc0_dev;
  body0_dev.jrk = *jrk0_dev;
  body0_dev.prs = *prs0_dev;
  body0_dev.nxt = *nxt0_dev;
  body0_dev.idx = *idx0_dev;
  body1_dev.pos = *pos1_dev;
  body1_dev.vel = *vel1_dev;
  body1_dev.acc = *acc1_dev;
  body1_dev.jrk = *jrk1_dev;
  body1_dev.prs = *prs1_dev;
  body1_dev.nxt = *nxt1_dev;
  body1_dev.idx = *idx1_dev;
}

///
/// @brief deallocate memory for N-body particles
///
/// @param[out] pos_hst position of N-body particles
/// @param[out] vel_hst velocity of N-body particles
/// @param[out] acc_hst acceleration of N-body particles
/// @param[out] jrk_hst jerk of N-body particles
/// @param[out] nxt_hst next time of N-body particles
/// @param[out] idx_hst ID of N-body particles
/// @param[out] pos_hst position of N-body particles
/// @param[out] vel0_dev velocity of N-body particles
/// @param[out] acc0_dev acceleration of N-body particles
/// @param[out] jrk0_dev jerk of N-body particles
/// @param[out] prs0_dev present time of N-body particles
/// @param[out] nxt0_dev next time of N-body particles
/// @param[out] idx0_dev ID of N-body particles
/// @param[out] pos1_dev position of N-body particles
/// @param[out] vel1_dev velocity of N-body particles
/// @param[out] acc1_dev acceleration of N-body particles
/// @param[out] jrk1_dev jerk of N-body particles
/// @param[out] prs1_dev present time of N-body particles
/// @param[out] nxt1_dev next time of N-body particles
/// @param[out] idx1_dev ID of N-body particles
/// @param[out] tag0_dev tag to sort N-body particles
/// @param[out] tag1_dev tag to sort N-body particles
/// @param[out] temp_storage temporary storage for sorting on accelerator device
///
static inline void release_Nbody_particles(
    type::position *pos_hst, type::velocity *vel_hst, type::acceleration *acc_hst, type::jerk *jrk_hst, type::fp_m *nxt_hst, type::int_idx *idx_hst,
    type::position *pos0_dev, type::velocity *vel0_dev, type::acceleration *acc0_dev, type::jerk *jrk0_dev, type::fp_m *prs0_dev, type::fp_m *nxt0_dev, type::int_idx *idx0_dev,
    type::position *pos1_dev, type::velocity *vel1_dev, type::acceleration *acc1_dev, type::jerk *jrk1_dev, type::fp_m *prs1_dev, type::fp_m *nxt1_dev, type::int_idx *idx1_dev,
    type::int_idx *tag0_dev, type::int_idx *tag1_dev, void *temp_storage) {
  cudaFreeHost(pos_hst);
  cudaFreeHost(vel_hst);
  cudaFreeHost(acc_hst);
  cudaFreeHost(jrk_hst);
  cudaFreeHost(nxt_hst);
  cudaFreeHost(idx_hst);
  cudaFree(pos0_dev);
  cudaFree(vel0_dev);
  cudaFree(acc0_dev);
  cudaFree(jrk0_dev);
  cudaFree(prs0_dev);
  cudaFree(nxt0_dev);
  cudaFree(idx0_dev);
  cudaFree(pos1_dev);
  cudaFree(vel1_dev);
  cudaFree(acc1_dev);
  cudaFree(jrk1_dev);
  cudaFree(prs1_dev);
  cudaFree(nxt1_dev);
  cudaFree(idx1_dev);
  cudaFree(tag0_dev);
  cudaFree(tag1_dev);
  cudaFree(temp_storage);
}

///
/// @brief configure the target GPU device if necessary
///
static inline void configure_gpu() {
  // this sample code does not utilize the shared memory, so maximize the capacity of the L1 cache (0% for shared memory)
  cudaFuncSetAttribute(calc_acc_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(reset_tag_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
#ifndef BENCHMARK_MODE
  cudaFuncSetAttribute(trim_acc_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(guess_initial_dt_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(predict_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(correct_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(sort_particles_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(set_next_time_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(reset_particle_time_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
#endif  // BENCHMARK_MODE
  cudaFuncSetAttribute(clear_particles_device, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
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
    alignas(MEMORY_ALIGNMENT) type::position *pos, *pos0_dev, *pos1_dev;
    alignas(MEMORY_ALIGNMENT) type::velocity *vel, *vel0_dev, *vel1_dev;
    alignas(MEMORY_ALIGNMENT) type::acceleration *acc, *acc0_dev, *acc1_dev;
    alignas(MEMORY_ALIGNMENT) type::jerk *jerk, *jerk0_dev, *jerk1_dev;
    alignas(MEMORY_ALIGNMENT) type::fp_m *pres0_dev, *pres1_dev;
    alignas(MEMORY_ALIGNMENT) type::fp_m *next, *next0_dev, *next1_dev;
    alignas(MEMORY_ALIGNMENT) type::int_idx *id, *id0_dev, *id1_dev;
    alignas(MEMORY_ALIGNMENT) type::int_idx *tag0_dev, *tag1_dev;
    alignas(MEMORY_ALIGNMENT) void *temp_storage;
    size_t temp_storage_size = 0U;
    auto body = type::nbody();
    auto body0_dev = type::nbody();
    auto body1_dev = type::nbody();
    allocate_Nbody_particles(
        body, &pos, &vel, &acc, &jerk, &next, &id,
        body0_dev, &pos0_dev, &vel0_dev, &acc0_dev, &jerk0_dev, &pres0_dev, &next0_dev, &id0_dev,
        body1_dev, &pos1_dev, &vel1_dev, &acc1_dev, &jerk1_dev, &pres1_dev, &next1_dev, &id1_dev,
        &tag0_dev, &tag1_dev, &temp_storage, temp_storage_size, num);

    // generate initial-condition
    init::set_uniform_sphere(num, body.pos, body.vel, M_tot, rad, virial, CAST2VEL(newton));
#pragma omp parallel for
    for (type::int_idx ii = 0U; ii < num; ii++) {
      body.idx[ii] = ii;
    }
    cudaMemcpy(body0_dev.pos, body.pos, num * sizeof(type::position), cudaMemcpyHostToDevice);
    cudaMemcpy(body0_dev.vel, body.vel, num * sizeof(type::velocity), cudaMemcpyHostToDevice);
    cudaMemcpy(body0_dev.idx, body.idx, num * sizeof(type::int_idx), cudaMemcpyHostToDevice);

#ifndef BENCHMARK_MODE
    // write the first snapshot
    auto Ni = num;
    auto N_pairs = static_cast<double>(Ni) * static_cast<double>(num);
    calc_acc(Ni, body0_dev.pos, body0_dev.vel, body0_dev.acc, body0_dev.jrk, num, body0_dev.pos, body0_dev.vel, eps2);
    trim_acc(Ni, body0_dev.acc, body0_dev.jrk
#ifdef CALCULATE_POTENTIAL
             ,
             body0_dev.pos, eps_inv
#endif  // CALCULATE_POTENTIAL
    );
    auto error = conservatives();
    cudaMemcpy(body.acc, body0_dev.acc, num * sizeof(type::acceleration), cudaMemcpyDeviceToHost);
    cudaMemcpy(body.jrk, body0_dev.jrk, num * sizeof(type::jerk), cudaMemcpyDeviceToHost);
    io::write_snapshot(num, body.pos, body.vel, body.acc, body.jrk, body.idx, file.c_str(), present, time, error);
    guess_initial_dt(num, body0_dev.pos, body0_dev.vel, body0_dev.acc, body0_dev.jrk, num, body0_dev.pos, body0_dev.vel, body0_dev.acc, body0_dev.jrk, eps2, eta, body0_dev.nxt);

    while (present < snp_fin) {
      step++;

      sort_particles(num, tag0_dev, tag1_dev, &body0_dev, &body1_dev, temp_storage, temp_storage_size);
      std::tie(Ni, time_from_snapshot) = set_time_step(num, body0_dev, time_from_snapshot, snapshot_interval, body);
      if (time_from_snapshot >= snapshot_interval) {
        present++;
        time_from_snapshot = snapshot_interval;
      }
      DEBUG_PRINT(std::cout, "t = " << time + time_from_snapshot << ", step = " << step)

      // orbit integration (4th-order Hermite scheme)
      N_pairs += static_cast<double>(Ni) * static_cast<double>(num);
      predict(num, body0_dev.prs, body0_dev.pos, body0_dev.vel, body0_dev.acc, body0_dev.jrk, body1_dev.pos, body1_dev.vel, time_from_snapshot);
      calc_acc(Ni, body1_dev.pos, body1_dev.vel, body1_dev.acc, body1_dev.jrk, num, body1_dev.pos, body1_dev.vel, eps2);
      trim_acc(Ni, body1_dev.acc, body1_dev.jrk
#ifdef CALCULATE_POTENTIAL
               ,
               body1_dev.pos, eps_inv
#endif  // CALCULATE_POTENTIAL
      );
      correct(Ni, body0_dev.prs, body0_dev.pos, body0_dev.vel, body0_dev.acc, body0_dev.jrk, body0_dev.nxt, body1_dev.pos, body1_dev.vel, body1_dev.acc, body1_dev.jrk, time_from_snapshot, eta);

      // write snapshot
      if (present > previous) {
        previous = present;
        time_from_snapshot = AS_FP_M(0.0);
        time += snapshot_interval;
        cudaMemcpy(body.pos, body0_dev.pos, num * sizeof(type::position), cudaMemcpyDeviceToHost);
        cudaMemcpy(body.vel, body0_dev.vel, num * sizeof(type::velocity), cudaMemcpyDeviceToHost);
        cudaMemcpy(body.acc, body0_dev.acc, num * sizeof(type::acceleration), cudaMemcpyDeviceToHost);
        cudaMemcpy(body.jrk, body0_dev.jrk, num * sizeof(type::jerk), cudaMemcpyDeviceToHost);
        cudaMemcpy(body.idx, body0_dev.idx, num * sizeof(type::int_idx), cudaMemcpyDeviceToHost);
        io::write_snapshot(num, body.pos, body.vel, body.acc, body.jrk, body.idx, file.c_str(), present, time, error);
        reset_particle_time(num, body0_dev, snapshot_interval);
      }
    }
#else   // BENCHMARK_MODE
  // launch benchmark
  auto timer = util::timer();
  cudaDeviceSynchronize();  // complete the calculation on GPU before reading results from CPU
  timer.start();
  calc_acc(num, body0_dev.pos, body0_dev.vel, body0_dev.acc, body0_dev.jrk, num, body0_dev.pos, body0_dev.vel, eps2);
  cudaDeviceSynchronize();  // complete the calculation on GPU before reading results from CPU
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
    cudaDeviceSynchronize();  // complete the calculation on GPU before reading results from CPU
    timer.start();
    for (int32_t loop = 0; loop < iter; loop++) {
      calc_acc(num, body0_dev.pos, body0_dev.vel, body0_dev.acc, body0_dev.jrk, num, body0_dev.pos, body0_dev.vel, eps2);
    }
    cudaDeviceSynchronize();  // complete the calculation on GPU before reading results from CPU
    timer.stop();
    elapsed = timer.get_elapsed_wall();
  }

  // finalize benchmark
  io::write_log(argv[0], elapsed, iter, num, file.c_str());
#endif  // BENCHMARK_MODE

    // memory deallocation
    release_Nbody_particles(
        pos, vel, acc, jerk, next, id,
        pos0_dev, vel0_dev, acc0_dev, jerk0_dev, pres0_dev, next0_dev, id0_dev,
        pos1_dev, vel1_dev, acc1_dev, jerk1_dev, pres1_dev, next1_dev, id1_dev,
        tag0_dev, tag1_dev, temp_storage);

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
