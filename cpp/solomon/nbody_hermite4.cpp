///
/// @file nbody_hermite4.cpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief sample implementation of direct N-body simulation (orbit integration: 4th-order Hermite scheme)
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

constexpr type::flt_acc newton = AS_FLT_ACC(1.0);  // gravitational constant

#ifndef NTHREADS
constexpr type::int_idx NTHREADS = 128U;
#endif  // NTHREADS

// FIXME: remove the below macro (port as much as function to GPU)
#define EXEC_SMALL_FUNC_ON_HOST

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
  OFFLOAD(AS_INDEPENDENT, NUM_THREADS(NTHREADS))
  for (std::remove_const_t<decltype(Ni)> ii = 0U; ii < Ni; ii++) {
    // initialization
    const auto pi = ipos[ii];
    const auto vi = ivel[ii];
    std::remove_reference_t<decltype(*iacc)> ai = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};
    std::remove_reference_t<decltype(*ijrk)> ji = {AS_FLT_JRK(0.0), AS_FLT_JRK(0.0), AS_FLT_JRK(0.0)};

    // force evaluation
    PRAGMA_ACC_LOOP(ACC_CLAUSE_SEQ)
    for (std::remove_const_t<decltype(Nj)> jj = 0U; jj < Nj; jj++) {
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
      const auto bet = AS_FLT_JRK(-3.0) * (dx * dvx + dy * dvy + dz * dvz) * CAST2JRK(r2_inv);  // 3 * beta
      // jerk accumulation
      ji.x += alp * (dvx + bet * dx);
      ji.y += alp * (dvy + bet * dy);
      ji.z += alp * (dvz + bet * dz);
    }
    iacc[ii] = ai;
    ijrk[ii] = ji;
  }
}

#ifndef BENCHMARK_MODE
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

    auto ji = jerk[ii];
    ji.x *= newton;
    ji.y *= newton;
    ji.z *= newton;
    jerk[ii] = ji;
  }
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
  OFFLOAD(AS_INDEPENDENT, NUM_THREADS(NTHREADS))
  for (std::remove_const_t<decltype(Ni)> ii = 0U; ii < Ni; ii++) {
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
    PRAGMA_ACC_LOOP(ACC_CLAUSE_SEQ)
    for (std::remove_const_t<decltype(Nj)> jj = 0U; jj < Nj; jj++) {
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
  OFFLOAD(AS_INDEPENDENT)
  for (std::remove_const_t<decltype(jnum)> jj = 0U; jj < jnum; jj++) {
    // set time step for this particle
    const auto dt = t_next - t_pres[jj];
    const auto dt_2 = AS_FP_M(0.5) * dt;
    const auto dt_3 = dt * boost::math::constants::third<type::fp_m>();

    // load j-particles
    const auto v0 = vel0[jj];
    const auto a0 = acc0[jj];
    const auto j0 = jrk0[jj];

    // predict particle position and velocity
    auto v1 = std::remove_reference_t<decltype(*vel1)>{};
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
  OFFLOAD(AS_INDEPENDENT)
  for (std::remove_const_t<decltype(inum)> ii = 0U; ii < inum; ii++) {
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
}

///
/// @brief sort N-body particles
///
/// @param[in] num total number of N-body particles
/// @param tag tentative array to sort N-body particles
/// @param[in] iacc acceleration of i-particles
/// @param[in,out] src input N-body particles (to be sorted)
/// @param[in] dst tentative SoA of N-body particles
///
static inline void sort_particles(const type::int_idx num, type::int_idx *__restrict tag, type::nbody *__restrict src, type::nbody *__restrict dst) {
  // sort particle time
  MEMCPY_D2H(src->nxt [0:num])
  std::iota(tag, tag + num, 0U);
  std::sort(tag, tag + num, [src](auto ii, auto jj) { return ((*src).nxt[ii] < (*src).nxt[jj]); });

  // sort N-body particles
#ifndef EXEC_SMALL_FUNC_ON_HOST
  MEMCPY_H2D(tag [0:num])
  OFFLOAD(AS_INDEPENDENT)
#else   // EXEC_SMALL_FUNC_ON_HOST
  MEMCPY_D2H(src->pos [0:num], src->vel [0:num], src->acc [0:num], src->jrk [0:num], src->prs [0:num], src->idx [0:num])
#endif  // EXEC_SMALL_FUNC_ON_HOST
  for (std::remove_const_t<decltype(num)> ii = 0U; ii < num; ii++) {
    const auto jj = tag[ii];

    (*dst).pos[ii] = (*src).pos[jj];
    (*dst).vel[ii] = (*src).vel[jj];
    (*dst).acc[ii] = (*src).acc[jj];
    (*dst).jrk[ii] = (*src).jrk[jj];
    (*dst).prs[ii] = (*src).prs[jj];
    (*dst).nxt[ii] = (*src).nxt[jj];
    (*dst).idx[ii] = (*src).idx[jj];
  }
#ifdef EXEC_SMALL_FUNC_ON_HOST
  MEMCPY_H2D(dst->pos [0:num], dst->vel [0:num], dst->acc [0:num], dst->jrk [0:num], dst->prs [0:num], dst->nxt [0:num], dst->idx [0:num])
#endif  // EXEC_SMALL_FUNC_ON_HOST

  // swap SoAs
  const auto _tmp = *src;
  *src = *dst;
  *dst = _tmp;
}

///
/// @brief set block time step
///
/// @param[in] num total number of N-body particles
/// @param[in,out] body N-body particles
/// @param[in] time_pres current time
/// @param[in] time_sync time when all N-body particles to be synchronized
///
/// @return number of i-particles and time after the orbit integration
///
static inline auto set_time_step(const type::int_idx num, type::nbody &body, const type::fp_m time_pres, const type::fp_m time_sync) {
  auto Ni = num;
  MEMCPY_D2H(body.nxt [0:num])
  const auto time_next = std::min(time_sync, time_pres + std::exp2(std::floor(std::log2(body.nxt[0] - time_pres))));
  if (time_next < time_sync) {
    // adopt block time step
    const auto next_time_next = time_pres + std::exp2(AS_FP_M(1.0) + std::floor(std::log2(time_next - time_pres)));
    for (std::remove_const_t<decltype(num)> ii = NTHREADS; ii < num; ii += NTHREADS) {
      if (body.nxt[ii] >= next_time_next) {
        Ni = ii;
        break;
      }
    }
  }

  // unify the next time within the integrated block
#ifndef EXEC_SMALL_FUNC_ON_HOST
  OFFLOAD(AS_INDEPENDENT)
#endif  // EXEC_SMALL_FUNC_ON_HOST
  for (std::remove_const_t<decltype(Ni)> ii = 0U; ii < Ni; ii++) {
    body.nxt[ii] = time_next;
  }
#ifdef EXEC_SMALL_FUNC_ON_HOST
  MEMCPY_H2D(body.nxt [0:num])
#endif  // EXEC_SMALL_FUNC_ON_HOST

  return (std::make_pair(Ni, time_next));
}

///
/// @brief reset particle time
///
/// @param[in] num total number of N-body particles
/// @param[in,out] body N-body particles
/// @param[in] snapshot_interval current time
///
static inline void reset_particle_time(const type::int_idx num, type::nbody &body, const type::fp_m snapshot_interval) {
#ifndef EXEC_SMALL_FUNC_ON_HOST
  OFFLOAD(AS_INDEPENDENT)
#else   // EXEC_SMALL_FUNC_ON_HOST
  MEMCPY_D2H(body.nxt [0:num])
#endif  // EXEC_SMALL_FUNC_ON_HOST
  for (std::remove_const_t<decltype(num)> ii = 0U; ii < num; ii++) {
    body.prs[ii] = AS_FP_M(0.0);
    body.nxt[ii] -= snapshot_interval;
  }
#ifdef EXEC_SMALL_FUNC_ON_HOST
  MEMCPY_H2D(body.prs [0:num], body.nxt [0:num])
#endif  // EXEC_SMALL_FUNC_ON_HOST
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
/// @param[out] tag tag to sort N-body particles
/// @param[in] num number of N-body particles
///
static inline void allocate_Nbody_particles(
    type::nbody &body0, type::position **pos0, type::velocity **vel0, type::acceleration **acc0, type::jerk **jrk0, type::fp_m **prs0, type::fp_m **nxt0, type::int_idx **idx0,
    type::nbody &body1, type::position **pos1, type::velocity **vel1, type::acceleration **acc1, type::jerk **jrk1, type::fp_m **prs1, type::fp_m **nxt1, type::int_idx **idx1,
    type::int_idx **tag, const type::int_idx num) {
  const auto size = static_cast<size_t>(num);
  *pos0 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**pos0)>[size];
  *pos1 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**pos1)>[size];
  *vel0 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**vel0)>[size];
  *vel1 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**vel1)>[size];
  *acc0 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**acc0)>[size];
  *acc1 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**acc1)>[size];
  *jrk0 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**jrk0)>[size];
  *jrk1 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**jrk1)>[size];
  *prs0 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**prs0)>[size];
  *prs1 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**prs1)>[size];
  *nxt0 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**nxt0)>[size];
  *nxt1 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**nxt1)>[size];
  *idx0 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**idx0)>[size];
  *idx1 = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**idx1)>[size];
  *tag = new (std::align_val_t{MEMORY_ALIGNMENT}) std::remove_reference_t<decltype(**tag)>[size];

  // initialize arrays (for safety of massless particles and first touch)
  constexpr std::remove_reference_t<decltype(**pos0)> p_zero = {AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0), AS_FLT_POS(0.0)};
  constexpr std::remove_reference_t<decltype(**vel0)> v_zero = {AS_FLT_VEL(0.0), AS_FLT_VEL(0.0), AS_FLT_VEL(0.0)};
  constexpr std::remove_reference_t<decltype(**acc0)> a_zero = {AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0), AS_FLT_ACC(0.0)};
  constexpr std::remove_reference_t<decltype(**jrk0)> j_zero = {AS_FLT_JRK(0.0), AS_FLT_JRK(0.0), AS_FLT_JRK(0.0)};
#pragma omp parallel for
  for (std::remove_const_t<decltype(size)> ii = 0U; ii < size; ii++) {
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
    (*nxt0)[ii] = std::numeric_limits<std::remove_reference_t<decltype(**nxt0)>>::max();
    (*nxt1)[ii] = std::numeric_limits<std::remove_reference_t<decltype(**nxt1)>>::max();
    (*idx0)[ii] = std::numeric_limits<std::remove_reference_t<decltype(**idx0)>>::max();
    (*idx1)[ii] = std::numeric_limits<std::remove_reference_t<decltype(**idx1)>>::max();
    (*tag)[ii] = std::numeric_limits<std::remove_reference_t<decltype(**tag)>>::max();
  }

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
/// @param[out] tag tag to sort N-body particles
///
static inline void release_Nbody_particles(
    type::position *pos0, type::velocity *vel0, type::acceleration *acc0, type::jerk *jrk0, type::fp_m *prs0, type::fp_m *nxt0, type::int_idx *idx0,
    type::position *pos1, type::velocity *vel1, type::acceleration *acc1, type::jerk *jrk1, type::fp_m *prs1, type::fp_m *nxt1, type::int_idx *idx1,
    type::int_idx *tag) {
  delete[] pos0;
  delete[] pos1;
  delete[] vel0;
  delete[] vel1;
  delete[] acc0;
  delete[] acc1;
  delete[] jrk0;
  delete[] jrk1;
  delete[] prs0;
  delete[] prs1;
  delete[] nxt0;
  delete[] nxt1;
  delete[] idx0;
  delete[] idx1;
  delete[] tag;
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

#ifdef BENCHMARK_MODE
  const auto num_logbin = std::log2(num_max / num_min) / static_cast<double>((num_bin > 1) ? (num_bin - 1) : (num_bin));
  for (std::remove_const_t<decltype(num_bin)> ii = 0; ii < num_bin; ii++) {
    const auto num = static_cast<type::int_idx>(std::nearbyint(num_min * std::exp2(static_cast<double>(ii) * num_logbin)));
#endif  // BENCHMARK_MODE

    // memory allocation
    type::position *pos0 = nullptr;
    type::position *pos1 = nullptr;
    type::velocity *vel0 = nullptr;
    type::velocity *vel1 = nullptr;
    type::acceleration *acc0 = nullptr;
    type::acceleration *acc1 = nullptr;
    type::jerk *jerk0 = nullptr;
    type::jerk *jerk1 = nullptr;
    type::fp_m *pres0 = nullptr;
    type::fp_m *pres1 = nullptr;
    type::fp_m *next0 = nullptr;
    type::fp_m *next1 = nullptr;
    type::int_idx *id0 = nullptr;
    type::int_idx *id1 = nullptr;
    type::int_idx *tag = nullptr;
    auto body0 = type::nbody{};
    auto body1 = type::nbody{};
    allocate_Nbody_particles(
        body0, &pos0, &vel0, &acc0, &jerk0, &pres0, &next0, &id0,
        body1, &pos1, &vel1, &acc1, &jerk1, &pres1, &next1, &id1, &tag, num);

    MALLOC_ON_DEVICE(body0.pos [0:num], body0.vel [0:num], body0.acc [0:num], body0.jrk [0:num], body0.prs [0:num], body0.nxt [0:num], body0.idx [0:num])
    MALLOC_ON_DEVICE(body1.pos [0:num], body1.vel [0:num], body1.acc [0:num], body1.jrk [0:num], body1.prs [0:num], body1.nxt [0:num], body1.idx [0:num])
    MALLOC_ON_DEVICE(tag [0:num])

    // generate initial-condition
    init::set_uniform_sphere(num, body0.pos, body0.vel, M_tot, rad, virial, CAST2VEL(newton));
#pragma omp parallel for
    for (std::remove_const_t<decltype(num)> ii = 0U; ii < num; ii++) {
      body0.idx[ii] = ii;
    }
    MEMCPY_H2D(body0.pos [0:num], body0.vel [0:num], body0.idx [0:num])

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
    MEMCPY_D2H(body0.acc [0:num], body0.jrk [0:num])
    auto error = conservatives();
    io::write_snapshot(num, body0.pos, body0.vel, body0.acc, body0.jrk, body0.idx, file.c_str(), present, time, error);
    guess_initial_dt(num, body0.pos, body0.vel, body0.acc, body0.jrk, num, body0.pos, body0.vel, body0.acc, body0.jrk, eps2, eta, body0.nxt);

    while (present < snp_fin) {
      step++;

      sort_particles(num, tag, &body0, &body1);
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
        MEMCPY_D2H(body0.pos [0:num], body0.vel [0:num], body0.acc [0:num], body0.jrk [0:num], body0.idx [0:num])
        io::write_snapshot(num, body0.pos, body0.vel, body0.acc, body0.jrk, body0.idx, file.c_str(), present, time, error);
        reset_particle_time(num, body0, snapshot_interval);
      }
    }
#else   // BENCHMARK_MODE
  // launch benchmark
  auto timer = util::timer();
  timer.start();
  calc_acc(num, body0.pos, body0.vel, body0.acc, body0.jrk, num, body0.pos, body0.vel, eps2);
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
      calc_acc(num, body0.pos, body0.vel, body0.acc, body0.jrk, num, body0.pos, body0.vel, eps2);
    }
    timer.stop();
    elapsed = timer.get_elapsed_wall();
  }

  // finalize benchmark
  io::write_log(argv[0], elapsed, iter, num, file.c_str());
#endif  // BENCHMARK_MODE

    // memory deallocation
    FREE_FROM_DEVICE(body0.pos [0:num], body0.vel [0:num], body0.acc [0:num], body0.jrk [0:num], body0.prs [0:num], body0.nxt [0:num], body0.idx [0:num])
    FREE_FROM_DEVICE(body1.pos [0:num], body1.vel [0:num], body1.acc [0:num], body1.jrk [0:num], body1.prs [0:num], body1.nxt [0:num], body1.idx [0:num])
    FREE_FROM_DEVICE(tag [0:num])
    release_Nbody_particles(
        pos0, vel0, acc0, jerk0, pres0, next0, id0,
        pos1, vel1, acc1, jerk1, pres1, next1, id1, tag);

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
