///
/// @file common/init.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief initial-condition generator for direct N-body simulation
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef COMMON_INIT_HPP
#define COMMON_INIT_HPP

#include <random>  ///< std::random_device

#include "common/type.hpp"

namespace init {
///
/// @brief set initial condition (uniform sphere)
///
/// @param[in] num number of N-body particles
/// @param[out] pos position of N-body particles
/// @param[out] vel velocity of N-body particles
/// @param[in] Mtot total mass of the system
/// @param[in] rad radius of the initial sphere
/// @param[in] virial virial ratio of the initial condition
/// @param[in] newton gravitational constant in the computational unit
///
static inline void set_uniform_sphere(const type::int_idx num, type::position *pos, type::velocity *vel, const type::flt_pos Mtot, const type::flt_pos rad, const type::flt_vel virial, const type::flt_vel newton = AS_FLT_VEL(1.0)) {
  // set velocity dispersion
  const auto sigma = std::sqrt(AS_FLT_VEL(1.2) * newton * CAST2VEL(Mtot) * virial / CAST2VEL(rad));
  const auto sig1d = sigma / std::sqrt(AS_FLT_VEL(3.0));

  // prepare pseudo random numbers
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution<type::flt_pos> dist_uni(AS_FLT_POS(0.0), AS_FLT_POS(1.0));
  std::normal_distribution<type::flt_vel> dist_nrm(AS_FLT_VEL(0.0), AS_FLT_VEL(1.0));

  // distribute N-body particles
  for (type::int_idx ii = 0U; ii < num; ii++) {
    const auto mass = (ii < num) ? (Mtot / static_cast<type::flt_pos>(num)) : AS_FLT_POS(0.0);  // set massless particle to remove if statements in gravity calculation

    const auto rr = rad * std::cbrt(dist_uni(engine));
    const auto prj = AS_FLT_POS(2.0) * dist_uni(engine) - AS_FLT_POS(1.0);
    const auto RR = rr * std::sqrt(AS_FLT_POS(1.0) - prj * prj);
    const auto theta = boost::math::constants::two_pi<type::flt_pos>() * dist_uni(engine);
    type::position pi;
    pi.x = RR * std::cos(theta);
    pi.y = RR * std::sin(theta);
    pi.z = rr * prj;
    pi.w = mass;

    // set particle velocity
    type::velocity vi;
    vi.x = sig1d * dist_nrm(engine);
    vi.y = sig1d * dist_nrm(engine);
    vi.z = sig1d * dist_nrm(engine);

    pos[ii] = pi;
    vel[ii] = vi;
  }

  // shift center-of-mass and remove bulk velocity
  auto cx = AS_FLT_POS(0.0);
  auto cy = AS_FLT_POS(0.0);
  auto cz = AS_FLT_POS(0.0);
  auto vx = AS_FLT_VEL(0.0);
  auto vy = AS_FLT_VEL(0.0);
  auto vz = AS_FLT_VEL(0.0);
#pragma omp parallel for reduction(+ \
                                   : cx, cy, cz, vx, vy, vz)
  for (type::int_idx ii = 0U; ii < num; ii++) {
    // evaluate center-of-mass
    const auto pi = pos[ii];
    cx += pi.w * pi.x;
    cy += pi.w * pi.y;
    cz += pi.w * pi.z;

    // evaluate bulk velocity
    const auto vi = vel[ii];
    vx += CAST2VEL(pi.w) * vi.x;
    vy += CAST2VEL(pi.w) * vi.y;
    vz += CAST2VEL(pi.w) * vi.z;
  }
  const auto M_inv = AS_FLT_POS(1.0) / Mtot;
  cx *= M_inv;
  cy *= M_inv;
  cz *= M_inv;
  vx *= CAST2VEL(M_inv);
  vy *= CAST2VEL(M_inv);
  vz *= CAST2VEL(M_inv);
#pragma omp parallel for
  for (type::int_idx ii = 0U; ii < num; ii++) {
    auto pi = pos[ii];
    pi.x -= cx;
    pi.y -= cy;
    pi.z -= cz;
    pos[ii] = pi;
    auto vi = vel[ii];
    vi.x -= vx;
    vi.y -= vy;
    vi.z -= vz;
    vel[ii] = vi;
  }
}

}  // namespace init

#endif  // COMMON_INIT_HPP
