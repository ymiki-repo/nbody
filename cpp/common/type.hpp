///
/// @file common/type.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief datatype for direct N-body simulation
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef COMMON_TYPE_HPP
#define COMMON_TYPE_HPP

#include <cstdint>  // uint??_t

#include "util/type.hpp"  // util::type::vec[3 4]

///
/// @brief number of bits for floating-point numbers in low-precision arithmetics (Type_low)
///
#ifndef FP_L
#define FP_L (32)
#endif  // FP_L

///
/// @brief number of bits for floating-point numbers in medium-precision arithmetics (Type_calc)
///
#ifndef FP_M
#define FP_M (64)
#endif  // FP_M

///
/// @brief number of bits for floating-point numbers in high-precision arithmetics (Type_high)
///
#ifndef FP_H
#define FP_H (128)
#endif  // FP_H

///
/// @brief datatype
///
namespace type {
#if FP_L == 32
using fp_l = float;  // use float as fp_l (floating-point numbers in low-precision arithmetics)
///
/// @brief append appropriate prefix for fp_l
///
#define AS_FP_L(val) (val##F)
///
/// @brief cast to fp_l
///
/// @param[in] input input
/// @return constexpr fp_l
///
static inline constexpr auto cast2fp_l(const float input) -> fp_l { return (input); }
///
/// @brief cast to fp_l
///
/// @param[in] input input
/// @return constexpr fp_l
///
static inline constexpr auto cast2fp_l(const double input) -> fp_l { return (static_cast<fp_l>(input)); }
///
/// @brief cast to fp_l
///
/// @param[in] input input
/// @return constexpr fp_l
///
static inline constexpr auto cast2fp_l(const long double input) -> fp_l { return (static_cast<fp_l>(input)); }
#elif FP_L == 64
using fp_l = double;  // use double as fp_l (floating-point numbers in low-precision arithmetics)
///
/// @brief append appropriate prefix for fp_l
///
#define AS_FP_L(val) (val)
///
/// @brief cast to fp_l
///
/// @param[in] input input
/// @return constexpr fp_l
///
static inline constexpr auto cast2fp_l(const float input) -> fp_l { return (static_cast<fp_l>(input)); }
///
/// @brief cast to fp_l
///
/// @param[in] input input
/// @return constexpr fp_l
///
static inline constexpr auto cast2fp_l(const double input) -> fp_l { return (input); }
///
/// @brief cast to fp_l
///
/// @param[in] input input
/// @return constexpr fp_l
///
static inline constexpr auto cast2fp_l(const long double input) -> fp_l { return (static_cast<fp_l>(input)); }
#else  // FP_L == 64
using fp_l = long double;  // use long double as fp_l (floating-point numbers in low-precision arithmetics)
///
/// @brief append appropriate prefix for fp_l
///
#define AS_FP_L(val) (val##L)
///
/// @brief cast to fp_l
///
/// @param[in] input input
/// @return constexpr fp_l
///
static inline constexpr auto cast2fp_l(const float input) -> fp_l { return (static_cast<fp_l>(input)); }
///
/// @brief cast to fp_l
///
/// @param[in] input input
/// @return constexpr fp_l
///
static inline constexpr auto cast2fp_l(const double input) -> fp_l { return (static_cast<fp_l>(input)); }
///
/// @brief cast to fp_l
///
/// @param[in] input input
/// @return constexpr fp_l
///
static inline constexpr auto cast2fp_l(const long double input) -> fp_l { return (input); }
#endif  // FP_L == 64

#if FP_M == 32
using fp_m = float;  // use float as fp_m (floating-point numbers in medium-precision arithmetics)
///
/// @brief append appropriate prefix for fp_m
///
#define AS_FP_M(val) (val##F)
///
/// @brief cast to fp_m
///
/// @param[in] input input
/// @return constexpr fp_m
///
static inline constexpr auto cast2fp_m(const float input) -> fp_m { return (input); }
///
/// @brief cast to fp_m
///
/// @param[in] input input
/// @return constexpr fp_m
///
static inline constexpr auto cast2fp_m(const double input) -> fp_m { return (static_cast<fp_m>(input)); }
///
/// @brief cast to fp_m
///
/// @param[in] input input
/// @return constexpr fp_m
///
static inline constexpr auto cast2fp_m(const long double input) -> fp_m { return (static_cast<fp_m>(input)); }
#elif FP_M == 64
using fp_m = double;  // use double as fp_m (floating-point numbers in medium-precision arithmetics)
///
/// @brief append appropriate prefix for fp_m
///
#define AS_FP_M(val) (val)
///
/// @brief cast to fp_m
///
/// @param[in] input input
/// @return constexpr fp_m
///
static inline constexpr auto cast2fp_m(const float input) -> fp_m { return (static_cast<fp_m>(input)); }
///
/// @brief cast to fp_m
///
/// @param[in] input input
/// @return constexpr fp_m
///
static inline constexpr auto cast2fp_m(const double input) -> fp_m { return (input); }
///
/// @brief cast to fp_m
///
/// @param[in] input input
/// @return constexpr fp_m
///
static inline constexpr auto cast2fp_m(const long double input) -> fp_m { return (static_cast<fp_m>(input)); }
#else  // FP_M
using fp_m = long double;  // use long double as fp_m (floating-point numbers in medium-precision arithmetics)
///
/// @brief append appropriate prefix for fp_m
///
#define AS_FP_M(val) (val##L)
///
/// @brief cast to fp_m
///
/// @param[in] input input
/// @return constexpr fp_m
///
static inline constexpr auto cast2fp_m(const float input) -> fp_m { return (static_cast<fp_m>(input)); }
///
/// @brief cast to fp_m
///
/// @param[in] input input
/// @return constexpr fp_m
///
static inline constexpr auto cast2fp_m(const double input) -> fp_m { return (static_cast<fp_m>(input)); }
///
/// @brief cast to fp_m
///
/// @param[in] input input
/// @return constexpr fp_m
///
static inline constexpr auto cast2fp_m(const long double input) -> fp_m { return (input); }
#endif  // FP_M

#if FP_H == 64
using fp_h = double;  // use double as fp_h (floating-point numbers in high-precision arithmetics)
///
/// @brief append appropriate prefix for fp_h
///
#define AS_FP_H(val) (val)
///
/// @brief cast to fp_h
///
/// @param[in] input input
/// @return constexpr fp_h
///
static inline constexpr auto cast2fp_h(const float input) -> fp_h { return (static_cast<fp_h>(input)); }
///
/// @brief cast to fp_h
///
/// @param[in] input input
/// @return constexpr fp_h
///
static inline constexpr auto cast2fp_h(const double input) -> fp_h { return (input); }
///
/// @brief cast to fp_h
///
/// @param[in] input input
/// @return constexpr fp_h
///
static inline constexpr auto cast2fp_h(const long double input) -> fp_h { return (static_cast<fp_h>(input)); }
#else  // FP_H
using fp_h = long double;  // use long double as fp_h (floating-point numbers in high-precision arithmetics)
///
/// @brief append appropriate prefix for fp_h
///
#define AS_FP_H(val) (val##L)
///
/// @brief cast to fp_h
///
/// @param[in] input input
/// @return constexpr fp_h
///
static inline constexpr auto cast2fp_h(const float input) -> fp_h { return (static_cast<fp_h>(input)); }
///
/// @brief cast to fp_h
///
/// @param[in] input input
/// @return constexpr fp_h
///
static inline constexpr auto cast2fp_h(const double input) -> fp_h { return (static_cast<fp_h>(input)); }
///
/// @brief cast to fp_h
///
/// @param[in] input input
/// @return constexpr fp_h
///
static inline constexpr auto cast2fp_h(const long double input) -> fp_h { return (input); }
#endif  // FP_H

using flt_pos = fp_m;  // use fp_m for the location
///
/// @brief append appropriate prefix for flt_pos
///
#define AS_FLT_POS(val) AS_FP_M(val)
///
/// @brief minimal cast to flt_pos
///
#define CAST2POS(val) (type::cast2fp_m(val))

using flt_vel = fp_m;  // use fp_m for the velocity
///
/// @brief append appropriate prefix for flt_vel
///
#define AS_FLT_VEL(val) AS_FP_M(val)
///
/// @brief minimal cast to flt_vel
///
#define CAST2VEL(val) (type::cast2fp_m(val))

using flt_acc = fp_m;  // use fp_m for the acceleration
///
/// @brief append appropriate prefix for flt_acc
///
#define AS_FLT_ACC(val) AS_FP_M(val)
///
/// @brief minimal cast to flt_acc
///
#define CAST2ACC(val) (type::cast2fp_m(val))

using position = util::type::vec4<flt_pos>;      ///< position vector + mass
using velocity = util::type::vec3<flt_vel>;      ///< velocity vector
using acceleration = util::type::vec4<flt_acc>;  ///< acceleration vector + potential

using int_idx = uint32_t;  // index for N-body particles

constexpr int_idx N_simd_fp_l = SIMD_BITS / FP_L;  // number of SIMD lanes for fp_l
// constexpr int_idx N_simd_fp_m = SIMD_BITS / FP_M;  // number of SIMD lanes for fp_m
// constexpr int_idx N_simd_fp_h = SIMD_BITS / FP_H;  // number of SIMD lanes for fp_h

#ifdef USE_HERMITE
using flt_jrk = fp_m;  // use fp_m for the jerk
///
/// @brief append appropriate prefix for flt_jrk
///
#define AS_FLT_JRK(val) AS_FP_M(val)
///
/// @brief minimal cast to flt_jrk
///
#define CAST2JRK(val) (type::cast2fp_m(val))
using jerk = util::type::vec3<flt_jrk>;  // jerk vector

///
/// @brief Structure-of-Array for N-body particles
///
struct nbody {
  position *pos = nullptr;      // position of N-body particles
  velocity *vel = nullptr;      // velocity of N-body particles
  acceleration *acc = nullptr;  // acceleration of N-body particles
  jerk *jrk = nullptr;          // jerk of N-body particles
  fp_m *prs = nullptr;          // present time of N-body particles
  fp_m *nxt = nullptr;          // next time just after the orbit integration
  int_idx *idx = nullptr;       // particle ID
};
#endif  // USE_HERMITE

}  // namespace type
#endif  // COMMON_TYPE_HPP
