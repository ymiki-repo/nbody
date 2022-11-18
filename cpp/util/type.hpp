///
/// @file util/type.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief additional datatypes
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef UTIL_TYPE_HPP
#define UTIL_TYPE_HPP

///
/// @brief additional datatypes
///
namespace util::type {
///
/// @brief cast to float
///
/// @param[in] input input
/// @return constexpr float
///
[[maybe_unused]] static inline constexpr auto cast2float(const float input) -> float { return (input); }
///
/// @brief cast to float
///
/// @param[in] input input
/// @return constexpr float
///
[[maybe_unused]] static inline constexpr auto cast2float(const double input) -> float { return (static_cast<float>(input)); }
///
/// @brief cast to float
///
/// @param[in] input input
/// @return constexpr float
///
[[maybe_unused]] static inline constexpr auto cast2float(const long double input) -> float { return (static_cast<float>(input)); }
///
/// @brief cast to double
///
/// @param[in] input input
/// @return constexpr double
///
[[maybe_unused]] static inline constexpr auto cast2double(const float input) -> double { return (static_cast<double>(input)); }
///
/// @brief cast to double
///
/// @param[in] input input
/// @return constexpr double
///
[[maybe_unused]] static inline constexpr auto cast2double(const double input) -> double { return (input); }
///
/// @brief cast to double
///
/// @param[in] input input
/// @return constexpr double
///
[[maybe_unused]] static inline constexpr auto cast2double(const long double input) -> double { return (static_cast<double>(input)); }
///
/// @brief cast to long double
///
/// @param[in] input input
/// @return constexpr long double
///
[[maybe_unused]] static inline constexpr auto cast2long_double(const float input) -> long double { return (static_cast<long double>(input)); }
///
/// @brief cast to long double
///
/// @param[in] input input
/// @return constexpr long double
///
[[maybe_unused]] static inline constexpr auto cast2long_double(const double input) -> long double { return (static_cast<long double>(input)); }
///
/// @brief cast to long double
///
/// @param[in] input input
/// @return constexpr long double
///
[[maybe_unused]] static inline constexpr auto cast2long_double(const long double input) -> long double { return (input); }

///
/// @brief 3-dimensional vector
///
template <class Type>
struct alignas(4 * sizeof(Type)) vec3 {
  Type x;  ///< x-component of the vector
  Type y;  ///< y-component of the vector
  Type z;  ///< z-component of the vector
};

///
/// @brief 4-dimensional vector
///
template <class Type>
struct alignas(4 * sizeof(Type)) vec4 {
  Type x;  ///< x-component of the vector
  Type y;  ///< y-component of the vector
  Type z;  ///< z-component of the vector
  Type w;  ///< w-component of the vector
};

}  // namespace util::type
#endif  // UTIL_TYPE_HPP
