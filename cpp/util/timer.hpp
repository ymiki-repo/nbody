///
/// @file util/timer.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief utility to measure elapsed time
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef UTIL_TIMER_HPP
#define UTIL_TIMER_HPP

#include <boost/timer/timer.hpp>  ///< boost::timer::cpu_timer

///
/// @brief utility tools
///
namespace util {
///
/// @brief measure elapsed time
///
class timer {
 public:
  ///
  /// @brief Construct a new timer object
  ///
  timer() = default;

  ///
  /// @brief read this.elapsed_wall
  ///
  /// @return this.elapsed_wall
  ///
  [[nodiscard]] inline auto get_elapsed_wall() const noexcept(true) {
    return (elapsed_wall);
  }

  ///
  /// @brief read this.elapsed_user
  ///
  /// @return this.elapsed_user
  ///
  [[nodiscard]] inline auto get_elapsed_user() const noexcept(true) {
    return (elapsed_user);
  }

  ///
  /// @brief start simple measurement
  ///
  inline void start() noexcept(true) {
    elapsed.start();
  }

  ///
  /// @brief stop simple measurement
  ///
  inline void stop() noexcept(true) {
    elapsed.stop();
    elapsed_wall += static_cast<double>(elapsed.elapsed().wall) * 1.0e-9;
    elapsed_user += static_cast<double>(elapsed.elapsed().user) * 1.0e-9;
  }

  ///
  /// @brief clear the previous measurement
  ///
  inline void clear() noexcept(true) {
    elapsed_wall = 0.0;
    elapsed_user = 0.0;
  }

 private:
  boost::timer::cpu_timer elapsed;  ///< stopwatch of the simulation to measure the elapsed time
  double elapsed_wall = 0.0;        ///< elapsed time as wall clock time
  double elapsed_user = 0.0;        ///< elapsed time as user CPU time
};
}  // namespace util

#endif  // UTIL_TIMER_HPP
