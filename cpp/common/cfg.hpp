///
/// @file common/cfg.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief configure direct N-body simulation
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef COMMON_CFG_HPP
#define COMMON_CFG_HPP

#include <boost/program_options.hpp>  // boost::program_options

#include "common/type.hpp"

class config {
 public:
  ///
  /// @brief Construct a new config object
  ///
  config() = default;

  void configure(const int32_t argc, const char *const *const argv) {
    boost::program_options::options_description opt("List of options");
    opt.add_options()(
        "file", boost::program_options::value<decltype(file)>()->default_value("collapse"), "Name of the output files")(
        "softening", boost::program_options::value<decltype(eps)>()->default_value(AS_FLT_POS(1.5625e-2)), "Softening length (Plummer softening)")(
        "mass", boost::program_options::value<decltype(M_tot)>()->default_value(AS_FLT_POS(1.0)), "Total mass of the system")(
        "radius", boost::program_options::value<decltype(radius)>()->default_value(AS_FLT_POS(1.0)), "Radius of the initial sphere")(
        "virial", boost::program_options::value<decltype(virial)>()->default_value(AS_FLT_VEL(0.2)), "Virial ratio of the initial condition")(
#ifdef BENCHMARK_MODE
        "num_min", boost::program_options::value<decltype(num_min)>()->default_value(1024.0), "Minimum number of N-body particles")(
        "num_max", boost::program_options::value<decltype(num_max)>()->default_value(4.0 * 1024.0 * 1024.0), "Maximum number of N-body particles")(
        "num_bin", boost::program_options::value<decltype(num_bin)>()->default_value(13), "Number of logarithmic grids about number of N-body particles")(
        "elapse_min", boost::program_options::value<decltype(elapse_min)>()->default_value(1.0), "Minimum elapsed time for each measurement")(
#else  // BENCHMARK_MODE
       // 1031 is the minimum prime number greater than 1024
        "num", boost::program_options::value<decltype(num)>()->default_value(1031U), "Number of N-body particles")(
        "finish", boost::program_options::value<decltype(ft)>()->default_value(AS_FP_M(10.0)), "Final time of the simulation")(
        "interval", boost::program_options::value<decltype(interval)>()->default_value(AS_FP_M(0.125)), "Interval between snapshots")(
#ifdef USE_HERMITE
        "eta", boost::program_options::value<decltype(eta)>()->default_value(AS_FP_M(0.1)), "Safety parameter to determine time step in the simulation")(
#else   // USE_HERMITE
        "time_step", boost::program_options::value<decltype(dt)>()->default_value(AS_FP_M(7.8125e-3)), "Time step in the simulation")(
#endif  // USE_HERMITE
#endif  // BENCHMARK_MODE
        "help,h", "Help");
    // read input arguments
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opt), vm);
    boost::program_options::notify(vm);
    if (vm.count("help") == 1UL) {
      std::cout << opt << std::endl;
      std::exit(EXIT_SUCCESS);
    }

    file = vm["file"].as<decltype(file)>();
    eps = vm["softening"].as<decltype(eps)>();
    M_tot = vm["mass"].as<decltype(M_tot)>();
    radius = vm["radius"].as<decltype(radius)>();
    virial = vm["virial"].as<decltype(virial)>();
#ifdef BENCHMARK_MODE
    num_min = vm["num_min"].as<decltype(num_min)>();
    num_max = vm["num_max"].as<decltype(num_max)>();
    num_bin = vm["num_bin"].as<decltype(num_bin)>();
    elapse_min = vm["elapse_min"].as<decltype(elapse_min)>();
#else  // BENCHMARK_MODE
    num = vm["num"].as<decltype(num)>();
    ft = vm["finish"].as<decltype(ft)>();
    interval = vm["interval"].as<decltype(interval)>();
#ifdef USE_HERMITE
    eta = vm["eta"].as<decltype(eta)>();
#else   // USE_HERMITE
    dt = vm["time_step"].as<decltype(dt)>();
#endif  // USE_HERMITE
#endif  // BENCHMARK_MODE

    vm.clear();
  }

  // accessors
  ///
  /// @brief Get the file object
  ///
  /// @return name of the simulation
  ///
  [[nodiscard]] inline auto get_file() const noexcept(true) {
    return (file);
  }
  ///
  /// @brief Get the eps object
  ///
  /// @return softening length
  ///
  [[nodiscard]] constexpr auto get_eps() const noexcept(true) {
    return (eps);
  }
  ///
  /// @brief Get the mass object
  ///
  /// @return total mass of the system
  ///
  [[nodiscard]] constexpr auto get_mass() const noexcept(true) {
    return (M_tot);
  }
  ///
  /// @brief Get the radius object
  ///
  /// @return radius of the initial sphere
  ///
  [[nodiscard]] constexpr auto get_radius() const noexcept(true) {
    return (radius);
  }
  ///
  /// @brief Get the virial object
  ///
  /// @return Virial ratio of the initial condition
  ///
  [[nodiscard]] constexpr auto get_virial() const noexcept(true) {
    return (virial);
  }
  ///
  /// @brief Get the num object
  ///
  /// @return number of N-body particles
  ///
  [[nodiscard]] constexpr auto get_num() const noexcept(true) {
#ifdef BENCHMARK_MODE
    return (std::make_tuple(num_min, num_max, num_bin));
#else   // BENCHMARK_MODE
    return (num);
#endif  // BENCHMARK_MODE
  }
#ifdef BENCHMARK_MODE
  ///
  /// @brief Get the elapse_min object
  ///
  /// @return minimum elapsed time for each measurement
  ///
  [[nodiscard]] constexpr auto get_minimum_elapsed_time() const noexcept(true) {
    return (elapse_min);
  }
#else  // BENCHMARK_MODE
  ///
  /// @brief Get the time object
  ///
  /// @return final time of the simulation and interval between snapshots
  ///
  [[nodiscard]] constexpr auto get_time() const noexcept(true) {
    return (std::make_pair(ft, interval));
  }
#ifdef USE_HERMITE
  ///
  /// @brief Get the eta object
  ///
  /// @return eta in the simulation
  ///
  [[nodiscard]] constexpr auto get_eta() const noexcept(true) {
    return (eta);
  }
#else   // USE_HERMITE
  ///
  /// @brief Get the dt object
  ///
  /// @return time step in the simulation
  ///
  [[nodiscard]] constexpr auto get_dt() const noexcept(true) {
    return (dt);
  }
#endif  // USE_HERMITE
#endif  // BENCHMARK_MODE

 private:
  std::string file = "collapse";           // name of the simulation
  type::flt_pos eps = AS_FLT_POS(0.0);     // softening length
  type::flt_pos M_tot = AS_FLT_POS(0.0);   // total mass of the system
  type::flt_pos radius = AS_FLT_POS(0.0);  // radius of the initial sphere
  type::flt_vel virial = AS_FLT_VEL(0.0);  // Virial ratio of the initial condition
#ifdef BENCHMARK_MODE
  double num_min = 0.0;     // minimum number of N-body particles
  double num_max = 0.0;     // maximum number of N-body particles
  int32_t num_bin = 0;      // number of logarithmic grids about number of N-body particles
  double elapse_min = 1.0;  // minimum elapsed time for each measurement
#else                       // BENCHMARK_MODE
  type::int_idx num = 0U;              // number of N-body particles
  type::fp_m ft = AS_FP_M(0.0);        // final time of the simulation
  type::fp_m interval = AS_FP_M(0.0);  // interval between snapshots
#ifdef USE_HERMITE
  type::fp_m eta = AS_FP_M(1.0);  // safety parameter to determine time step in the simulation
#else   // USE_HERMITE
  type::fp_m dt = AS_FP_M(0.0);  // time step in the simulation
#endif  // USE_HERMITE
#endif  // BENCHMARK_MODE
};

#endif  // COMMON_CFG_HPP
