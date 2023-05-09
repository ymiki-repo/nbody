///
/// @file common/conservatives.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief evaluate conservatives in direct N-body simulation
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef COMMON_CONSERVATIVES_HPP
#define COMMON_CONSERVATIVES_HPP

#include <type_traits>  // std::remove_const_t

#include "common/type.hpp"
#include "util/hdf5.hpp"

class conservatives {
 public:
  ///
  /// @brief Construct a new conservatives object
  ///
  conservatives() = default;

  ///
  /// @brief record conservation errors
  ///
  /// @param[in] num number of N-body particles
  /// @param[in] pos position of N-body particles
  /// @param[in] vel velocity of N-body particles
  /// @param[in] acc acceleration of N-body particles
  /// @param[in] loc location or object identifier (group in most cases)
  /// @param[in] grp_name name of the new sub-group
  ///
  inline void write_hdf5(const type::int_idx num, const type::position *const pos, const type::velocity *const vel, const type::acceleration *const acc, const hid_t loc, const char *grp_name = "conservative") {
    calc(num, pos, vel, acc);

    const auto grp = H5Gcreate(loc, grp_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const auto d_spc = util::hdf5::setup_dataspace(static_cast<hsize_t>(1));

    // write the conservatives
    util::hdf5::write_type(grp, "fp_h", sizeof(type::fp_h), d_spc);

    // write the worst error
    auto subgrp = H5Gcreate(grp, "worst error", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#ifdef CALCULATE_POTENTIAL
    util::hdf5::write_attr(subgrp, "E_tot", &E_tot_worst, d_spc);
#endif  // CALCULATE_POTENTIAL
    util::hdf5::write_attr(subgrp, "px", &px_worst, d_spc);
    util::hdf5::write_attr(subgrp, "py", &py_worst, d_spc);
    util::hdf5::write_attr(subgrp, "pz", &pz_worst, d_spc);
    util::hdf5::write_attr(subgrp, "Lx", &Lx_worst, d_spc);
    util::hdf5::write_attr(subgrp, "Ly", &Ly_worst, d_spc);
    util::hdf5::write_attr(subgrp, "Lz", &Lz_worst, d_spc);
    util::hdf5::call(H5Gclose(subgrp));

    // write the latest error
    subgrp = H5Gcreate(grp, "latest error", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#ifdef CALCULATE_POTENTIAL
    util::hdf5::write_attr(subgrp, "E_tot", &E_tot_err, d_spc);
#endif  // CALCULATE_POTENTIAL
    util::hdf5::write_attr(subgrp, "px", &px_err, d_spc);
    util::hdf5::write_attr(subgrp, "py", &py_err, d_spc);
    util::hdf5::write_attr(subgrp, "pz", &pz_err, d_spc);
    util::hdf5::write_attr(subgrp, "Lx", &Lx_err, d_spc);
    util::hdf5::write_attr(subgrp, "Ly", &Ly_err, d_spc);
    util::hdf5::write_attr(subgrp, "Lz", &Lz_err, d_spc);
    util::hdf5::call(H5Gclose(subgrp));

    // write the current conservatives
    subgrp = H5Gcreate(grp, "current", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#ifdef CALCULATE_POTENTIAL
    util::hdf5::write_attr(subgrp, "E_tot", &E_tot, d_spc);
    util::hdf5::write_attr(subgrp, "E_kin", &E_kin, d_spc);
    util::hdf5::write_attr(subgrp, "E_pot", &E_pot, d_spc);
#endif  // CALCULATE_POTENTIAL
    util::hdf5::write_attr(subgrp, "px", &px, d_spc);
    util::hdf5::write_attr(subgrp, "py", &py, d_spc);
    util::hdf5::write_attr(subgrp, "pz", &pz, d_spc);
    util::hdf5::write_attr(subgrp, "Lx", &Lx, d_spc);
    util::hdf5::write_attr(subgrp, "Ly", &Ly, d_spc);
    util::hdf5::write_attr(subgrp, "Lz", &Lz, d_spc);
    util::hdf5::call(H5Gclose(subgrp));

    // write the initial information
    subgrp = H5Gcreate(grp, "initial", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    util::hdf5::write_attr(subgrp, "inv_E_tot", &inv_E_tot_ini, d_spc);
    util::hdf5::write_attr(subgrp, "px", &px_ini, d_spc);
    util::hdf5::write_attr(subgrp, "py", &py_ini, d_spc);
    util::hdf5::write_attr(subgrp, "pz", &pz_ini, d_spc);
    util::hdf5::write_attr(subgrp, "Lx", &Lx_ini, d_spc);
    util::hdf5::write_attr(subgrp, "Ly", &Ly_ini, d_spc);
    util::hdf5::write_attr(subgrp, "Lz", &Lz_ini, d_spc);
    util::hdf5::call(H5Gclose(subgrp));

    util::hdf5::close_dataspace(d_spc);
    util::hdf5::call(H5Gclose(grp));
  }

  // accessors
  ///
  /// @brief Get the energy error at the end of the simulation
  ///
  /// @return energy error at the end of the simulation
  ///
  [[nodiscard]] inline auto get_energy_error_final() const noexcept(true) {
    return (E_tot_err);
  }
  ///
  /// @brief Get the worst energy error of the simulation
  ///
  /// @return the worst energy error of the simulation
  ///
  [[nodiscard]] inline auto get_energy_error_worst() const noexcept(true) {
    return (E_tot_worst);
  }

 private:
  ///
  /// @brief calculate conservatives
  ///
  /// @param[in] num number of N-body particles
  /// @param[in] pos position of N-body particles
  /// @param[in] vel velocity of N-body particles
  /// @param[in] acc acceleration of N-body particles
  ///
  inline void calc(const type::int_idx num, const type::position *const pos, const type::velocity *const vel, [[maybe_unused]] const type::acceleration *const acc) {
    // calculate energy and momentum
#ifdef CALCULATE_POTENTIAL
    E_kin = AS_FP_H(0.0);
    E_pot = AS_FP_H(0.0);
#endif  // CALCULATE_POTENTIAL
    px = AS_FP_H(0.0);
    py = AS_FP_H(0.0);
    pz = AS_FP_H(0.0);
    Lx = AS_FP_H(0.0);
    Ly = AS_FP_H(0.0);
    Lz = AS_FP_H(0.0);
#pragma omp parallel for reduction(+ : px, py, pz, Lx, Ly, Lz, E_kin, E_pot)
    for (std::remove_const_t<decltype(num)> ii = 0U; ii < num; ii++) {
      const auto pi = pos[ii];
      const auto xi = type::cast2fp_h(pi.x);
      const auto yi = type::cast2fp_h(pi.y);
      const auto zi = type::cast2fp_h(pi.z);
      const auto mi = type::cast2fp_h(pi.w);
      px += mi * xi;
      py += mi * yi;
      pz += mi * zi;

      const auto vi = vel[ii];
      const auto vx = type::cast2fp_h(vi.x);
      const auto vy = type::cast2fp_h(vi.y);
      const auto vz = type::cast2fp_h(vi.z);
      Lx += mi * (yi * vz - zi * vy);
      Ly += mi * (zi * vx - xi * vz);
      Lz += mi * (xi * vy - yi * vx);

#ifdef CALCULATE_POTENTIAL
      E_kin += mi * (vx * vx + vy * vy + vz * vz);
      E_pot += mi * type::cast2fp_h(acc[ii].w);
#endif  // CALCULATE_POTENTIAL
    }
#ifdef CALCULATE_POTENTIAL
    E_kin = AS_FP_H(0.5) * E_kin;
    E_pot = AS_FP_H(0.5) * E_pot;
    E_tot = E_kin + E_pot;
#endif  // CALCULATE_POTENTIAL

    if (initialized) {
      // evaluate error of conservatives
#ifdef CALCULATE_POTENTIAL
      E_tot_err = (E_tot * inv_E_tot_ini - AS_FP_H(1.0));
#endif  // CALCULATE_POTENTIAL
      px_err = px - px_ini;
      py_err = py - py_ini;
      pz_err = pz - pz_ini;
      Lx_err = Lx - Lx_ini;
      Ly_err = Ly - Ly_ini;
      Lz_err = Lz - Lz_ini;

// record the worst error
#ifdef CALCULATE_POTENTIAL
      if (std::abs(E_tot_err) > std::abs(E_tot_worst)) {
        E_tot_worst = E_tot_err;
      }
#endif  // CALCULATE_POTENTIAL
      if (std::abs(px_err) > std::abs(px_worst)) {
        px_worst = px_err;
      }
      if (std::abs(py_err) > std::abs(py_worst)) {
        py_worst = py_err;
      }
      if (std::abs(pz_err) > std::abs(pz_worst)) {
        pz_worst = pz_err;
      }
      if (std::abs(Lx_err) > std::abs(Lx_worst)) {
        Lx_worst = Lx_err;
      }
      if (std::abs(Ly_err) > std::abs(Ly_worst)) {
        Ly_worst = Ly_err;
      }
      if (std::abs(Lz_err) > std::abs(Lz_worst)) {
        Lz_worst = Lz_err;
      }
    } else {
#ifdef CALCULATE_POTENTIAL
      inv_E_tot_ini = AS_FP_H(1.0) / E_tot;
#endif  // CALCULATE_POTENTIAL
      px_ini = px;
      py_ini = py;
      pz_ini = pz;
      Lx_ini = Lx;
      Ly_ini = Ly;
      Lz_ini = Lz;
      initialized = true;
    }
  }

  type::fp_h E_kin = AS_FP_H(0.0);          // kinetic energy
  type::fp_h E_pot = AS_FP_H(0.0);          // potential energy
  type::fp_h E_tot = AS_FP_H(0.0);          // total energy
  type::fp_h px = AS_FP_H(0.0);             // x-component of momentum
  type::fp_h py = AS_FP_H(0.0);             // y-component of momentum
  type::fp_h pz = AS_FP_H(0.0);             // z-component of momentum
  type::fp_h Lx = AS_FP_H(0.0);             // x-component of angular momentum
  type::fp_h Ly = AS_FP_H(0.0);             // y-component of angular momentum
  type::fp_h Lz = AS_FP_H(0.0);             // z-component of angular momentum
  type::fp_h inv_E_tot_ini = AS_FP_H(1.0);  // inverse of total energy at the initial step
  type::fp_h px_ini = AS_FP_H(0.0);         // x-component of momentum at the initial step
  type::fp_h py_ini = AS_FP_H(0.0);         // y-component of momentum at the initial step
  type::fp_h pz_ini = AS_FP_H(0.0);         // z-component of momentum at the initial step
  type::fp_h Lx_ini = AS_FP_H(0.0);         // x-component of angular momentum at the initial step
  type::fp_h Ly_ini = AS_FP_H(0.0);         // y-component of angular momentum at the initial step
  type::fp_h Lz_ini = AS_FP_H(0.0);         // z-component of angular momentum at the initial step
  type::fp_h E_tot_err = AS_FP_H(0.0);      // relative error for total energy
  type::fp_h px_err = AS_FP_H(0.0);         // absolute error for x-component of momentum
  type::fp_h py_err = AS_FP_H(0.0);         // absolute error for y-component of momentum
  type::fp_h pz_err = AS_FP_H(0.0);         // absolute error for z-component of momentum
  type::fp_h Lx_err = AS_FP_H(0.0);         // absolute error for x-component of angular momentum
  type::fp_h Ly_err = AS_FP_H(0.0);         // absolute error for y-component of angular momentum
  type::fp_h Lz_err = AS_FP_H(0.0);         // absolute error for z-component of angular momentum
  type::fp_h E_tot_worst = AS_FP_H(0.0);    // the worst relative error for total energy
  type::fp_h px_worst = AS_FP_H(0.0);       // the worst absolute error for x-component of momentum
  type::fp_h py_worst = AS_FP_H(0.0);       // the worst absolute error for y-component of momentum
  type::fp_h pz_worst = AS_FP_H(0.0);       // the worst absolute error for z-component of momentum
  type::fp_h Lx_worst = AS_FP_H(0.0);       // the worst absolute error for x-component of angular momentum
  type::fp_h Ly_worst = AS_FP_H(0.0);       // the worst absolute error for y-component of angular momentum
  type::fp_h Lz_worst = AS_FP_H(0.0);       // the worst absolute error for z-component of angular momentum
  bool initialized = false;                 // true if the conservatives are initialized
};

#endif  // COMMON_CONSERVATIVES_HPP
