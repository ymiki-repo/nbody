///
/// @file common/io.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief file IO for direct N-body simulation
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef COMMON_IO_HPP
#define COMMON_IO_HPP

#include <fstream>  ///< std::ofstream
#include <iomanip>  ///< std::setw
#include <sstream>  ///< std::stringstream

// #include <cstdint>  ///< uint??_t
#include "common/conservatives.hpp"
#include "common/type.hpp"
#include "util/hdf5.hpp"

namespace io {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
#pragma GCC diagnostic ignored "-Wexit-time-destructors"
static const std::string folder_dat = "dat/";  ///< folder for output data files
static const std::string folder_log = "log/";  ///< folder for output log files
#pragma GCC diagnostic pop

// assume number of floating-point operations per interaction
#ifdef USE_HERMITE
// FP_M: 6 subtractions, 6 additions = 12 Flops
// FP_L: 8 FMA operations, 12 multiplications = 28 Flops
// FP_L: 1 reciprocal square root: assume as 4 Flops (estimation based on NVIDIA A100: 4 cycles for rsqrtf)
#ifdef USE_POTENTIAL
// FP_L: additional 1 multiplication for potential calculation
// FP_M: additional 1 addition for potential calculation
constexpr double FLOPS_PER_INTERACTION = 46.0;  ///< assumed number of floating-point operations per interaction
#else                                           // USE_POTENTIAL
constexpr double FLOPS_PER_INTERACTION = 44.0;  ///< assumed number of floating-point operations per interaction
#endif                                          // USE_POTENTIAL
#else                                           // USE_HERMITE
// FP_M: 3 subtractions, 3 additions = 6 Flops
// FP_L: 3 FMA operations, 6 multiplications = 12 Flops
// FP_L: 1 reciprocal square root: assume as 4 Flops (estimation based on NVIDIA A100: 4 cycles for rsqrtf)
#ifdef USE_POTENTIAL
// FP_L: additional 1 multiplication for potential calculation
// FP_M: additional 1 addition for potential calculation
constexpr double FLOPS_PER_INTERACTION = 24.0;  ///< assumed number of floating-point operations per interaction
#else   // USE_POTENTIAL
constexpr double FLOPS_PER_INTERACTION = 22.0;  ///< assumed number of floating-point operations per interaction
#endif  // USE_POTENTIAL
#endif  // USE_HERMITE

///
/// @brief write snapshot (HDF5 file) and corresponding XDMF file for VisIt/ParaView
///
/// @param[in] num number of N-body particles
/// @param[in] pos position of N-body particles
/// @param[in] vel_src velocity of N-body particles (dt/2 ahead in leapfrog integrator)
/// @param vel velocity of N-body particles (tentative array)
/// @param[in] acc acceleration of N-body particles
/// @param[in] filename name of the simulation
/// @param[in] snp_id snapshot ID
/// @param[in] time current time in the simulation
/// @param[in] dt time step in the simulation
/// @param[in,out] error error of conservatives in the simulation
///
static inline void write_snapshot(const type::int_idx num, type::position *__restrict pos, type::velocity *__restrict vel, type::acceleration *__restrict acc,
#ifdef USE_HERMITE
                                  type::jerk *__restrict jrk, const type::int_idx *idx,
#else   // USE_HERMITE
                                  const type::fp_m dt, const type::velocity *const vel_src,
#endif  // USE_HERMITE
                                  const char *const filename, const int32_t snp_id, const type::fp_m time, conservatives &error) {
#ifndef USE_HERMITE
  // backward integration for velocity
  const auto dt_2 = AS_FLT_VEL(0.5) * CAST2VEL(dt);
  for (type::int_idx ii = 0U; ii < num; ii++) {
    const auto ai = acc[ii];
    auto vi = vel_src[ii];
    vi.x -= dt_2 * CAST2VEL(ai.x);
    vi.y -= dt_2 * CAST2VEL(ai.y);
    vi.z -= dt_2 * CAST2VEL(ai.z);
    vel[ii] = vi;
  }
#endif  // USE_HERMITE

  // open the target file
  std::stringstream name;
  name.str("");
  name << filename << "_snp" << std::setw(3) << std::setfill('0') << snp_id;
  const auto hdf5_file = name.str() + ".h5";
  const auto file = std::string(folder_dat + hdf5_file);
  const auto loc = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // write particle data
  const char *grp_body = "particle";
  auto grp = H5Gcreate(loc, grp_body, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // preparation for hyperslab
  const int32_t N_dims = 2;
  const auto dims_vec4 = std::array<hsize_t, 2>{static_cast<hsize_t>(num), static_cast<hsize_t>(4)};
  const auto start = std::array<hsize_t, 2>{static_cast<hsize_t>(0), static_cast<hsize_t>(0)};
  const auto block = std::array<hsize_t, 2>{static_cast<hsize_t>(1), static_cast<hsize_t>(1)};
  // write 3-components vector from struct
  const auto dims_vec3 = std::array<hsize_t, 2>{static_cast<hsize_t>(num), static_cast<hsize_t>(3)};
  auto [d_spc, prop] = util::hdf5::setup_write2d(dims_vec3.begin());
  util::hdf5::write_data_partial(grp, "pos", reinterpret_cast<type::flt_pos *>(pos), d_spc, dims_vec4.begin(), dims_vec3.begin(), start.begin(), block.begin(), N_dims, prop);
  util::hdf5::write_data_partial(grp, "acc", reinterpret_cast<type::flt_acc *>(acc), d_spc, dims_vec4.begin(), dims_vec3.begin(), start.begin(), block.begin(), N_dims, prop);
  util::hdf5::write_data_partial(grp, "vel", reinterpret_cast<type::flt_vel *>(vel), d_spc, dims_vec4.begin(), dims_vec3.begin(), start.begin(), block.begin(), N_dims, prop);
#ifdef USE_HERMITE
  util::hdf5::write_data_partial(grp, "jerk", reinterpret_cast<type::flt_jrk *>(jrk), d_spc, dims_vec4.begin(), dims_vec3.begin(), start.begin(), block.begin(), N_dims, prop);
#endif  // USE_HERMITE
  util::hdf5::close_write(d_spc, prop);
  // write scalar from struct
  const auto dims_scalar = std::array<hsize_t, 2>{static_cast<hsize_t>(num), static_cast<hsize_t>(1)};
  std::tie(d_spc, prop) = util::hdf5::setup_write2d(dims_scalar.begin());
  const auto offset = std::array<hsize_t, 2>{static_cast<hsize_t>(0), static_cast<hsize_t>(3)};
  util::hdf5::write_data_partial(grp, "mass", reinterpret_cast<type::flt_pos *>(pos), d_spc, dims_vec4.begin(), dims_scalar.begin(), offset.begin(), block.begin(), N_dims, prop);
#ifdef CALCULATE_POTENTIAL
  util::hdf5::write_data_partial(grp, "pot", reinterpret_cast<type::flt_acc *>(acc), d_spc, dims_vec4.begin(), dims_scalar.begin(), offset.begin(), block.begin(), N_dims, prop);
#endif  // CALCULATE_POTENTIAL
  util::hdf5::close_write(d_spc, prop);
#ifdef USE_HERMITE
  // write scalar
  std::tie(d_spc, prop) = util::hdf5::setup_write1d(static_cast<hsize_t>(num));
  util::hdf5::write_data(grp, "ID", idx, d_spc, prop);
  util::hdf5::close_write(d_spc, prop);
#endif  // USE_HERMITE

  // write the particle information
  d_spc = util::hdf5::setup_dataspace(static_cast<hsize_t>(1));
  util::hdf5::write_attr(grp, "num", &num, d_spc);
  auto subgrp = H5Gcreate(grp, "datatype", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  util::hdf5::write_type(subgrp, "fp_m", sizeof(type::fp_m), d_spc);
  util::hdf5::write_type(subgrp, "flt_pos", sizeof(type::flt_pos), d_spc);
  util::hdf5::write_type(subgrp, "flt_vel", sizeof(type::flt_vel), d_spc);
  util::hdf5::write_type(subgrp, "flt_acc", sizeof(type::flt_acc), d_spc);
#ifdef USE_HERMITE
  util::hdf5::write_type(subgrp, "flt_jrk", sizeof(type::flt_jrk), d_spc);
#endif  // USE_HERMITE
  util::hdf5::call(H5Gclose(subgrp));
  util::hdf5::call(H5Gclose(grp));

  // write current time
  util::hdf5::write_attr(loc, "time", &time, d_spc);
  util::hdf5::close_dataspace(d_spc);

  // write conservation errors
  error.write_hdf5(num, pos, vel, acc, loc);

  // close the file
  util::hdf5::call(H5Fclose(loc));

  // generate XDMF file for VisIt or ParaView
  std::ofstream xml(folder_dat + name.str() + ".xdmf", std::ios::out);
  xml << R"(<?xml version="1.0" ?>)" << std::endl;
  xml << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>)" << std::endl;
  xml << R"(<Xdmf Version="3.0">)" << std::endl;
  xml << "  <Domain>" << std::endl;
  xml << R"(    <Grid Name="N-body" GridType="Uniform">)" << std::endl;
  xml << R"(      <Time Value=")" << time << R"("/>)" << std::endl;
  xml << R"(      <Topology TopologyType="Polyvertex" NumberOfElements=")" << num << R"("/>)" << std::endl;

  xml << R"(      <Geometry GeometryType="XYZ">)" << std::endl;
  xml << R"(        <DataItem Dimensions=")" << num << R"( 3" NumberType="Float" Precision=")" << sizeof(type::flt_pos) << R"(" Format="HDF">)" << std::endl;
  xml << "          " << hdf5_file << ":/" << grp_body << "/pos" << std::endl;
  xml << "        </DataItem>" << std::endl;
  xml << "      </Geometry>" << std::endl;

  xml << R"(      <Attribute Name="velocity" AttributeType="Vector" Center="Node">)" << std::endl;
  xml << R"(        <DataItem Dimensions=")" << num << R"( 3" NumberType="Float" Precision=")" << sizeof(type::flt_vel) << R"(" Format="HDF">)" << std::endl;
  xml << "          " << hdf5_file << ":/" << grp_body << "/vel" << std::endl;
  xml << "        </DataItem>" << std::endl;
  xml << "      </Attribute>" << std::endl;

  xml << R"(      <Attribute Name="acceleration" AttributeType="Vector" Center="Node">)" << std::endl;
  xml << R"(        <DataItem Dimensions=")" << num << R"( 3" NumberType="Float" Precision=")" << sizeof(type::flt_acc) << R"(" Format="HDF">)" << std::endl;
  xml << "          " << hdf5_file << ":/" << grp_body << "/acc" << std::endl;
  xml << "        </DataItem>" << std::endl;
  xml << "      </Attribute>" << std::endl;

#ifdef CALCULATE_POTENTIAL
  xml << R"(      <Attribute Name="potential" AttributeType="Scalar" Center="Node">)" << std::endl;
  xml << R"(        <DataItem Dimensions=")" << num << R"(" NumberType="Float" Precision=")" << sizeof(type::flt_acc) << R"(" Format="HDF">)" << std::endl;
  xml << "          " << hdf5_file << ":/" << grp_body << "/pot" << std::endl;
  xml << "        </DataItem>" << std::endl;
  xml << "      </Attribute>" << std::endl;
#endif  // CALCULATE_POTENTIAL

#ifdef USE_HERMITE
  xml << R"(      <Attribute Name="ID" AttributeType="Scalar" Center="Node">)" << std::endl;
  xml << R"(        <DataItem Dimensions=")" << num << R"(" NumberType="UInt" Precision=")" << sizeof(type::int_idx) << R"(" Format="HDF">)" << std::endl;
  xml << "          " << hdf5_file << ":/" << grp_body << "/ID" << std::endl;
  xml << "        </DataItem>" << std::endl;
  xml << "      </Attribute>" << std::endl;
#endif  // USE_HERMITE

  xml << "    </Grid>" << std::endl;
  xml << "  </Domain>" << std::endl;
  xml << "</Xdmf>" << std::endl;
  xml.close();
}

///
/// @brief write snapshot (HDF5 file) and corresponding XDMF file for VisIt/ParaView
///
/// @param[in] exec name of the executed binary
/// @param[in] elapsed elapsed time of the simulation
/// @param[in] step number of steps
/// @param[in] num number of N-body particles
/// @param[in] filename name of the simulation
/// @param[in] dt time step of the simulation
/// @param[in] error error of conservatives in the simulation
///
static inline void write_log(const char *const exec, const double elapsed, const int32_t step, const type::int_idx num, const char *const filename
#ifndef BENCHMARK_MODE
#ifdef CALCULATE_POTENTIAL
                             ,
#ifdef USE_HERMITE
                             const type::fp_m eta,
#else   // USE_HERMITE
                             const type::fp_m dt,
#endif  // USE_HERMITE
                             const conservatives &error
#endif  // CALCULATE_POTENTIAL
#ifdef USE_HERMITE
                             ,
                             const double N_pairs
#endif  // USE_HERMITE
#endif  // BENCHMARK_MODE
) {
  const boost::filesystem::path note(folder_log + filename + "_run.csv");
  boost::system::error_code err;
  const auto exist = boost::filesystem::exists(note, err);
  std::ofstream stats(note, std::ios::app);
  stats << std::scientific << std::setprecision(16);
  if (!exist || err) {
    stats << "exec,N,time[s],step,time_per_step[s],interactions_per_sec,Flop/s,FP_L,FP_M";
#if defined(CALCULATE_POTENTIAL) && !defined(BENCHMARK_MODE)
#ifdef USE_HERMITE
    stats << ",eta";
#else   // USE_HERMITE
    stats << ",dt";
#endif  // USE_HERMITE
    stats << ",energy_error_final,energy_error_worst";
#endif  // defined(CALCULATE_POTENTIAL) && !defined(BENCHMARK_MODE)
    stats << std::endl;
  }
  stats << exec;
  stats << "," << num;
  stats << "," << elapsed;
  stats << "," << step;
  stats << "," << elapsed / static_cast<double>(step);
#if !defined(USE_HERMITE) || defined(BENCHMARK_MODE)
  const auto N_pairs = static_cast<double>(num) * static_cast<double>(num) * static_cast<double>(step);
#endif  // !defined(USE_HERMITE) || defined(BENCHMARK_MODE)
  const auto pairs_per_sec = N_pairs / elapsed;
  stats << "," << pairs_per_sec;
  stats << "," << pairs_per_sec * FLOPS_PER_INTERACTION;
  stats << "," << FP_L;
  stats << "," << FP_M;
#if defined(CALCULATE_POTENTIAL) && !defined(BENCHMARK_MODE)
#ifdef USE_HERMITE
  stats << "," << eta;
#else   // USE_HERMITE
  stats << "," << dt;
#endif  // USE_HERMITE
  stats << "," << error.get_energy_error_final();
  stats << "," << error.get_energy_error_worst();
#endif  // defined(CALCULATE_POTENTIAL) && !defined(BENCHMARK_MODE)
  stats << std::endl;
  stats.close();
}

}  // namespace io

#endif  // COMMON_IO_HPP
