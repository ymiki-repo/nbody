///
/// @file util/hdf5.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief utility functions using HDF5
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef UTIL_HDF5_HPP
#define UTIL_HDF5_HPP

#include <hdf5.h>  ///< HDF5

#include <complex>   ///< std::complex (to commit a new datatype)
#include <cstdint>   ///< int??_t
#include <cstdlib>   ///< std::exit
#include <iostream>  ///< std::cerr
#include <string>    ///< std::string
#include <vector>    ///< std::vector

#include "util/macro.hpp"  ///< KILL_JOB
#include "util/type.hpp"   ///< util::type

// ///
// /// @brief enable Gzip compression
// ///
// #define USE_GZIP_COMPRESSION

///
/// @brief utility functions for HDF5
///
namespace util::hdf5 {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
#pragma GCC diagnostic ignored "-Wexit-time-destructors"
#pragma GCC diagnostic pop

///
/// @brief The maximum number of elements in a chunk in HDF5 is 2^32-1 which is equal to 4,294,967,295
///
static inline constexpr auto max_chunk_size() { return (static_cast<hsize_t>(1) << 31U); }

///
/// @brief The maximum size for any chunk in HDF5 is 4 GB; it is 2^27 for 32bits variables
///
static inline constexpr auto max_chunk_size_32bit() { return (static_cast<hsize_t>(1) << 27U); }

///
/// @brief The maximum size for any chunk in HDF5 is 4 GB; it is 2^26 for 64bits variables
///
static inline constexpr auto max_chunk_size_64bit() { return (static_cast<hsize_t>(1) << 26U); }

///
/// @brief error handler of HDF5 function
///
/// @param[in] err error ID of HDF5 function
/// @param[in] file file name who calls the HDF5 function
/// @param[in] line number of line which calls the HDF5 function
/// @param[in] func name of function which calls the HDF5 function
///
static inline constexpr void _call_hdf5(const herr_t err, const char *file, const int32_t line, const char *func) noexcept(false) {
  if (err < 0) {
    std::cerr << file << "(" << line << "): " << func << ": ERROR: HDF5 returns error ID " << err << std::endl
              << std::flush;
    std::exit(EXIT_FAILURE);
  }
}

///
/// @brief parser to call error handler of HDF5 function
///
static inline constexpr void call(const herr_t err) noexcept(false) { _call_hdf5(err, __FILE__, __LINE__, __func__); }
// #define call(err) _call_hdf5((err), __FILE__, __LINE__, __func__)

#ifdef USE_MPI
///
/// @brief error handler of HDF5 function
///
/// @param[in] err error ID of HDF5 function
/// @param[in] file file name who calls the HDF5 function
/// @param[in] line number of line which calls the HDF5 function
/// @param[in] func name of function which calls the HDF5 function
///
static inline constexpr void _call_phdf5(const herr_t err, const char *file, const int32_t line, const char *func) noexcept(false) {
  if (err < 0) {
    std::cerr << file << "(" << line << "): " << func << ": ERROR: HDF5 returns error ID " << err << std::endl
              << std::flush;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(EXIT_FAILURE);
  }
}

///
/// @brief parser to call error handler of parallel HDF5 function
///
static inline constexpr void pcall(const herr_t err) noexcept(false) { _call_phdf5(err, __FILE__, __LINE__, __func__); }
// #define pcall(err) _call_phdf5((err), __FILE__, __LINE__, __func__)
#endif  // USE_MPI

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"

static hid_t h5t_complex_flt;   ///< user-defined datatype for std::complex<float> in HDF5
static hid_t h5t_complex_dbl;   ///< user-defined datatype for std::complex<double> in HDF5
static hid_t h5t_complex_ldbl;  ///< user-defined datatype for std::complex<long double> in HDF5
///
/// @brief prepare datatype for complex numbers
///
inline void commit_datatype_complex() noexcept(false) {
  h5t_complex_flt = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<float>));
  call(H5Tinsert(h5t_complex_flt, "real", 0, H5T_NATIVE_FLOAT));
  call(H5Tinsert(h5t_complex_flt, "imag", sizeof(float), H5T_NATIVE_FLOAT));
  h5t_complex_dbl = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
  call(H5Tinsert(h5t_complex_dbl, "real", 0, H5T_NATIVE_DOUBLE));
  call(H5Tinsert(h5t_complex_dbl, "imag", sizeof(double), H5T_NATIVE_DOUBLE));
  h5t_complex_ldbl = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<long double>));
  call(H5Tinsert(h5t_complex_ldbl, "real", 0, H5T_NATIVE_LDOUBLE));
  call(H5Tinsert(h5t_complex_ldbl, "imag", sizeof(long double), H5T_NATIVE_LDOUBLE));
}

///
/// @brief remove datatype for complex numbers
///
inline void remove_datatype_complex() noexcept(false) {
  call(H5Tclose(h5t_complex_flt));
  call(H5Tclose(h5t_complex_dbl));
  call(H5Tclose(h5t_complex_ldbl));
}

static hid_t h5t_vec3_flt;   ///< user-defined datatype for util::type::vec3<float> in HDF5
static hid_t h5t_vec3_dbl;   ///< user-defined datatype for util::type::vec3<double> in HDF5
static hid_t h5t_vec3_ldbl;  ///< user-defined datatype for util::type::vec3<long double> in HDF5
///
/// @brief prepare datatype for util::type::vec3
///
inline void commit_datatype_vec3() noexcept(false) {
  h5t_vec3_flt = H5Tcreate(H5T_COMPOUND, sizeof(util::type::vec3<float>));
  call(H5Tinsert(h5t_vec3_flt, "x", HOFFSET(util::type::vec3<float>, x), H5T_NATIVE_FLOAT));
  call(H5Tinsert(h5t_vec3_flt, "y", HOFFSET(util::type::vec3<float>, y), H5T_NATIVE_FLOAT));
  call(H5Tinsert(h5t_vec3_flt, "z", HOFFSET(util::type::vec3<float>, z), H5T_NATIVE_FLOAT));
  h5t_vec3_dbl = H5Tcreate(H5T_COMPOUND, sizeof(util::type::vec3<double>));
  call(H5Tinsert(h5t_vec3_dbl, "x", HOFFSET(util::type::vec3<double>, x), H5T_NATIVE_DOUBLE));
  call(H5Tinsert(h5t_vec3_dbl, "y", HOFFSET(util::type::vec3<double>, y), H5T_NATIVE_DOUBLE));
  call(H5Tinsert(h5t_vec3_dbl, "z", HOFFSET(util::type::vec3<double>, z), H5T_NATIVE_DOUBLE));
  h5t_vec3_ldbl = H5Tcreate(H5T_COMPOUND, sizeof(util::type::vec3<long double>));
  call(H5Tinsert(h5t_vec3_ldbl, "x", HOFFSET(util::type::vec3<long double>, x), H5T_NATIVE_LDOUBLE));
  call(H5Tinsert(h5t_vec3_ldbl, "y", HOFFSET(util::type::vec3<long double>, y), H5T_NATIVE_LDOUBLE));
  call(H5Tinsert(h5t_vec3_ldbl, "z", HOFFSET(util::type::vec3<long double>, z), H5T_NATIVE_LDOUBLE));
}

///
/// @brief remove datatype for util::type::vec3
///
inline void remove_datatype_vec3() noexcept(false) {
  call(H5Tclose(h5t_vec3_flt));
  call(H5Tclose(h5t_vec3_dbl));
  call(H5Tclose(h5t_vec3_ldbl));
}

static hid_t h5t_vec4_flt;   ///< user-defined datatype for util::type::vec4<float> in HDF5
static hid_t h5t_vec4_dbl;   ///< user-defined datatype for util::type::vec4<double> in HDF5
static hid_t h5t_vec4_ldbl;  ///< user-defined datatype for util::type::vec4<long double> in HDF5
///
/// @brief prepare datatype for util::type::vec4
///
inline void commit_datatype_vec4() noexcept(false) {
  // all members
  h5t_vec4_flt = H5Tcreate(H5T_COMPOUND, sizeof(util::type::vec4<float>));
  call(H5Tinsert(h5t_vec4_flt, "x", HOFFSET(util::type::vec4<float>, x), H5T_NATIVE_FLOAT));
  call(H5Tinsert(h5t_vec4_flt, "y", HOFFSET(util::type::vec4<float>, y), H5T_NATIVE_FLOAT));
  call(H5Tinsert(h5t_vec4_flt, "z", HOFFSET(util::type::vec4<float>, z), H5T_NATIVE_FLOAT));
  call(H5Tinsert(h5t_vec4_flt, "w", HOFFSET(util::type::vec4<float>, w), H5T_NATIVE_FLOAT));
  h5t_vec4_dbl = H5Tcreate(H5T_COMPOUND, sizeof(util::type::vec4<double>));
  call(H5Tinsert(h5t_vec4_dbl, "x", HOFFSET(util::type::vec4<double>, x), H5T_NATIVE_DOUBLE));
  call(H5Tinsert(h5t_vec4_dbl, "y", HOFFSET(util::type::vec4<double>, y), H5T_NATIVE_DOUBLE));
  call(H5Tinsert(h5t_vec4_dbl, "z", HOFFSET(util::type::vec4<double>, z), H5T_NATIVE_DOUBLE));
  call(H5Tinsert(h5t_vec4_dbl, "w", HOFFSET(util::type::vec4<double>, w), H5T_NATIVE_DOUBLE));
  h5t_vec4_ldbl = H5Tcreate(H5T_COMPOUND, sizeof(util::type::vec4<long double>));
  call(H5Tinsert(h5t_vec4_ldbl, "x", HOFFSET(util::type::vec4<long double>, x), H5T_NATIVE_LDOUBLE));
  call(H5Tinsert(h5t_vec4_ldbl, "y", HOFFSET(util::type::vec4<long double>, y), H5T_NATIVE_LDOUBLE));
  call(H5Tinsert(h5t_vec4_ldbl, "z", HOFFSET(util::type::vec4<long double>, z), H5T_NATIVE_LDOUBLE));
  call(H5Tinsert(h5t_vec4_ldbl, "w", HOFFSET(util::type::vec4<long double>, w), H5T_NATIVE_LDOUBLE));
}

///
/// @brief remove datatype for util::type::vec4
///
inline void remove_datatype_vec4() noexcept(false) {
  call(H5Tclose(h5t_vec4_flt));
  call(H5Tclose(h5t_vec4_dbl));
  call(H5Tclose(h5t_vec4_ldbl));
}

static inline auto h5type([[maybe_unused]] const int16_t val) noexcept(true) { return (H5T_NATIVE_SHORT); }                      ///< returns predefined native datatype for int16_t
static inline auto h5type([[maybe_unused]] const int32_t val) noexcept(true) { return (H5T_NATIVE_INT); }                        ///< returns predefined native datatype for int32_t
static inline auto h5type([[maybe_unused]] const int64_t val) noexcept(true) { return (H5T_NATIVE_LONG); }                       ///< returns predefined native datatype for int64_t
static inline auto h5type([[maybe_unused]] const uint16_t val) noexcept(true) { return (H5T_NATIVE_USHORT); }                    ///< returns predefined native datatype for uint16_t
static inline auto h5type([[maybe_unused]] const uint32_t val) noexcept(true) { return (H5T_NATIVE_UINT); }                      ///< returns predefined native datatype for uint32_t
static inline auto h5type([[maybe_unused]] const uint64_t val) noexcept(true) { return (H5T_NATIVE_ULONG); }                     ///< returns predefined native datatype for uint64_t
static inline auto h5type([[maybe_unused]] const float val) noexcept(true) { return (H5T_NATIVE_FLOAT); }                        ///< returns predefined native datatype for float
static inline auto h5type([[maybe_unused]] const double val) noexcept(true) { return (H5T_NATIVE_DOUBLE); }                      ///< returns predefined native datatype for double
static inline auto h5type([[maybe_unused]] const long double val) noexcept(true) { return (H5T_NATIVE_LDOUBLE); }                ///< returns predefined native datatype for long double
static inline auto h5type([[maybe_unused]] const std::complex<float> val) noexcept(true) { return (h5t_complex_flt); }           ///< returns user-define datatype for std::complex<float>
static inline auto h5type([[maybe_unused]] const std::complex<double> val) noexcept(true) { return (h5t_complex_dbl); }          ///< returns user-define datatype for std::complex<double>
static inline auto h5type([[maybe_unused]] const std::complex<long double> val) noexcept(true) { return (h5t_complex_ldbl); }    ///< returns user-define datatype for std::complex<long double>
static inline auto h5type([[maybe_unused]] const util::type::vec3<float> &val) noexcept(true) { return (h5t_vec3_flt); }         ///< returns user-define datatype for util::type::vec3<float>
static inline auto h5type([[maybe_unused]] const util::type::vec3<double> &val) noexcept(true) { return (h5t_vec3_dbl); }        ///< returns user-define datatype for util::type::vec3<double>
static inline auto h5type([[maybe_unused]] const util::type::vec3<long double> &val) noexcept(true) { return (h5t_vec3_ldbl); }  ///< returns user-define datatype for util::type::vec3<long double>
static inline auto h5type([[maybe_unused]] const util::type::vec4<float> &val) noexcept(true) { return (h5t_vec4_flt); }         ///< returns user-define datatype for util::type::vec4<float>
static inline auto h5type([[maybe_unused]] const util::type::vec4<double> &val) noexcept(true) { return (h5t_vec4_dbl); }        ///< returns user-define datatype for util::type::vec4<double>
static inline auto h5type([[maybe_unused]] const util::type::vec4<long double> &val) noexcept(true) { return (h5t_vec4_ldbl); }  ///< returns user-define datatype for util::type::vec4<long double>

///
/// @brief create a new simple dataspace
///
/// @param[in] dims size of the dataspace
/// @return identifier of the new dataspace
///
inline auto setup_dataspace(const hsize_t dims) noexcept(false) {
  return (H5Screate_simple(1, &dims, NULL));
}
///
/// @brief create a new simple dataspace
///
/// @param[in] dims size of the dataspace in each dimension
/// @param[in] N_dim number of dimensions for the dataspace
/// @return identifier of the new dataspace
///
inline auto setup_dataspace(const hsize_t *const dims, const int32_t N_dim = 1) noexcept(false) {
  return (H5Screate_simple(N_dim, dims, NULL));
}

///
/// @brief close the dataspace
///
/// @param[in] dataspace identifier of the dataspace
///
inline void close_dataspace(const hid_t dataspace) noexcept(false) {
  call(H5Sclose(dataspace));
}

///
/// @brief write an attribute (for std::string)
///
/// @param[in] loc location or object identifier (group in most cases)
/// @param[in] name name of the attribute
/// @param[in] obj attribute to be written
/// @param[in] dataspace dataspace for the attribute
///
inline void write_attr(const hid_t loc, const char *name, const std::string &obj, const hid_t dataspace) noexcept(false) {
  const auto datatype = H5Tcopy(H5T_C_S1);
  call(H5Tset_size(datatype, H5T_VARIABLE));
  const auto *const tmp = obj.c_str();
  constexpr auto creation_property = H5P_DEFAULT;
  constexpr auto access_property = H5P_DEFAULT;
  const auto attribute = H5Acreate(loc, name, datatype, dataspace, creation_property, access_property);
  call(H5Awrite(attribute, datatype, &tmp));
  call(H5Aclose(attribute));
  call(H5Tclose(datatype));
}
///
/// @brief write an attribute (as a template function)
///
/// @param[in] loc location or object identifier (group in most cases)
/// @param[in] name name of the attribute
/// @param[in] obj attribute to be written
/// @param[in] dataspace dataspace for the attribute
///
template <class Type>
inline void write_attr(const hid_t loc, const char *name, const Type *const obj, const hid_t dataspace) noexcept(false) {
  const auto datatype = h5type(*obj);
  constexpr auto creation_property = H5P_DEFAULT;
  constexpr auto access_property = H5P_DEFAULT;
  const auto attribute = H5Acreate(loc, name, datatype, dataspace, creation_property, access_property);
  call(H5Awrite(attribute, datatype, obj));
  call(H5Aclose(attribute));
}

///
/// @brief read an attribute (for std::string)
///
/// @param[in] loc location or object identifier (group in most cases)
/// @param[in] name name of the attribute
/// @param[in,out] obj attribute to be read
///
inline void read_attr(const hid_t loc, const char *name, std::string &obj) noexcept(false) {
  if (H5Aexists(loc, name) > 0) {
    std::vector<char> val;
    const auto datatype = H5Tcopy(H5T_C_S1);
    call(H5Tset_size(datatype, H5T_VARIABLE));
    constexpr auto access_property = H5P_DEFAULT;
    const auto attribute = H5Aopen(loc, name, access_property);
    call(H5Aread(attribute, datatype, &val));
    call(H5Aclose(attribute));
    call(H5Tclose(datatype));
    obj = val.data();
  }
}
///
/// @brief read an attribute (as a template function)
///
/// @param[in] loc location or object identifier (group in most cases)
/// @param[in] name name of the attribute
/// @param[in,out] obj attribute to be read
///
template <class Type>
inline void read_attr(const hid_t loc, const char *name, Type *const obj) noexcept(false) {
  if (H5Aexists(loc, name) > 0) {
    constexpr auto access_property = H5P_DEFAULT;
    const auto attribute = H5Aopen(loc, name, access_property);
    call(H5Aread(attribute, h5type(*obj), obj));
    call(H5Aclose(attribute));
  }
}

///
/// @brief write a datatype as the size
///
/// @param[in] loc location or object identifier (group in most cases)
/// @param[in] name name of the attribute
/// @param[in] obj size of the datatype
/// @param[in] dataspace dataspace for the attribute
///
inline void write_type(const hid_t loc, const char *name, const size_t obj, const hid_t dataspace) noexcept(false) {
  write_attr(loc, name, &obj, dataspace);
}

///
/// @brief read a datatype as the size
///
/// @param[in] loc location or object identifier (group in most cases)
/// @param[in] name name of the attribute
/// @param[in] obj expected size of the datatype
///
inline void read_type(const hid_t loc, const char *name, const size_t obj) noexcept(false) {
  size_t get = 0;  ///< obj must not be 0
  read_attr(loc, name, &get);
  if (obj != get) {
    KILL_JOB("ERROR: " << name << " (" << obj << " byte) does not match with that in the input file (" << get << " byte)")
  }
}

///
/// @brief prepare to write dataset in 1D
///
/// @param[in] dims size of the the dataspace
/// @return identifier of the new dataspace and the transfer property list for this I/O operation (compression in this case)
///
inline auto setup_write1d(const hsize_t dims) noexcept(false) {
  // create data space
  const auto dataspace = setup_dataspace(dims);

#ifdef USE_GZIP_COMPRESSION
  // prepare on-the-fly data compression
  const auto c_dims_max = std::min(MAXIMUM_CHUNK_SIZE, MAXIMUM_CHUNK_SIZE_64BIT);
  const auto c_dims_gzip = static_cast<hsize_t>(1024);
  const uint32_t compress_level = 9;  ///< 9 is the maximum compression level
  const auto transfer_property = H5Pcreate(H5P_DATASET_CREATE);
  const auto c_dims_loc = std::min({dims, c_dims_gzip, c_dims_max});
  call(H5Pset_chunk(transfer_property, 1, &c_dims_loc));
  call(H5Pset_deflate(transfer_property, compress_level));
#else   // USE_GZIP_COMPRESSION
  const auto transfer_property = H5P_DEFAULT;
#endif  // USE_GZIP_COMPRESSION

  return (std::make_pair(dataspace, transfer_property));
}

///
/// @brief prepare to write dataset in 1D
///
/// @param[in] dims size of the the dataspace
/// @return identifier of the new dataspace and the transfer property list for this I/O operation (compression in this case)
///
inline auto setup_write2d(const hsize_t *const dims) noexcept(false) {
  // create dataspace
  const auto dataspace = setup_dataspace(dims, 2);

// prepare on-the-fly data compression
#ifdef USE_GZIP_COMPRESSION
  const hsize_t c_dims_max = std::min(MAXIMUM_CHUNK_SIZE, MAXIMUM_CHUNK_SIZE_64BIT);
  hsize_t c_dims_gzip[2] = {
      std::min(dims[0], static_cast<hsize_t>(1024) / std::min(dims[1], static_cast<hsize_t>(1024))),
      std::min(dims[1], static_cast<hsize_t>(1024))};
  const uint32_t compress_level = 9;  ///< 9 is the maximum compression level
  const auto transfer_property = H5Pcreate(H5P_DATASET_CREATE);
  auto c_dims_loc = c_dims_gzip;
  if (c_dims_loc[0] * c_dims_loc[1] > c_dims_max)
    c_dims_loc[0] = c_dims_max / c_dims_loc[1];
  call(H5Pset_chunk(transfer_property, 2, c_dims_loc));
  call(H5Pset_deflate(transfer_property, compress_level));
#else   // USE_GZIP_COMPRESSION
  const auto transfer_property = H5P_DEFAULT;
#endif  // USE_GZIP_COMPRESSION

  return (std::make_pair(dataspace, transfer_property));
}

///
/// @brief close writing dataset
///
/// @param[in] dataspace identifier of the dataspace
/// @param[in] transfer_property identifier of the transfer property list for this I/O operation
///
inline void close_write(const hid_t dataspace, [[maybe_unused]] const hid_t transfer_property) noexcept(false) {
  call(H5Sclose(dataspace));
#ifdef USE_GZIP_COMPRESSION
  call(H5Pclose(transfer_property));
#endif  // USE_GZIP_COMPRESSION
}

///
/// @brief write a dataset
///
/// @param[in] loc location or object identifier (group in most cases)
/// @param[in] name name of the dataset
/// @param[in] buf buffer with data to be written to the file
/// @param[in] dataspace dataspace
/// @param[in] transfer_property identifier of the transfer property list for this I/O operation (optional)
/// @param[in] input_datatype identifier of the specified datatype (optional)
/// @param[in] memory_space identifier of the memory dataspace
/// @param[in] file_space identifier of the dataset's dataspace in the file
///
template <class Type>
inline void write_data(const hid_t loc, const char *name, const Type *const buf, const hid_t dataspace, const hid_t transfer_property = H5P_DEFAULT, const hid_t input_datatype = H5I_UNINIT, const hid_t memory_space = H5S_ALL, const hid_t file_space = H5S_ALL) noexcept(false) {
  const auto memory_datatype = (input_datatype == H5I_UNINIT) ? h5type(*buf) : input_datatype;
  constexpr auto link_creation = H5P_DEFAULT;
  constexpr auto creation_property = H5P_DEFAULT;
  constexpr auto access_property = H5P_DEFAULT;
  const auto dataset = H5Dcreate(loc, name, memory_datatype, dataspace, link_creation, creation_property, access_property);
  call(H5Dwrite(dataset, memory_datatype, memory_space, file_space, transfer_property, buf));
  call(H5Dclose(dataset));
}

///
/// @brief read a dataset
///
/// @param[in] loc location or object identifier (group in most cases)
/// @param[in] name name of the dataset
/// @param[in] buf dataset to be read
/// @param[in] input_datatype identifier of the specified datatype (optional)
///
template <class Type>
inline void read_data(const hid_t loc, const char *name, Type *const buf, const hid_t input_datatype = H5I_UNINIT) noexcept(false) {
  constexpr auto link_access_property = H5P_DEFAULT;
  if (H5Lexists(loc, name, link_access_property)) {
    const auto memory_datatype = (input_datatype == H5I_UNINIT) ? h5type(*buf) : input_datatype;
    constexpr auto access_property = H5P_DEFAULT;
    const auto dataset = H5Dopen(loc, name, access_property);
    constexpr auto memory_space = H5S_ALL;
    constexpr auto file_space = H5S_ALL;
    constexpr auto transfer_property = H5P_DEFAULT;
    call(H5Dread(dataset, memory_datatype, memory_space, file_space, transfer_property, buf));
    call(H5Dclose(dataset));
  }
}

///
/// @brief write subset of a dataset using hyperslab
///
/// @param[in] loc location or object identifier (group in most cases)
/// @param[in] name name of the dataset
/// @param[in] buf buffer with data to be written to the file
/// @param[in] file_space identifier of the dataset's dataspace in the file
/// @param[in] whole_space dataspace in memory, used as hyperslab stride
/// @param[in] sub_space dataspace to be written, use ad size of block in hyperslab
/// @param[in] start offset of start of the hyperslab
/// @param[in] count number of blocks included in hyperslab
/// @param[in] N_dim number of dimensions for the dataspace
/// @param[in] transfer_property identifier of the transfer property list for this I/O operation (optional)
///
template <class Type>
inline void write_data_partial(const hid_t loc, const char *name, const Type *const buf, const hid_t file_space, const hsize_t *const whole_space, const hsize_t *const sub_space, const hsize_t *const start, const hsize_t *const count, const int32_t N_dim = 1, const hid_t transfer_property = H5P_DEFAULT) noexcept(false) {
  const auto memory_space = setup_dataspace(whole_space, N_dim);
  call(H5Sselect_hyperslab(memory_space, H5S_SELECT_SET, start, whole_space, count, sub_space));
  write_data(loc, name, buf, file_space, transfer_property, h5type(*buf), memory_space, file_space);
}
}  // namespace util::hdf5

#pragma GCC diagnostic pop
#endif  // UTIL_HDF5_HPP
