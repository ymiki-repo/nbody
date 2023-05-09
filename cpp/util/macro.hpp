///
/// @file util/macro.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief macros
///
/// @copyright Copyright (c) 2022 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef UTIL_MACRO_HPP
#define UTIL_MACRO_HPP

#include <cstdlib>   // std::exit, EXIT_FAILURE
#include <iostream>  // std::ostream (std::cout, std::cerr)
#include <string>    // std::string

///
/// @brief print message with the source information
///
#define PRINT_MSG(dst, msg)                                          \
  {                                                                  \
    dst << __FILE__ << "(" << __LINE__ << "): " << __func__ << ": "; \
    dst << msg << std::endl;                                         \
    dst << std::flush;                                               \
  }
// constexpr void PRINT_MSG(std::ostream dst, const char *msg) noexcept(true) {
//   dst << __FILE__ << "(" << __LINE__ << "): " << __func__ << ": ";
//   dst << msg << std::endl;
//   dst << std::flush;
// }

///
/// @brief print debug messages
///
#ifdef NDEBUG
#define DEBUG_PRINT(dst, msg)
#else  // NDEBUG
#define DEBUG_PRINT(dst, msg) PRINT_MSG(dst, msg)
#endif  // NDEBUG

///
/// @brief print information
///
#ifndef SUPPRESS_PRINT_INFO
#define PRINT_INFO(msg) PRINT_MSG(std::cout, "INFO: " << msg)
// constexpr void PRINT_INFO(const char *msg) noexcept(true) { PRINT_MSG(std::cout, "INFO: " << msg); }
#else  /// SUPPRESS_PRINT_INFO
#define PRINT_INFO(msg)
#endif  // SUPPRESS_PRINT_INFO

///
/// @brief print warning
///
#ifndef SUPPRESS_PRINT_ALERT
#define PRINT_ALERT(msg) PRINT_MSG(std::cout, "WARNING: " << msg)
#else  /// SUPPRESS_PRINT_ALERT
#define PRINT_ALERT(msg)
#endif  // SUPPRESS_PRINT_ALERT

///
/// @brief print error
///
#define PRINT_ERROR(msg) PRINT_MSG(std::cerr, "ERROR: " << msg)

///
/// @brief kill job with the error message
///
#define KILL_JOB(msg)        \
  {                          \
    PRINT_ERROR(msg);        \
    std::exit(EXIT_FAILURE); \
  }

///
/// @brief kill MPI job with error message
///
#ifdef USE_MPI
#include <mpi.h>
#define KILL_MPI_JOB(msg)         \
  {                               \
    PRINT_ERROR(msg);             \
    MPI_Abort(MPI_COMM_WORLD, 1); \
    std::exit(EXIT_FAILURE);      \
  }
#else  /// USE_MPI
#define KILL_MPI_JOB(msg) KILL_JOB(msg)
#endif  // USE_MPI
#endif  // UTIL_MACRO_HPP
