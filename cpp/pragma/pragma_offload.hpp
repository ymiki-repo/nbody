///
/// @file pragma_offload.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief pragmas to offload operations by directives
///
/// @copyright Copyright (c) 2023 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef PRAGMA_OFFLOAD_HPP
#define PRAGMA_OFFLOAD_HPP

#ifdef USE_PRAGMA_OFFLOAD

#ifdef _OPENACC
// use OpenACC directives for GPU offloading
#include "pragma_acc.hpp"

///
/// @brief offload the specified loop as thread-blocks with n threads
///
#define PRAGMA_OFFLOAD_LOOP_THREAD(n) PRAGMA_ACC_LOOP_THREAD(n)

///
/// @brief offload the specified loop as thread-blocks
///
#define PRAGMA_OFFLOAD_LOOP PRAGMA_ACC_LOOP

///
/// @brief allocate device memory
///
#define PRAGMA_OFFLOAD_MALLOC(...) PRAGMA_ACC_MALLOC(__VA_ARGS__)

///
/// @brief release device memory
///
#define PRAGMA_OFFLOAD_FREE(...) PRAGMA_ACC_FREE(__VA_ARGS__)

///
/// @brief memcpy from device to host
///
#define PRAGMA_OFFLOAD_MEMCPY_D2H(...) PRAGMA_ACC_MEMCPY_D2H(__VA_ARGS__)

///
/// @brief memcpy from host to device
///
#define PRAGMA_OFFLOAD_MEMCPY_H2D(...) PRAGMA_ACC_MEMCPY_H2D(__VA_ARGS__)

#else  // _OPENACC

// use OpenMP target directives for GPU offloading
#include "pragma_omp_target.hpp"

///
/// @brief offload the specified loop as thread-blocks with n threads
///
#define PRAGMA_OFFLOAD_LOOP_THREAD(n) PRAGMA_OMP_TARGET_LOOP_THREAD(n)

///
/// @brief offload the specified loop as thread-blocks
///
#define PRAGMA_OFFLOAD_LOOP PRAGMA_OMP_TARGET_LOOP

///
/// @brief allocate device memory
///
#define PRAGMA_OFFLOAD_MALLOC(...) PRAGMA_OMP_TARGET_MALLOC(__VA_ARGS__)

///
/// @brief release device memory
///
#define PRAGMA_OFFLOAD_FREE(...) PRAGMA_OMP_TARGET_FREE(__VA_ARGS__)

///
/// @brief memcpy from device to host
///
#define PRAGMA_OFFLOAD_MEMCPY_D2H(...) PRAGMA_OMP_TARGET_MEMCPY_D2H(__VA_ARGS__)

///
/// @brief memcpy from host to device
///
#define PRAGMA_OFFLOAD_MEMCPY_H2D(...) PRAGMA_OMP_TARGET_MEMCPY_H2D(__VA_ARGS__)

#endif  // _OPENACC

#else  // USE_PRAGMA_OFFLOAD

// disable macros for offloading

#define PRAGMA_OFFLOAD_LOOP_THREAD(n)
#define PRAGMA_OFFLOAD_LOOP

#define PRAGMA_OFFLOAD_MALLOC(...)
#define PRAGMA_OFFLOAD_FREE(...)
#define PRAGMA_OFFLOAD_MEMCPY_D2H(...)
#define PRAGMA_OFFLOAD_MEMCPY_H2D(...)

#endif  // USE_PRAGMA_OFFLOAD

#endif  // PRAGMA_OFFLOAD_HPP
