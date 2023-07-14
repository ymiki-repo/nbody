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

#ifdef OFFLOAD_BY_OPENACC
// use OpenACC directives for GPU offloading
#include "pragma_acc.hpp"

///
/// @brief offload the specified loop as thread-blocks
///
#define OFFLOAD_LOOP(...) PRAGMA_ACC_LOOP(__VA_ARGS__)

///
/// @brief optional argument to enhance SIMD vectorization
///
#define OPTARG_ENHANCE_SIMD PRAGMA_ACC_INDEPENDENT

///
/// @brief optional argument to suggest number of threads per thread-block
///
#define OPTARG_NUM_THREADS(n) PRAGMA_ACC_VECTOR(n)

///
/// @brief optional argument to collapse tightly-nested loops
///
#define OPTARG_COLLAPSE(n) PRAGMA_ACC_COLLAPSE(n)

///
/// @brief optional argument to perform reduction
///
#define OPTARG_REDUCTION(...) PRAGMA_ACC_REDUCTION(__VA_ARGS__)

///
/// @brief allocate device memory
///
#define OFFLOAD_MALLOC(...) PRAGMA_ACC_MALLOC(__VA_ARGS__)

///
/// @brief release device memory
///
#define OFFLOAD_FREE(...) PRAGMA_ACC_FREE(__VA_ARGS__)

///
/// @brief memcpy from device to host
///
#define OFFLOAD_MEMCPY_D2H(...) PRAGMA_ACC_MEMCPY_D2H(__VA_ARGS__)

///
/// @brief memcpy from host to device
///
#define OFFLOAD_MEMCPY_H2D(...) PRAGMA_ACC_MEMCPY_H2D(__VA_ARGS__)

#else  // OFFLOAD_BY_OPENACC

// use OpenMP directives (target is CPU or GPU)
#include "pragma_omp.hpp"

///
/// @brief optional argument to enhance SIMD vectorization
///
#define OPTARG_ENHANCE_SIMD PRAGMA_OMP_SIMD

///
/// @brief optional argument to collapse tightly-nested loops
///
#define OPTARG_COLLAPSE(n) PRAGMA_OMP_COLLAPSE(n)

///
/// @brief optional argument to perform reduction
///
#define OPTARG_REDUCTION(...) PRAGMA_OMP_REDUCTION(__VA_ARGS__)

#ifdef OFFLOAD_BY_OPENMP_TARGET

// use OpenMP target directives for GPU offloading
#include "pragma_omp_target.hpp"

///
/// @brief offload the specified loop as thread-blocks
///
#define OFFLOAD_LOOP(...) PRAGMA_OMP_TARGET_LOOP(__VA_ARGS__)

///
/// @brief optional argument to suggest number of threads per thread-block
///
#define OPTARG_NUM_THREADS(n) PRAGMA_OMP_THREAD_LIMIT(n)

///
/// @brief allocate device memory
///
#define OFFLOAD_MALLOC(...) PRAGMA_OMP_TARGET_MALLOC(__VA_ARGS__)

///
/// @brief release device memory
///
#define OFFLOAD_FREE(...) PRAGMA_OMP_TARGET_FREE(__VA_ARGS__)

///
/// @brief memcpy from device to host
///
#define OFFLOAD_MEMCPY_D2H(...) PRAGMA_OMP_TARGET_MEMCPY_D2H(__VA_ARGS__)

///
/// @brief memcpy from host to device
///
#define OFFLOAD_MEMCPY_H2D(...) PRAGMA_OMP_TARGET_MEMCPY_H2D(__VA_ARGS__)

#else  // OFFLOAD_BY_OPENMP_TARGET

// target of OpenMP directives are CPU
///
/// @brief offload the specified loop as thread-blocks
///
#define OFFLOAD_LOOP(...) PRAGMA_OMP_PARALLEL_FOR(__VA_ARGS__)

// disable macros only for offloading
#define OPTARG_NUM_THREADS(n)
#define OFFLOAD_MALLOC(...)
#define OFFLOAD_FREE(...)
#define OFFLOAD_MEMCPY_D2H(...)
#define OFFLOAD_MEMCPY_H2D(...)

#endif  // OFFLOAD_BY_OPENMP_TARGET
#endif  // OFFLOAD_BY_OPENACC

#endif  // PRAGMA_OFFLOAD_HPP
