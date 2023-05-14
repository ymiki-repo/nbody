///
/// @file pragma_gpu.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief pragmas to use GPU by directives
///
/// @copyright Copyright (c) 2023 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef PRAGMA_GPU_HPP
#define PRAGMA_GPU_HPP

#ifdef _OPENACC
// use OpenACC directives for GPU offloading
#include "pragma_gpu_acc.hpp"

///
/// @brief offload the specified loop as thread-blocks with n threads
///
#define PRAGMA_GPU_OFFLOAD_THREAD(n) PRAGMA_ACC_KERNELS_LOOP_VECTOR(n)

///
/// @brief offload the specified loop as thread-blocks
///
#define PRAGMA_GPU_OFFLOAD PRAGMA_ACC_KERNELS_LOOP

///
/// @brief allocate device memory
///
#define PRAGMA_GPU_MALLOC(...) PRAGMA_ACC_MALLOC(__VA_ARGS__)

///
/// @brief release device memory
///
#define PRAGMA_GPU_FREE(...) PRAGMA_ACC_FREE(__VA_ARGS__)

///
/// @brief memcpy from device to host
///
#define PRAGMA_GPU_MEMCPY_D2H(...) PRAGMA_ACC_MEMCPY_D2H(__VA_ARGS__)

///
/// @brief memcpy from host to device
///
#define PRAGMA_GPU_MEMCPY_H2D(...) PRAGMA_ACC_MEMCPY_H2D(__VA_ARGS__)

#else  // _OPENACC

// use OpenMP target directives for GPU offloading
#include "pragma_gpu_omp.hpp"

///
/// @brief offload the specified loop as thread-blocks with n threads
///
#define PRAGMA_GPU_OFFLOAD_THREAD(n) PRAGMA_OMP_TARGET_TEAMS_DISTRIBUTE_THREAD(n)

///
/// @brief offload the specified loop as thread-blocks
///
#define PRAGMA_GPU_OFFLOAD PRAGMA_OMP_TARGET_TEAMS_LOOP

///
/// @brief allocate device memory
///
#define PRAGMA_GPU_MALLOC(...) PRAGMA_OMP_TARGET_MALLOC(__VA_ARGS__)

///
/// @brief release device memory
///
#define PRAGMA_GPU_FREE(...) PRAGMA_OMP_TARGET_FREE(__VA_ARGS__)

///
/// @brief memcpy from device to host
///
#define PRAGMA_GPU_MEMCPY_D2H(...) PRAGMA_OMP_TARGET_MEMCPY_D2H(__VA_ARGS__)

///
/// @brief memcpy from host to device
///
#define PRAGMA_GPU_MEMCPY_H2D(...) PRAGMA_OMP_TARGET_MEMCPY_H2D(__VA_ARGS__)

#endif  // _OPENACC

#endif  // PRAGMA_GPU_HPP
