///
/// @file pragma_omp_target.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief pragmas to use GPU by OpenMP target directives
///
/// @copyright Copyright (c) 2023 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef PRAGMA_OMP_TARGET_HPP
#define PRAGMA_OMP_TARGET_HPP

#include "pragma_omp.hpp"

// nvc++ (_OPENMP 202011) has #pragma omp target loop
// amdclang++ (_OPENMP 201811) does not have #pragma omp target loop
#if _OPENMP >= 202011
#define OMP_TARGET_HAS_LOOP_DIRECTIVE true
#else
#define OMP_TARGET_HAS_LOOP_DIRECTIVE false
#endif

///
/// @brief add arguments into _Pragma("omp target")
///
#define PRAGMA_OMP_TARGET(...) PRAGMA_OMP(target __VA_ARGS__)

///
/// @brief add arguments into _Pragma("omp target teams distribute parallel for simd")
///
#define PRAGMA_OMP_TARGET_TEAMS_DISTRIBUTE(...) PRAGMA_OMP_TARGET(teams distribute parallel for simd __VA_ARGS__)

///
/// @brief offload the specified loop as thread-blocks with n threads
///
#define PRAGMA_OMP_TARGET_LOOP_THREAD(n) PRAGMA_OMP_TARGET_TEAMS_DISTRIBUTE(thread_limit(n))

///
/// @brief offload the specified loop as thread-blocks
///
#if OMP_TARGET_HAS_LOOP_DIRECTIVE
#define PRAGMA_OMP_TARGET_LOOP PRAGMA_OMP_TARGET(teams loop)
#define PRAGMA_OMP_TARGET_LOOP_WITH_ARGS(...) PRAGMA_OMP_TARGET(teams loop __VA_ARGS__)
#else  // OMP_TARGET_HAS_LOOP_DIRECTIVE
#define PRAGMA_OMP_TARGET_LOOP PRAGMA_OMP_TARGET_TEAMS_DISTRIBUTE()
#define PRAGMA_OMP_TARGET_LOOP_WITH_ARGS(...) PRAGMA_OMP_TARGET_TEAMS_DISTRIBUTE(__VA_ARGS__)
#endif  // OMP_TARGET_HAS_LOOP_DIRECTIVE

///
/// @brief offload the specified loop with reduction
///
#define PRAGMA_OMP_TARGET_LOOP_REDUCE(...) PRAGMA_OMP_TARGET_LOOP_WITH_ARGS(reduction(__VA_ARGS__))

///
/// @brief collapse the offloaded loops (specify number of loops collapsed)
///
#define PRAGMA_OMP_TARGET_LOOP_COLLAPSE(n) PRAGMA_OMP_TARGET_LOOP_WITH_ARGS(collapse(n))

///
/// @brief collapse the offloaded loops (specify number of loops collapsed) with reduction
///
#define PRAGMA_OMP_TARGET_LOOP_COLLAPSE_REDUCE(n, ...) PRAGMA_OMP_TARGET_LOOP_WITH_ARGS(collapse(n) reduction(__VA_ARGS__))

///
/// @brief allocate device memory
///
#define PRAGMA_OMP_TARGET_MALLOC(...) PRAGMA_OMP_TARGET(enter data map(alloc \
                                                                       : __VA_ARGS__))
///
/// @brief release device memory
///
#define PRAGMA_OMP_TARGET_FREE(...) PRAGMA_OMP_TARGET(exit data map(delete \
                                                                    : __VA_ARGS__))
///
/// @brief memcpy from device to host
///
#define PRAGMA_OMP_TARGET_MEMCPY_D2H(...) PRAGMA_OMP_TARGET(update from(__VA_ARGS__))

///
/// @brief memcpy from host to device
///
#define PRAGMA_OMP_TARGET_MEMCPY_H2D(...) PRAGMA_OMP_TARGET(update to(__VA_ARGS__))

#endif  // PRAGMA_OMP_TARGET_HPP
