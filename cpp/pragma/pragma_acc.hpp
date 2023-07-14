///
/// @file pragma_acc.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief pragmas to use GPU by OpenACC directives
///
/// @copyright Copyright (c) 2023 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef PRAGMA_ACC_HPP
#define PRAGMA_ACC_HPP

#ifndef PRAGMA
///
/// @brief input arguments into _Pragma
///
#define PRAGMA(...) _Pragma(#__VA_ARGS__)
#endif  // PRAGMA

///
/// @brief add arguments into _Pragma("acc")
///
#define PRAGMA_ACC(...) PRAGMA(acc __VA_ARGS__)

///
/// @brief add arguments into _Pragma("acc kernels") _Pragma("acc loop")
///
#define PRAGMA_ACC_KERNELS_LOOP(...) PRAGMA_ACC(kernels) PRAGMA_ACC(loop __VA_ARGS__)

///
/// @brief offload the specified loop as thread-blocks
///
#define PRAGMA_ACC_LOOP(...) PRAGMA_ACC_KERNELS_LOOP(__VA_ARGS__)

///
/// @brief argument to enhance SIMD vectorization
///
#define PRAGMA_ACC_INDEPENDENT independent

///
/// @brief argument to suggest number of threads per thread-block
///
#define PRAGMA_ACC_VECTOR(n) vector(n)

///
/// @brief argument to collapse tightly-nested loops
///
#define PRAGMA_ACC_COLLAPSE(n) collapse(n)

///
/// @brief argument to perform reduction
///
#define PRAGMA_ACC_REDUCTION(...) reduction(__VA_ARGS__)

///
/// @brief allocate device memory
///
#define PRAGMA_ACC_MALLOC(...) PRAGMA_ACC(enter data create(__VA_ARGS__))

///
/// @brief release device memory
///
#define PRAGMA_ACC_FREE(...) PRAGMA_ACC(exit data delete (__VA_ARGS__))

///
/// @brief memcpy from device to host
///
#define PRAGMA_ACC_MEMCPY_D2H(...) PRAGMA_ACC(update host(__VA_ARGS__))

///
/// @brief memcpy from host to device
///
#define PRAGMA_ACC_MEMCPY_H2D(...) PRAGMA_ACC(update device(__VA_ARGS__))

#endif  // PRAGMA_ACC_HPP
