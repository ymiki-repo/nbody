///
/// @file pragma_gpu_acc.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief pragmas to use GPU by OpenACC directives
///
/// @copyright Copyright (c) 2023 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef PRAGMA_GPU_ACC_HPP
#define PRAGMA_GPU_ACC_HPP

///
/// @brief offload the specified loop as thread-blocks with n threads
///
#define PRAGMA_ACC_OFFLOAD_THREAD(n)            \
  {                                             \
    _Pragma(acc kernels)                        \
        _Pragma(acc loop independent vector(n)) \
  }

///
/// @brief offload the specified loop as thread-blocks
///
#define PRAGMA_ACC_OFFLOAD            \
  {                                   \
    _Pragma(acc kernels)              \
        _Pragma(acc loop independent) \
  }

///
/// @brief allocate device memory
///
#define PRAGMA_ACC_MALLOC(...) _Pragma(acc enter data create(__VA_ARGS__))

///
/// @brief release device memory
///
#define PRAGMA_ACC_FREE(...) _Pragma(acc exit data delete (__VA_ARGS__))

///
/// @brief memcpy from device to host
///
#define PRAGMA_ACC_MEMCPY_D2H(...) _Pragma(acc update host(__VA_ARGS__))

///
/// @brief memcpy from host to device
///
#define PRAGMA_ACC_MEMCPY_H2D(...) _Pragma(acc update device(__VA_ARGS__))

#endif  // PRAGMA_GPU_ACC_HPP
