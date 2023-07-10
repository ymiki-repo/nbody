///
/// @file pragma_omp.hpp
/// @author Yohei MIKI (Information Technology Center, The University of Tokyo)
/// @brief pragmas to use OpenMP directives
///
/// @copyright Copyright (c) 2023 Yohei MIKI
///
/// The MIT License is applied to this software, see LICENSE.txt
///
#ifndef PRAGMA_OMP_HPP
#define PRAGMA_OMP_HPP

#ifndef PRAGMA
///
/// @brief input arguments into _Pragma
///
#define PRAGMA(...) _Pragma(#__VA_ARGS__)
#endif  // PRAGMA

///
/// @brief add arguments into _Pragma("omp")
///
#define PRAGMA_OMP(...) PRAGMA(omp __VA_ARGS__)

///
/// @brief add arguments into _Pragma("omp parallel for")
///
#define PRAGMA_OMP_PARALLEL_FOR(...) PRAGMA_OMP(parallel for __VA_ARGS__)

///
/// @brief collapse the offloaded loops (specify number of loops collapsed)
///
#define PRAGMA_OMP_COLLAPSE(...) PRAGMA_OMP(collapse(__VA_ARGS__))

///
/// @brief collapse the offloaded loops
///
#define PRAGMA_OMP_COLLAPSE_ALL PRAGMA_OMP(collapse)

///
/// @brief collapse the offloaded loops
///
#define PRAGMA_OMP_COLLAPSE_ALL PRAGMA_OMP(collapse)

#endif  // PRAGMA_OMP_HPP
