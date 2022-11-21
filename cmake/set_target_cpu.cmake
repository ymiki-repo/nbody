# for GCC or LLVM
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(TARGET_CPU native CACHE STRING "target CPU architecture")
    set_property(CACHE TARGET_CPU PROPERTY STRINGS znver3 znver2 cascadelake icelake-server icelake-client sapphirerapids alderlake rocketlake x86-64-v4 x86-64-v3 x86-64-v2 x86-64 native)
    set(SET_TARGET_CPU "-march=${TARGET_CPU}")
endif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")

# for Intel oneAPI
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    set(TARGET_CPU core-avx2 CACHE STRING "target CPU architecture")
    set_property(CACHE TARGET_CPU PROPERTY STRINGS icelake-server cascadelake sapphirerapids alderlake rocketlake core-avx2)

    if(${TARGET_CPU} STREQUAL core-avx2)
        set(SET_TARGET_CPU "-mtune=${TARGET_CPU}")
    else(${TARGET_CPU} STREQUAL core-avx2)
        set(SET_TARGET_CPU "-march=${TARGET_CPU}")
    endif(${TARGET_CPU} STREQUAL core-avx2)
endif("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")

# for NVIDIA HPC SDK
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
    set(TARGET_CPU native CACHE STRING "target CPU architecture")
    set_property(CACHE TARGET_CPU PROPERTY STRINGS zen3 zen2 skylake native)
    set(SET_TARGET_CPU "-tp=${TARGET_CPU}")
endif("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")

# output the specified CPU
message(STATUS "TARGET_CPU = ${TARGET_CPU}")
message(STATUS "SET_TARGET_CPU = ${SET_TARGET_CPU}")
