# # CMAKE_<LANG>_COMPILER_ID
# ARMCC = ARM Compiler (arm.com)
# ARMClang = ARM Compiler based on Clang (arm.com)
# Clang = LLVM Clang (clang.llvm.org)
# Cray = Cray Compiler (cray.com)
# Fujitsu = Fujitsu HPC compiler (Trad mode)
# FujitsuClang = Fujitsu HPC compiler (Clang mode)
# GNU = GNU Compiler Collection (gcc.gnu.org)
# HP = Hewlett-Packard Compiler (hp.com)
# Intel = Intel Compiler (intel.com)
# IntelLLVM = Intel LLVM-Based Compiler (intel.com)
# NVHPC = NVIDIA HPC SDK Compiler (nvidia.com)
# NVIDIA = NVIDIA CUDA Compiler (nvidia.com)

# GCC: -march=[x86-64-v4 cascadelake icelake-server icelake-client sapphirerapids alderlake rocketlake znver2 znver3]
# LLVM: -march=[x86-64-v4 cascadelake icelake-server icelake-client sapphirerapids alderlake rocketlake znver2 znver3] (check by $ llc --version; $ llc -march=ARCH -mattr=help # ARCH = [amdgcn, x86-64])
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(TARGET_CPU native CACHE STRING "target CPU architecture")
    set_property(CACHE TARGET_CPU PROPERTY STRINGS znver3 znver2 cascadelake icelake-server icelake-client sapphirerapids alderlake rocketlake x86-64-v4 x86-64-v3 x86-64-v2 x86-64 native)
    set(SET_TARGET_CPU "-march=${TARGET_CPU}")
endif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")

# Intel LLVM: -march=[cascadelake icelake-server icelake-client sapphirerapids alderlake rocketlake core-avx2]
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

# NVHPC: -tp=[zen3 zen2 skylake native]
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
    set(TARGET_CPU native CACHE STRING "target CPU architecture")
    set_property(CACHE TARGET_CPU PROPERTY STRINGS zen3 zen2 skylake native)
    set(SET_TARGET_CPU "-tp=${TARGET_CPU}")
endif("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")

# output the specified CPU
message(STATUS "TARGET_CPU = ${TARGET_CPU}")
message(STATUS "SET_TARGET_CPU = ${SET_TARGET_CPU}")
