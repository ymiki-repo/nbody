set(TARGET_GPU NVIDIA_CC80 CACHE STRING "target GPU architecture")

# for NVIDIA HPC SDK
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
    set_property(CACHE TARGET_GPU PROPERTY STRINGS NVIDIA_CC90 NVIDIA_CC80 NVIDIA_CC70 NVIDIA_CC86)

    if("${TARGET_GPU}" MATCHES "NVIDIA_CC90")
        set(SET_TARGET_GPU "-gpu=cc90")
    elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC80")
        set(SET_TARGET_GPU "-gpu=cc80")
    elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC70")
        set(SET_TARGET_GPU "-gpu=cc70")
    elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC86")
        set(SET_TARGET_GPU "-gpu=cc86")
    endif("${TARGET_GPU}" MATCHES "NVIDIA_CC90")
endif("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")

# for CUDA C++
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVIDIA")
    set_property(CACHE TARGET_GPU PROPERTY STRINGS NVIDIA_CC90 NVIDIA_CC80 NVIDIA_CC70 NVIDIA_CC86)

    if("${TARGET_GPU}" MATCHES "NVIDIA_CC90")
        set(SET_TARGET_GPU "-gencode arch=compute_90,code=sm_90")
    elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC80")
        set(SET_TARGET_GPU "-gencode arch=compute_80,code=sm_80")
    elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC70")
        set(SET_TARGET_GPU "-gencode arch=compute_70,code=sm_70")
    elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC86")
        set(SET_TARGET_GPU "-gencode arch=compute_86,code=sm_86")
    endif("${TARGET_GPU}" MATCHES "NVIDIA_CC90")
endif("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVIDIA")

# output the specified GPU
message(STATUS "TARGET_GPU = ${TARGET_GPU}")
message(STATUS "SET_TARGET_GPU = ${SET_TARGET_GPU}")
