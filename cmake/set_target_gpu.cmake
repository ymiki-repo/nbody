set(TARGET_GPU NVIDIA_CC90 CACHE STRING "target GPU architecture")
set_property(CACHE TARGET_GPU PROPERTY STRINGS NVIDIA_CC90 NVIDIA_CC80 NVIDIA_CC70)

# set_property(CACHE TARGET_GPU PROPERTY STRINGS NVIDIA_CC90 NVIDIA_CC80 NVIDIA_CC70 NVIDIA_CC60 NVIDIA_CC86 NVIDIA_CC61 NVIDIA_CC80_60)

# extract compute capability for NVIDIA GPUs
# -gencode arch=compute_XX,code=sm_YY
if("${TARGET_GPU}" MATCHES "^NVIDIA_CC")
    string(LENGTH "${TARGET_GPU}" LENGTH_ALL)

    # remove the tag "NVIDIA_CC" (9 characters)
    math(EXPR NUM_CHARS "${LENGTH_ALL} - 9")
    string(SUBSTRING "${TARGET_GPU}" 9 ${NUM_CHARS} TARGET_NVIDIA_GPU)
    string(FIND "${TARGET_NVIDIA_GPU}" "_" SEPARATE_POSITION)

    if(${SEPARATE_POSITION} EQUAL -1)
        # XX = YY
        set(NVIDIA_GPU_HARD ${TARGET_NVIDIA_GPU})
        set(NVIDIA_GPU_SOFT ${TARGET_NVIDIA_GPU})
    else(${SEPARATE_POSITION} EQUAL -1)
        # XX != YY
        string(SUBSTRING "${TARGET_NVIDIA_GPU}" 0 ${SEPARATE_POSITION} NVIDIA_GPU_HARD)
        math(EXPR HEAD_POS "${SEPARATE_POSITION} + 1")
        string(LENGTH "${TARGET_NVIDIA_GPU}" LENGTH_ALL)
        math(EXPR NUM_STR "${LENGTH_ALL} - ${SEPARATE_POSITION} - 1")
        string(SUBSTRING "${TARGET_NVIDIA_GPU}" ${HEAD_POS} ${NUM_STR} NVIDIA_GPU_SOFT)
    endif(${SEPARATE_POSITION} EQUAL -1)

    message(STATUS "NVIDIA_GPU_HARD = ${NVIDIA_GPU_HARD}")
    message(STATUS "NVIDIA_GPU_SOFT = ${NVIDIA_GPU_SOFT}")

    option(TIGHTLY_COUPLED_CPU_GPU "ON for NVIDIA CPU and GPU fused via NVLink-C2C" ON)

    if((${NVIDIA_GPU_HARD} LESS 90) OR(NOT ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64"))
        # NVIDIA GH200: NVIDIA Grace (aarch64) + NVIDIA H100 (CC90)
        # NVIDIA GB200: NVIDIA Grace (aarch64) + NVIDIA Blackwell (CC100)
        set(TIGHTLY_COUPLED_CPU_GPU OFF)
    endif((${NVIDIA_GPU_HARD} LESS 90) OR(NOT ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64"))

    message(STATUS "TIGHTLY_COUPLED_CPU_GPU = ${TIGHTLY_COUPLED_CPU_GPU}")
endif("${TARGET_GPU}" MATCHES "^NVIDIA_CC")

# output the specified GPU
message(STATUS "TARGET_GPU = ${TARGET_GPU}")
