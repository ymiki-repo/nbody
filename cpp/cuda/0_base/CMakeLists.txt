# compile direct N-body code (base implementation in CUDA C++: no kernel optimization, use of Unified Memory)
project(cuda_base CXX CUDA)
message(STATUS "Generate ${PROJECT_NAME} ...")

# define the executable
if(HERMITE_SCHEME)
    set(SOURCES ${PROJECT_SOURCE_DIR}/nbody_hermite4.cu)
else(HERMITE_SCHEME)
    set(SOURCES ${PROJECT_SOURCE_DIR}/nbody_leapfrog2.cu)
endif(HERMITE_SCHEME)

add_executable(${PROJECT_NAME}
    ${SOURCES}
)

find_package(CUDA REQUIRED)

# apply common settings
include(${USER_MODULE_PATH}/apply_common_settings.cmake)

# add GPU configuration
# set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${SET_TARGET_GPU})
if("${TARGET_GPU}" MATCHES "NVIDIA_CC90")
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 90)
elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC80")
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 80)
elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC70")
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70)
elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC60")
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60)
elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC86")
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 86)
elseif("${TARGET_GPU}" MATCHES "NVIDIA_CC61")
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)
endif("${TARGET_GPU}" MATCHES "NVIDIA_CC90")
