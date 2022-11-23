# compile direct N-body code (implementation using C++17 Parallel Algorithms version)
project(stdpar CXX)
message(STATUS "Generate ${PROJECT_NAME} ...")

# define the executable
if(HERMITE_SCHEME)
    set(SOURCES ${PROJECT_SOURCE_DIR}/nbody_hermite4.cpp)
else(HERMITE_SCHEME)
    set(SOURCES ${PROJECT_SOURCE_DIR}/nbody_leapfrog2.cpp)
endif(HERMITE_SCHEME)

add_executable(${PROJECT_NAME}
    ${SOURCES}
)

# apply common settings
include(${USER_MODULE_PATH}/apply_common_settings.cmake)

# add GPU configuration
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>>:-stdpar=gpu -gpu=${NVIDIA_GPU_HARD},managed -Minfo=accel,opt,stdpar>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<CXX_COMPILER_ID:NVHPC>>:${OpenMP_CXX_FLAGS}=multicore>
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>>:-stdpar=gpu>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<CXX_COMPILER_ID:NVHPC>>:${OpenMP_CXX_FLAGS}=multicore>
)

# # additional recipe for NVIDIA HPC SDK with CMake
# find_package(CUDA REQUIRED)
# target_link_libraries(${PROJECT_NAME}
# PRIVATE
# ${CUDA_LIBRARIES})
