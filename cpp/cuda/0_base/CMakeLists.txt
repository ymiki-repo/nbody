# compile direct N-body code (base implementation in CUDA C++: no kernel optimization, use of Unified Memory)
project(base CXX)
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

# apply common settings
include(${USER_MODULE_PATH}/apply_common_settings.cmake)

# add GPU configuration
target_compile_options(${PROJECT_NAME}
    PRIVATE

    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVIDIA>>:${SET_TARGET_GPU}>
)
