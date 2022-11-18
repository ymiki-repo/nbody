# compile direct N-body code (base implementation)
project(omp_target CXX)
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
# include(${USER_MODULE_PATH}/set_target_gpu.cmake)
target_compile_options(${PROJECT_NAME}
    PRIVATE

    # option to enable GPU offloading
    # -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa
    # -march=gfx1032

    # for NVIDIA HPC SDK (OpenMP offloading: -mp=[gpu multicore], ISO C++17: -stdpar=[gpu multicore], -acc=[gpu multicore])
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>>:-mp=gpu -gpu=managed>
    $<$<AND:$<NOT:$<BOOL:${GPU_EXECUTION}>>,$<CXX_COMPILER_ID:NVHPC>>:-mp=multicore>
)
