# compile direct N-body code (GPU implementation using OpenMP target directives)
project(omp_exercise CXX)
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
target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<BOOL:${OVERWRITE_DEFAULT}>>:NTHREADS=\(${NTHREADS}U\)>
)

# add GPU configuration
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${RELAX_RSQRT_ACCURACY}>>:-mp=gpu -gpu=cc${NVIDIA_GPU_HARD},managed,fastmath -Minfo=accel,opt,mp>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<NOT:$<BOOL:${RELAX_RSQRT_ACCURACY}>>>:-mp=gpu -gpu=cc${NVIDIA_GPU_HARD},managed -Minfo=accel,opt,mp>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<NOT:$<BOOL:${GPU_EXECUTION}>>,$<CXX_COMPILER_ID:NVHPC>>:-mp=multicore>
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>>:-mp=gpu>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>>:-gpu=managed>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<NOT:$<BOOL:${GPU_EXECUTION}>>,$<CXX_COMPILER_ID:NVHPC>>:-mp=multicore>
)
