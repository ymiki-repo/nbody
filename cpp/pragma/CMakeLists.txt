# compile direct N-body code (GPU implementation using OpenACC/OpenMP switching macro)
project(pragma CXX)
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

# option
option(OFFLOAD_BY_OPENACC "On to use OpenACC for GPU offloading" ON)
option(OFFLOAD_BY_OPENMP_TARGET "On to use OpenMP target directive for GPU offloading" ON)

if(OFFLOAD_BY_OPENACC)
    set(OFFLOAD_BY_OPENMP_TARGET OFF)
endif(OFFLOAD_BY_OPENACC)

message(STATUS "OFFLOAD_BY_OPENACC is ${OFFLOAD_BY_OPENACC}")
message(STATUS "OFFLOAD_BY_OPENMP_TARGET is ${OFFLOAD_BY_OPENMP_TARGET}")

# apply common settings
include(${USER_MODULE_PATH}/apply_common_settings.cmake)
target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<BOOL:${OFFLOAD_BY_OPENACC}>:OFFLOAD_BY_OPENACC>
    $<$<BOOL:${OFFLOAD_BY_OPENMP_TARGET}>:OFFLOAD_BY_OPENMP_TARGET>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<BOOL:${OVERWRITE_DEFAULT}>>:NTHREADS=\(${NTHREADS}U\)>
)

# add GPU configuration
target_compile_options(${PROJECT_NAME} PRIVATE

    # for OpenACC offloading
    $<$<AND:$<BOOL:${OFFLOAD_BY_OPENACC}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${GPU_EXECUTION}>>:-acc=gpu>
    $<$<AND:$<BOOL:${OFFLOAD_BY_OPENACC}>,$<CXX_COMPILER_ID:NVHPC>,$<NOT:$<BOOL:${GPU_EXECUTION}>>>:-acc=multicore>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${OFFLOAD_BY_OPENACC}>,$<CXX_COMPILER_ID:NVHPC>>:${OpenMP_CXX_FLAGS}=multicore>

    # for OpenMP target offloading
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${OFFLOAD_BY_OPENMP_TARGET}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${GPU_EXECUTION}>>:${OpenMP_CXX_FLAGS}=gpu>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${OFFLOAD_BY_OPENMP_TARGET}>,$<CXX_COMPILER_ID:NVHPC>,$<NOT:$<BOOL:${GPU_EXECUTION}>>>:${OpenMP_CXX_FLAGS}=multicore>

    # specify GPU generation
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>>:-gpu=cc${NVIDIA_GPU_HARD}>

    # enhance compiler messages
    $<$<AND:$<BOOL:${OFFLOAD_BY_OPENACC}>,$<CXX_COMPILER_ID:NVHPC>>:-Minfo=accel,opt>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${OFFLOAD_BY_OPENMP_TARGET}>,$<CXX_COMPILER_ID:NVHPC>>:-Minfo=accel,opt,mp>
)
target_link_libraries(${PROJECT_NAME} PRIVATE

    # for OpenACC offloading
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<BOOL:${OFFLOAD_BY_OPENACC}>,$<CXX_COMPILER_ID:NVHPC>>:-acc=gpu>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${OFFLOAD_BY_OPENACC}>,$<CXX_COMPILER_ID:NVHPC>>:${OpenMP_CXX_FLAGS}=multicore>

    # for OpenMP target offloading
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${OFFLOAD_BY_OPENMP_TARGET}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${GPU_EXECUTION}>>:${OpenMP_CXX_FLAGS}=gpu>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${OFFLOAD_BY_OPENMP_TARGET}>,$<CXX_COMPILER_ID:NVHPC>,$<NOT:$<BOOL:${GPU_EXECUTION}>>>:${OpenMP_CXX_FLAGS}=multicore>
)
