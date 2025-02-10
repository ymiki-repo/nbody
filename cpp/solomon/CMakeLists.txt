# compile direct N-body code (implementation using Solomon)
project(solomon CXX)
message(STATUS "Generate ${PROJECT_NAME} ...")

option(SOLOMON_ON_OPENACC "ON to use OpenACC as backend of Solomon (OFF uses OpenMP target as backend of Solomon)" ON)

if(${SOLOMON_ON_OPENACC})
    option(SOLOMON_USES_PARALLEL_CONSTRUCT "ON to use parallel construct in OpenACC (OFF uses kernels construct)" ON)
else(${SOLOMON_ON_OPENACC})
    option(SOLOMON_USES_LOOP_DIRECTIVE "ON to use loop directive in OpenMP target (OFF uses distribute directive)" ON)
endif(${SOLOMON_ON_OPENACC})

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

# include Solomon
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/solomon/solomon
)

# add GPU configuration
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${SOLOMON_ON_OPENACC}>>:OFFLOAD_BY_OPENACC>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${SOLOMON_ON_OPENACC}>,$<BOOL:${SOLOMON_USES_PARALLEL_CONSTRUCT}>>:OFFLOAD_BY_OPENACC_PARALLEL>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<NOT:$<BOOL:${SOLOMON_ON_OPENACC}>>>:OFFLOAD_BY_OPENMP_TARGET>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<NOT:$<BOOL:${SOLOMON_ON_OPENACC}>>,$<NOT:$<BOOL:${SOLOMON_USES_LOOP_DIRECTIVE}>>>:OFFLOAD_BY_OPENMP_TARGET_DISTRIBUTE>
)
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${SOLOMON_ON_OPENACC}>>:-acc=gpu>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<NOT:$<BOOL:${SOLOMON_ON_OPENACC}>>>:-mp=gpu>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${SOLOMON_ON_OPENACC}>>:${OpenMP_CXX_FLAGS}=multicore>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${RELAX_RSQRT_ACCURACY}>>:-gpu=cc${NVIDIA_GPU_HARD},fastmath>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<NOT:$<BOOL:${RELAX_RSQRT_ACCURACY}>>>:-gpu=cc${NVIDIA_GPU_HARD}>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>>:-Minfo=accel,opt,mp>
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${SOLOMON_ON_OPENACC}>>:-acc=gpu>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<CXX_COMPILER_ID:NVHPC>,$<NOT:$<BOOL:${SOLOMON_ON_OPENACC}>>>:-mp=gpu>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${SOLOMON_ON_OPENACC}>>:${OpenMP_CXX_FLAGS}=multicore>
)
