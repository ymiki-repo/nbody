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
option(USE_OPENACC_AS_DIRECTIVE "On to use OpenACC for GPU offloading" ON)
option(USE_OPENMP_AS_DIRECTIVE "On to use OpenMP target directive for GPU offloading" ON)

if(USE_OPENACC_AS_DIRECTIVE)
    set(USE_OPENMP_AS_DIRECTIVE OFF)
else(USE_OPENACC_AS_DIRECTIVE)
    set(USE_OPENMP_AS_DIRECTIVE ON)
endif(USE_OPENACC_AS_DIRECTIVE)

message(STATUS "USE_OPENACC_AS_DIRECTIVE is ${USE_OPENACC_AS_DIRECTIVE}")
message(STATUS "USE_OPENMP_AS_DIRECTIVE is ${USE_OPENMP_AS_DIRECTIVE}")

# apply common settings
include(${USER_MODULE_PATH}/apply_common_settings.cmake)
target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<BOOL:${GPU_EXECUTION}>:USE_PRAGMA_OFFLOAD>
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<BOOL:${OVERWRITE_DEFAULT}>>:NTHREADS=\(${NTHREADS}U\)>
)

# add GPU configuration
target_compile_options(${PROJECT_NAME} PRIVATE

    # for OpenACC offloading
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<BOOL:${USE_OPENACC_AS_DIRECTIVE}>,$<CXX_COMPILER_ID:NVHPC>>:-acc=gpu -gpu=cc${NVIDIA_GPU_HARD} -Minfo=accel,opt>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${USE_OPENACC_AS_DIRECTIVE}>,$<CXX_COMPILER_ID:NVHPC>>:${OpenMP_CXX_FLAGS}=multicore>

    # for OpenMP target offloading
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${GPU_EXECUTION}>,$<BOOL:${USE_OPENMP_AS_DIRECTIVE}>,$<CXX_COMPILER_ID:NVHPC>>:-mp=gpu -gpu=cc${NVIDIA_GPU_HARD} -Minfo=accel,opt,mp>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<NOT:$<BOOL:${GPU_EXECUTION}>>,$<BOOL:${USE_OPENMP_AS_DIRECTIVE}>,$<CXX_COMPILER_ID:NVHPC>>:-mp=multicore>
)
target_link_libraries(${PROJECT_NAME} PRIVATE

    # for OpenACC offloading
    $<$<AND:$<BOOL:${GPU_EXECUTION}>,$<BOOL:${USE_OPENACC_AS_DIRECTIVE}>,$<CXX_COMPILER_ID:NVHPC>>:-acc=gpu>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${USE_OPENACC_AS_DIRECTIVE}>,$<CXX_COMPILER_ID:NVHPC>>:${OpenMP_CXX_FLAGS}=multicore>

    # for OpenMP target offloading
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<BOOL:${GPU_EXECUTION}>,$<BOOL:${USE_OPENMP_AS_DIRECTIVE}>,$<CXX_COMPILER_ID:NVHPC>>:-mp=gpu>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<NOT:$<BOOL:${GPU_EXECUTION}>>,$<BOOL:${USE_OPENMP_AS_DIRECTIVE}>,$<CXX_COMPILER_ID:NVHPC>>:-mp=multicore>
)
