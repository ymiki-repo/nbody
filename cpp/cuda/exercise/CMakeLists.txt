# compile direct N-body code (GPU implementation using CUDA C++)
project(cuda_excercise CXX CUDA)
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
target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<BOOL:${OVERWRITE_DEFAULT}>:NTHREADS=\(${NTHREADS}U\)>
    $<$<BOOL:${OVERWRITE_DEFAULT}>:NUNROLL=\(${NUNROLL}\)>
)

if(${NVIDIA_GPU_HARD} EQUAL ${NVIDIA_GPU_SOFT})
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${NVIDIA_GPU_HARD})
else(${NVIDIA_GPU_HARD} EQUAL ${NVIDIA_GPU_SOFT})
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_${NVIDIA_GPU_SOFT},code=sm_${NVIDIA_GPU_HARD}>
    )
endif(${NVIDIA_GPU_HARD} EQUAL ${NVIDIA_GPU_SOFT})
