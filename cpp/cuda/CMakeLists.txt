if(NOT BENCHMARK_MODE)
    add_subdirectory(00_unified_base)
    add_subdirectory(01_unified_rsqrt)
    add_subdirectory(02_unified_shmem)
endif(NOT BENCHMARK_MODE)

add_subdirectory(10_memcpy_base)
add_subdirectory(11_memcpy_rsqrt)
add_subdirectory(12_memcpy_shmem)

if(EXERCISE_MODE)
    add_subdirectory(exercise)
endif(EXERCISE_MODE)
