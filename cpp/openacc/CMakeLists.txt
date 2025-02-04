if(NOT BENCHMARK_MODE)
    add_subdirectory(0_unified) # implementation using Unified Memory
endif(NOT BENCHMARK_MODE)

add_subdirectory(1_data) # implementation using data directives

if(EXERCISE_MODE)
    add_subdirectory(exercise)
endif(EXERCISE_MODE)
