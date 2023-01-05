if(NOT BENCHMARK_MODE)
    add_subdirectory(0_dist) # implementation using distribute directives (data transfer via Unified Memory)
    add_subdirectory(a_loop) # implementation using loop directives (data transfer via Unified Memory)
endif(NOT BENCHMARK_MODE)

add_subdirectory(1_dist_data) # implementation using distribute directives (data transfer via data directives)
add_subdirectory(b_loop_data) # implementation using loop directives (data transfer via data directives)

if(EXERCISE_MODE)
    add_subdirectory(exercise)
endif(EXERCISE_MODE)
