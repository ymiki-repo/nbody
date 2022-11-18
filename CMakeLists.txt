cmake_minimum_required(VERSION 3.1) # to set CMAKE_CXX_STANDARD

# enable_testing() # activate CTest
message(STATUS "CMAKE_GENERATOR is ${CMAKE_GENERATOR}")
message(STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

# global setting(s)
set(USER_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# global options
option(BENCHMARK_MODE "On to perform benchmark" OFF)
option(CALCULATE_POTENTIAL "On to calculate gravitational potential" ON)
option(HERMITE_SCHEME "On to adopt 4th-order Hermite scheme" OFF)
message(STATUS "BENCHMARK_MODE is ${BENCHMARK_MODE}")
message(STATUS "CALCULATE_POTENTIAL is ${CALCULATE_POTENTIAL}")
message(STATUS "HERMITE_SCHEME is ${HERMITE_SCHEME}")

# enforce Release as build type in benchmark mode
if(BENCHMARK_MODE)
    set(CMAKE_BUILD_TYPE "Release")
endif(BENCHMARK_MODE)

# options for code development
option(USE_SANITIZER_ADDRESS "On to enable AddressSanitizer, a fast memory error detector" OFF)
option(USE_SANITIZER_LEAK "On to enable LeakSanitizer, a memory leak detector" OFF)
option(USE_SANITIZER_UNDEFINED "On to enable UndefinedBehaviorSanitizer, a fast undefined behavior detector" OFF)
option(USE_SANITIZER_THREAD "On to enable ThreadSanitizer, a fast data race detector" OFF)

if(USE_SANITIZER_THREAD)
    # -fsanitize=thread cannot be combined with -fsanitize=address or -fsanitize=leak
    set(USE_SANITIZER_ADDRESS OFF)
    set(USE_SANITIZER_LEAK OFF)
endif(USE_SANITIZER_THREAD)

message(STATUS "USE_SANITIZER_ADDRESS is ${USE_SANITIZER_ADDRESS}")
message(STATUS "USE_SANITIZER_LEAK is ${USE_SANITIZER_LEAK}")
message(STATUS "USE_SANITIZER_UNDEFINED is ${USE_SANITIZER_UNDEFINED}")
message(STATUS "USE_SANITIZER_THREAD is ${USE_SANITIZER_THREAD}")

# check loaded modulefiles
find_program(MODULECMD_CONFIG "modulecmd")

if(MODULECMD_CONFIG)
    get_filename_component(SH_NAME "$ENV{SHELL}" NAME)
    execute_process(COMMAND ${MODULECMD_CONFIG} ${SH_NAME} list OUTPUT_VARIABLE LOADED_MODULE_LIST)
    message(STATUS "${LOADED_MODULE_LIST}")
endif(MODULECMD_CONFIG)

# set C++ compiler
if(NOT DEFINED ENV{CXX})
    # find CXX compiler
    set(TMP_LIST "nvc++" "icpx" "clang++" "g++")

    foreach(CANDIDATE IN LISTS TMP_LIST)
        find_program(BASE_CXX ${CANDIDATE})

        if(BASE_CXX)
            set(CMAKE_CXX_COMPILER_INIT ${BASE_CXX})
            break()
        endif(BASE_CXX)
    endforeach(CANDIDATE IN LISTS TMP_LIST)
endif(NOT DEFINED ENV{CXX})

# set C compiler
if(NOT DEFINED ENV{CC})
    # find C compiler
    set(TMP_LIST "nvc" "icx" "clang" "gcc")

    foreach(CANDIDATE IN LISTS TMP_LIST)
        find_program(BASE_CC ${CANDIDATE})

        if(BASE_CC)
            set(CMAKE_C_COMPILER_INIT ${BASE_CC})
            break()
        endif(BASE_CC)
    endforeach(CANDIDATE IN LISTS TMP_LIST)
endif(NOT DEFINED ENV{CC})

# set C/CXX compilers
project(dummy NONE)
set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER_INIT})
set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER_INIT})
message(STATUS "CMAKE_CXX_COMPILER is ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_C_COMPILER is ${CMAKE_C_COMPILER}")

# specify CXX standard
set(CMAKE_CXX_STANDARD 17) # make_unique() is introduced in C++14, std::execution is introduced in C++17
set(CMAKE_CXX_STANDARD_REQUIRED_ON)
set(CMAKE_CXX_EXTENSIONS OFF) # use -std=c++17 instead of -std=gnu++17

# specify options for Makefile
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_COLOR_MAKEFILE ON)

# set directory for executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# generate directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dat)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/fig)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/log)

file(COPY sh DESTINATION ${CMAKE_BINARY_DIR})
file(COPY jl DESTINATION ${CMAKE_BINARY_DIR})
file(COPY README.md DESTINATION ${CMAKE_BINARY_DIR})
file(COPY LICENSE.txt DESTINATION ${CMAKE_BINARY_DIR})
file(COPY .editorconfig DESTINATION ${CMAKE_BINARY_DIR})

# preparation for finding library path
string(REPLACE ":" ";" SEARCH_PATH $ENV{CPATH})

# set SIMD_BITS
# future update: automatic determination of SIMD width by using the CPU/GPU/FPGA information
set(SIMD_BITS 512 CACHE STRING "SIMD width in units of bit: 128, 256, or 512")
set_property(CACHE SIMD_BITS PROPERTY STRINGS 128 256 512)
math(EXPR MEMORY_ALIGNMENT "${SIMD_BITS} >> 3") # 1 byte = 8 bits
message(STATUS "SIMD_BITS = ${SIMD_BITS}")
message(STATUS "MEMORY_ALIGNMENT = ${MEMORY_ALIGNMENT}")

# set FP_L
set(FP_L 32 CACHE STRING "Specification of floating-point numbers (low-precision): 32, 64, or 128")
set_property(CACHE FP_L PROPERTY STRINGS 32 64 128)

if((NOT ${FP_L} EQUAL 32) AND(NOT ${FP_L} EQUAL 64) AND(NOT ${FP_L} EQUAL 128))
    set(FP_L 32)
endif((NOT ${FP_L} EQUAL 32) AND(NOT ${FP_L} EQUAL 64) AND(NOT ${FP_L} EQUAL 128))

message(STATUS "FP_L = ${FP_L}")

# set FP_M
set(FP_M 64 CACHE STRING "Specification of floating-point numbers (medium-precision): 32, 64, or 128")
set_property(CACHE FP_M PROPERTY STRINGS 32 64 128)

if((NOT ${FP_M} EQUAL 32) AND(NOT ${FP_M} EQUAL 64) AND(NOT ${FP_M} EQUAL 128))
    set(FP_M 64)
endif((NOT ${FP_M} EQUAL 32) AND(NOT ${FP_M} EQUAL 64) AND(NOT ${FP_M} EQUAL 128))

if(${FP_L} GREATER ${FP_M})
    set(FP_M ${FP_L})
endif(${FP_L} GREATER ${FP_M})

message(STATUS "FP_M = ${FP_M}")

# set FP_H
set(FP_H 64 CACHE STRING "Specification of floating-point numbers (high-precision): 64 or 128")
set_property(CACHE FP_H PROPERTY STRINGS 64 128)

if((NOT ${FP_H} EQUAL 64) AND(NOT ${FP_H} EQUAL 128))
    set(FP_H 64)
endif((NOT ${FP_H} EQUAL 64) AND(NOT ${FP_H} EQUAL 128))

if(${FP_M} GREATER ${FP_H})
    set(FP_H ${FP_M})
endif(${FP_M} GREATER ${FP_H})

message(STATUS "FP_H = ${FP_H}")

# add subdirectories
add_subdirectory(cpp/base)
add_subdirectory(cpp/omp_simd)
