# set CPU architecture
include(${USER_MODULE_PATH}/set_target_cpu.cmake)

# set GPU architecture
if(GPU_EXECUTION)
    include(${USER_MODULE_PATH}/set_target_gpu.cmake)
endif(GPU_EXECUTION)

# find OpenMP
find_package(OpenMP)

# if(NOT DEFINED OMP_THREADS)
# cmake_host_system_information(RESULT PHYSICAL_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
# set(OMP_THREADS ${PHYSICAL_CORES})
# endif(NOT DEFINED OMP_THREADS)

# set compilation flags
include(${USER_MODULE_PATH}/set_compile_flag.cmake)

# find Boost
set(BOOST_INCLUDEDIR ${SEARCH_PATH})

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
    set(Boost_NO_WARN_NEW_VERSIONS ON)
endif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_DEBUG_LIBS OFF)
set(Boost_USE_RELEASE_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS program_options filesystem timer system)

# for Intel oneAPI
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    find_package(TBB REQUIRED)
endif("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")

# find HDF5
enable_language(C)
find_package(HDF5 REQUIRED COMPONENTS C)

# link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${Boost_LIBRARIES}
    ${HDF5_LIBRARIES}

    # OpenMP
    # $<$<BOOL:${OpenMP_FOUND}>:${OpenMP_CXX_FLAGS}>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<NOT:$<CXX_COMPILER_ID:NVHPC>>>:${OpenMP_CXX_FLAGS}>

    # link tbb for C++ parallel algorithms
    $<$<NOT:$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:IntelLLVM>,$<CXX_COMPILER_ID:Intel>>>:-ltbb>
    $<$<OR:$<CXX_COMPILER_ID:IntelLLVM>,$<CXX_COMPILER_ID:Intel>>:TBB::tbb>

    # memory sanitizer
    $<$<BOOL:${USE_SANITIZER_ADDRESS}>:-fsanitize=address>
    $<$<BOOL:${USE_SANITIZER_LEAK}>:-fsanitize=leak>
    $<$<BOOL:${USE_SANITIZER_UNDEFINED}>:-fsanitize=undefined>
    $<$<BOOL:${USE_SANITIZER_THREAD}>:-fsanitize=thread>
)

# include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/../..
)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
)

# add definitions
target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    $<$<BOOL:${SUPPRESS_PRINT_INFO}>:SUPPRESS_PRINT_INFO>
    $<$<BOOL:${SUPPRESS_PRINT_ALERT}>:SUPPRESS_PRINT_ALERT>

    # # OpenMP
    # $<$<BOOL:${OpenMP_FOUND}>:CPUS_PER_PROCESS=\(${OMP_THREADS}\)>
    SIMD_BITS=\(${SIMD_BITS}\)
    MEMORY_ALIGNMENT=\(${MEMORY_ALIGNMENT}\)
    FP_L=\(${FP_L}\)
    FP_M=\(${FP_M}\)
    FP_H=\(${FP_H}\)
    $<$<BOOL:${BENCHMARK_MODE}>:BENCHMARK_MODE>
    $<$<BOOL:${CALCULATE_POTENTIAL}>:CALCULATE_POTENTIAL>
    $<$<BOOL:${HERMITE_SCHEME}>:USE_HERMITE>
)
