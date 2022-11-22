# for NVIDIA HPC SDK
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
    option(RELAX_RSQRT_ACCURACY "On to relax precision for reciprocal square root to accelerate simulations" OFF)
    message(STATUS "RELAX_RSQRT_ACCURACY is ${RELAX_RSQRT_ACCURACY}")
endif("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")

# set compilation flags
target_compile_options(${PROJECT_NAME}
    PRIVATE

    # warnings
    $<$<OR:$<CXX_COMPILER_ID:Intel>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wshadow -Wcast-qual -Wconversion -Wno-sign-conversion -Wdisabled-optimization -Wuninitialized -Winit-self -Wsign-promo -Wundef -Wunused -Wmissing-declarations>
    $<$<CXX_COMPILER_ID:Intel>:-Wcheck -Wdeprecated -Wformat -Wsign-compare -Wstrict-prototypes -Wtrigraphs -Wwrite-strings -Wunknown-pragmas -Wunused-function -Wunused-variable>
    $<$<OR:$<CXX_COMPILER_ID:IntelLLVM>,$<CXX_COMPILER_ID:Clang>>:-Weverything -Wno-unknown-warning-option -Wno-documentation -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-zero-as-null-pointer-constant>
    $<$<CXX_COMPILER_ID:GNU>:-Wbad-function-cast -Wcast-align -Wdouble-promotion -Wlogical-op -Wstrict-null-sentinel -Wuseless-cast>
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER:CXX_COMPILER_VERSION,4.9>>:-fdiagnostics-color>

    # disable warning #11071: Inline not honored for call * in icpc
    $<$<CXX_COMPILER_ID:Intel>:-diag-disable=11071>

    # OpenMP
    # $<$<BOOL:${OpenMP_FOUND}>:${OpenMP_CXX_FLAGS}>
    $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<NOT:$<CXX_COMPILER_ID:NVHPC>>>:${OpenMP_CXX_FLAGS}>

    # memory sanitizer
    $<$<BOOL:${USE_SANITIZER_ADDRESS}>:-fsanitize=address>
    $<$<BOOL:${USE_SANITIZER_LEAK}>:-fsanitize=leak>
    $<$<BOOL:${USE_SANITIZER_UNDEFINED}>:-fsanitize=undefined>
    $<$<BOOL:${USE_SANITIZER_THREAD}>:-fsanitize=thread>

    # performance optimization
    $<$<NOT:$<CONFIG:Debug>>:-O3>
    $<$<CXX_COMPILER_ID:Intel>:-restrict>
    $<$<AND:$<NOT:$<CONFIG:Debug>>,$<CXX_COMPILER_ID:Intel>>:-ip -xHost -qopt-report=3>
    $<$<AND:$<NOT:$<CONFIG:Debug>>,$<OR:$<CXX_COMPILER_ID:IntelLLVM>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>>:-ffast-math -funroll-loops>
    $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<BOOL:${RELAX_RSQRT_ACCURACY}>>:-Mfprelaxed=rsqrt>

    # specify CPU architecture
    ${SET_TARGET_CPU}

    # flags for CUDA
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v,-warn-spills,-warn-lmem-usage -lineinfo>
    # NOTE: to call a constexpr __host__ function from a __global__ function (experimental flag)
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
