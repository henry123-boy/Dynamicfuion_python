cmake_minimum_required(VERSION 3.18)


# region ====================== Define switches ====================================================
# the user can use these to control the build
option(WITH_OPENMP                "Use OpenMP multi-threading"               ON )
option(DEVELOPER_BUILD            "Add +commit_hash to the project version number" ON )


# 3rd-party build options
if(LINUX_AARCH64 OR APPLE_AARCH64)
    option(USE_BLAS               "Use BLAS/LAPACK instead of MKL"           ON )
else()
    option(USE_BLAS               "Use BLAS/LAPACK instead of MKL"           OFF)
endif()

option(USE_SYSTEM_EIGEN3          "Use system pre-installed eigen3"          OFF)
option(USE_SYSTEM_FLANN           "Use system pre-installed flann"           OFF)
option(USE_SYSTEM_PYBIND11        "Use system pre-installed pybind11"        OFF)
option(USE_SYSTEM_CATCH2          "Use system pre-installed Catch2"          OFF)
option(USE_SYSTEM_GLEW            "Use system pre-installed glew"            OFF)
option(USE_SYSTEM_PNG             "Use system pre-installed png"             OFF)

option(BUILD_CUDA_MODULE          "Build the CUDA module"                    ON)
option(BUILD_CPP_TESTS            "Build C++ tests"                          OFF)
option(SET_UP_U_2_NET             "Set up a slimmed-down version of U^2 Net for mask generation using Saliency Object Detection (requires a large ~177 Mb  download to a local folder)" ON)

option(GLIBCXX_USE_CXX11_ABI      "Set use the newer (C++11 and later) ABI (Relevant to GCC compiler only)"           OFF)
# endregion =======================================================================================

### CMake configuration based on switches
if(BUILD_CPP_TESTS)
    enable_testing()
endif()
if(USE_BLAS)
    option(BUILD_BLAS_FROM_SOURCE "Build OpenBLAS from source"               ON )
else()
    option(BUILD_BLAS_FROM_SOURCE "Build OpenBLAS from source"               OFF)
endif()

if(GLIBCXX_USE_CXX11_ABI)
    set(CXX11_ABI_usage "1")
else()
    set(CXX11_ABI_usage "0")
endif()

### Diagnostic/reporting CMAKE messaging
message(STATUS "Force GLIBCXX_USE_CXX11_ABI=" "${CXX11_ABI_usage}")

### Set up CCACHE as the default compiler launcher
# (to speed up builds on compatible systems)
find_program(CCACHE "ccache")
if (CCACHE)
    message(STATUS "ccache found at ${CCACHE}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    if(BUILD_CUDA_MODULE)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
    endif()
endif()

# In ExternalProject_Add, if NNRT_THIRD_PARTY_DOWNLOAD_DIR is specified, CMake will
# use this directory to cache downloaded 3rd party dependencies and automatically skip
# downloading from the Internet if the files are available. This is only supported by
# a limited number of 3rd party libraries.
set(NNRT_THIRD_PARTY_DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty_downloads")
message(STATUS "Downloading 3rdparty dependencies to ${NNRT_THIRD_PARTY_DOWNLOAD_DIR}")


### Parse NNRT version number
file(STRINGS "csrc/version.txt" NNRT_VERSION_READ)
foreach(ver ${NNRT_VERSION_READ})
    if (ver MATCHES "NNRT_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(NNRT_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()

set(NNRT_VERSION_DEVHASH "")

if(DEVELOPER_BUILD)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" log --pretty=format:%h -n 1
                    OUTPUT_VARIABLE GIT_REV)
    if (GIT_REV)
        set(NNRT_VERSION_DEVHASH "+${GIT_REV}")
    endif()
endif()

string(CONCAT NNRT_VERSION
    "${NNRT_VERSION_MAJOR}"
    ".${NNRT_VERSION_MINOR}"
    ".${NNRT_VERSION_PATCH}"
)
set(NNRT_VERSION_FULL "${NNRT_VERSION}${NNRT_VERSION_DEVHASH}")

project(NNRT VERSION ${NNRT_VERSION} LANGUAGES C CXX)
message(STATUS "NNRT ${NNRT_VERSION_FULL}")

# Set installation paths
if(UNIX OR CYGWIN)
    include(GNUInstallDirs)
    set(NNRT_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(NNRT_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(NNRT_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(NNRT_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else()
    set(NNRT_INSTALL_INCLUDE_DIR include)
    set(NNRT_INSTALL_BIN_DIR bin)
    set(NNRT_INSTALL_LIB_DIR lib)
    set(NNRT_INSTALL_CMAKE_DIR CMake)
endif()


# Set up CUDA settings
if(BUILD_CUDA_MODULE)
    find_package(CUDAToolkit REQUIRED)

    if(BUILD_COMMON_CUDA_ARCHS)
        # Build with all known (and deprecated) architectures.
        # See https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#building-for-maximum-compatibility.
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
            set(CMAKE_CUDA_ARCHITECTURES 35-real 37-real 50-real 52-real 53-real
                60-real 61-real 62-real 70-real 72-real 75-real 80-real 86)
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
            set(CMAKE_CUDA_ARCHITECTURES 35-real 37-real 50-real 52-real 53-real
                60-real 61-real 62-real 70-real 72-real 75-real 80)
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.0")
            set(CMAKE_CUDA_ARCHITECTURES 30-real 32-real 35-real 37-real 50-real
                52-real 53-real 60-real 61-real 62-real 70-real 72-real 75)
        else()
            message(STATUS "Old CUDA <10.0 detected. Falling back to CMake's default architectures.")
        endif()
    else()
        if(CMAKE_CUDA_ARCHITECTURES)
            message(STATUS "Building with user-provided architectures")
        else()
            file(WRITE
                "${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.c"
                "
                #include <stdio.h>
                #include <cuda_runtime_api.h>
                int main() {
                    int n;
                    if (cudaGetDeviceCount(&n) == cudaSuccess) {
                        for (int i = 0; i < n; ++i) {
                            int major, minor;
                            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                                    i) == cudaSuccess &&
                                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                                    i) == cudaSuccess) {
                                if (i > 0) {
                                    printf(\";\");
                                }
                                printf(\"%d%d-real\", major, minor);
                            }
                        }
                    }
                    return 0;
                }
                ")
            try_run(
                DETECTION_RETURN_VALUE DETECTION_COMPILED
                "${CMAKE_CURRENT_BINARY_DIR}"
                "${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.c"
                LINK_LIBRARIES CUDA::cudart
                RUN_OUTPUT_VARIABLE DETECTED_ARCHITECTURES)
            if(DETECTED_ARCHITECTURES)
                message(STATUS "Building with detected architectures.")
                set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCHITECTURES})
            else()
                message(STATUS "Failed to detect architectures. Falling back to CMake's default architectures.")
            endif()
        endif()
    endif()

    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()


### Set installation paths
if(UNIX OR CYGWIN)
    include(GNUInstallDirs)
    set(NNRT_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(NNRT_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(NNRT_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(NNRT_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else()
    set(NNRT_INSTALL_INCLUDE_DIR include)
    set(NNRT_INSTALL_BIN_DIR bin)
    set(NNRT_INSTALL_LIB_DIR lib)
    set(NNRT_INSTALL_CMAKE_DIR CMake)
endif()

### project definition, build configuration, compiler standard

project(NNRT VERSION ${NNRT_VERSION} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "NNRT ${NNRT_VERSION_FULL}")

### global functions

function(nnrt_set_global_properties target)
    # Libraries need to be compiled with position independent code
    get_target_property(target_type ${target} TYPE)
    if (NOT target_type STREQUAL "EXECUTABLE")
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    if(BUILD_CUDA_MODULE)
        target_compile_definitions(${target} PRIVATE BUILD_CUDA_MODULE)
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
        endif()

        if(BUILD_CACHED_CUDA_MANAGER)
            target_compile_definitions(${target} PRIVATE BUILD_CACHED_CUDA_MANAGER)
        endif()
    endif()
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")
#    if(GLIBCXX_USE_CXX11_ABI)
#        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
#    else()
#        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
#    endif()
    if (USE_BLAS)
        target_compile_definitions(${target} PRIVATE USE_BLAS)
    endif()
endfunction()

### includes for dependency processing

include(3rd-party/find_dependencies.cmake)

### subdirectories

add_subdirectory(csrc)

