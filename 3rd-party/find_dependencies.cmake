
#
# NNRT 3rd party library integration
#
set(NNRT_3RDPARTY_DIR "${PROJECT_SOURCE_DIR}/3rd-party")

# EXTERNAL_MODULES
# CMake modules we depend on in our public interface. These are modules we
# need to find_package() in our CMake config script, because we will use their
# targets.
set(NNRT_3RDPARTY_EXTERNAL_MODULES)

# PUBLIC_TARGETS
# CMake targets we link against in our public interface. They are
# either locally defined and installed, or imported from an external module
# (see above).
set(NNRT_3RDPARTY_PUBLIC_TARGETS)

# HEADER_TARGETS
# CMake targets we use in our public interface, but as a special case we do not
# need to link against the library. This simplifies dependencies where we merely
# expose declared data types from other libraries in our public headers, so it
# would be overkill to require all library users to link against that dependency.
set(NNRT_3RDPARTY_HEADER_TARGETS)

# PRIVATE_TARGETS
# CMake targets for dependencies which are not exposed in the public API. This
# will probably include HEADER_TARGETS, but also anything else we use internally.
set(NNRT_3RDPARTY_PRIVATE_TARGETS)


find_package(PkgConfig QUIET)
include(find_package_3rdparty_library)



# List of linker options for nnrt_cpp client binaries (eg: pybind) to hide 3rd-party
# dependencies. Only needed with GCC, not AppleClang.
set(NNRT_HIDDEN_3RDPARTY_LINK_OPTIONS)

if (CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
    find_library(LexLIB libl.a)    # test archive in macOS
    if (LexLIB)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_LINK_OPTIONS -load_hidden ${LexLIB})
        check_cxx_source_compiles("int main() {return 0;}" FLAG_load_hidden)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
    endif ()
endif ()
if (NOT FLAG_load_hidden)
    set(FLAG_load_hidden 0)
endif ()

# CMake arguments for configuring ExternalProjects. Use the second _hidden
# version by default.
set(ExternalProject_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW
    -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )
# Keep 3rd party symbols hidden from NNRT user code. Do not use if 3rd party
# libraries throw exceptions that escape NNRT.
set(ExternalProject_CMAKE_ARGS_hidden
    ${ExternalProject_CMAKE_ARGS}
    # Apply LANG_VISIBILITY_PRESET to static libraries and archives as well
    -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_CUDA_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    )

include(import_3rdparty_library)
include(build_3rdparty_library)
# Convenience function to link against all third-party libraries
# We need this because we create a lot of object libraries to assemble
# the main library
function(nnrt_link_3rdparty_libraries target)
    target_link_libraries(${target} PRIVATE ${NNRT_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${NNRT_3RDPARTY_PUBLIC_TARGETS})
    foreach (dep IN LISTS NNRT_3RDPARTY_HEADER_TARGETS)
        if (TARGET ${dep})
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            if (inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${inc})
            endif ()
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            if (inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
            endif ()
            get_property(def TARGET ${dep} PROPERTY INTERFACE_COMPILE_DEFINITIONS)
            if (def)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${def})
            endif ()
        endif ()
    endforeach ()
endfunction()


# Python
find_package(PythonExecutable REQUIRED) # invokes the module in 3rd-party/CMake

# Threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE) # -pthread instead of -lpthread
find_package(Threads REQUIRED)
list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "Threads")

# OpenMP
if (WITH_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Building with OpenMP")
        set(OPENMP_TARGET "OpenMP::OpenMP_CXX")
        list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS "${OPENMP_TARGET}")
        if (NOT BUILD_SHARED_LIBS)
            list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "OpenMP")
        endif ()
    endif ()
endif ()


# GLEW
if (USE_SYSTEM_GLEW)
    find_package(GLEW)
    if (TARGET GLEW::GLEW)
        message(STATUS "Using installed third-party library GLEW ${GLEW_VERSION}")
        list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "GLEW")
        set(GLEW_TARGET "GLEW::GLEW")
    else ()
        pkg_config_3rdparty_library(3rdparty_glew glew)
        if (3rdparty_glew_FOUND)
            set(GLEW_TARGET "3rdparty_glew")
        else ()
            set(USE_SYSTEM_GLEW OFF)
        endif ()
    endif ()
endif ()
if (NOT USE_SYSTEM_GLEW)
    build_3rdparty_library(3rdparty_glew HEADER DIRECTORY glew SOURCES src/glew.c INCLUDE_DIRS include/)
    if (ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(3rdparty_glew PUBLIC GLEW_OSMESA)
    endif ()
    if (WIN32)
        target_compile_definitions(3rdparty_glew PUBLIC GLEW_STATIC)
    endif ()
    set(GLEW_TARGET "3rdparty_glew")
endif ()
list(APPEND NNRT_3RDPARTY_HEADER_TARGETS "${GLEW_TARGET}")
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS "${GLEW_TARGET}")


# Eigen3
if (USE_SYSTEM_EIGEN3)
    find_package(Eigen3)
    if (TARGET Eigen3::Eigen)
        message(STATUS "Using installed third-party library Eigen3 ${EIGEN3_VERSION_STRING}")
        # Eigen3 is a publicly visible dependency, so add it to the list of
        # modules we need to find in the NNRT config script.
        list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "Eigen3")
        set(EIGEN3_TARGET "Eigen3::Eigen")
    else ()
        message(STATUS "Unable to find installed third-party library Eigen3")
        set(USE_SYSTEM_EIGEN3 OFF)
    endif ()
endif ()
if (NOT USE_SYSTEM_EIGEN3)
    build_3rdparty_library(3rdparty_eigen3 PUBLIC DIRECTORY Eigen INCLUDE_DIRS Eigen INCLUDE_ALL)
    set(EIGEN3_TARGET "3rdparty_eigen3")
endif ()
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS "${EIGEN3_TARGET}")

# fmt
if (USE_SYSTEM_FMT)
    nnrt_find_package_3rdparty_library(3rdparty_fmt
        PACKAGE fmt
        TARGETS fmt::fmt-header-only fmt::fmt
        )
    if (3rdparty_fmt_FOUND)
        if (NOT BUILD_SHARED_LIBS)
            list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "fmt")
        endif ()
    else ()
        set(USE_SYSTEM_FMT OFF)
    endif ()
endif ()
if (NOT USE_SYSTEM_FMT)
    # We set the FMT_HEADER_ONLY macro, so no need to actually compile the source
    include(${NNRT_3RDPARTY_DIR}/fmt/fmt.cmake)
    nnrt_import_3rdparty_library(3rdparty_fmt
        PUBLIC
        INCLUDE_DIRS ${FMT_INCLUDE_DIRS}
        DEPENDS ext_fmt
        )
    #    target_compile_definitions(3rdparty_fmt INTERFACE FMT_HEADER_ONLY=1)
endif ()
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS NNRT::3rdparty_fmt)

# Threads
nnrt_find_package_3rdparty_library(3rdparty_threads
    REQUIRED
    PACKAGE Threads
    TARGETS Threads::Threads
    )

# Pybind11
if (USE_SYSTEM_PYBIND11)
    find_package(pybind11)
endif ()
if (NOT USE_SYSTEM_PYBIND11 OR NOT TARGET pybind11::module)
    set(USE_SYSTEM_PYBIND11 OFF)
    add_subdirectory(${NNRT_3RDPARTY_DIR}/pybind11)
endif ()
if (TARGET pybind11::module)
    set(PYBIND11_TARGET "pybind11::module")
endif ()
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS "${PYBIND11_TARGET}")


# Open3D
find_package(Open3D REQUIRED)
if (NOT WIN32)
    list(APPEND Open3D_LIBRARIES dl)
    list(APPEND Open3D_LIBRARIES stdc++fs)
endif ()
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS Open3D::Open3D)
get_target_property(OPEN3D_COMPILE_DEFINITIONS Open3D::Open3D INTERFACE_COMPILE_DEFINITIONS)
list(FIND OPEN3D_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=0" Open3D_CXX11_ABI_0_INDEX)
list(FIND OPEN3D_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1" Open3D_CXX11_ABI_1_INDEX)
set(NNRT_EXTRA_DEP_ABI_COMPILE_DEFS)
set(NNRT_DEPS_AVOID_CXX11_ABI OFF)
if (Open3D_CXX11_ABI_0_INDEX GREATER -1)
    set(NNRT_EXTRA_DEP_ABI_COMPILE_DEFS _GLIBCXX_USE_CXX11_ABI=0)
    set(NNRT_DEPS_AVOID_CXX11_ABI ON)
    #elseif(Open3D_CXX11_ABI_0_INDEX GREATER -1)
    #    set(NNRT_EXTRA_DEP_ABI_COMPILE_DEFS _GLIBCXX_USE_CXX11_ABI=1)
endif ()

# Catch2
if (BUILD_CPP_TESTS)
    if (USE_SYSTEM_CATCH2)
        find_package(Catch2)
        if (TARGET Catch2::Catch2)
            message(STATUS "Using installed third-party library Catch2")
            list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "Catch2")
            set(CATCH2_TARGET "Catch2::Catch2")
        else ()
            message(STATUS "Unable to find installed third-party library Catch2")
            set(USE_SYSTEM_CATCH2 OFF)
        endif ()
    endif ()

    if (NOT USE_SYSTEM_CATCH2)
        include(FetchContent)
        if (NNRT_DEPS_AVOID_CXX11_ABI)
            FetchContent_Declare(
                Catch2
                GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                GIT_TAG v3.3.2
                PATCH_COMMAND git apply --ignore-space-change --ignore-whitespace ${NNRT_3RDPARTY_DIR}/Catch2/0001-Build-Catch2-without-cxx11-abi.patch || true
            )
        else ()
            FetchContent_Declare(
                Catch2
                GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                GIT_TAG v3.3.2
            )
        endif ()
        FetchContent_MakeAvailable(Catch2)
        list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
        include(CTest)
        include(Catch)
        set(CATCH2_TARGET "Catch2::Catch2")
    endif ()
endif ()

# Python3
if (DEFINED Python3_VERSION)
    find_package(Python3 ${Python3_VERSION} EXACT COMPONENTS Interpreter Development REQUIRED)
else ()
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif ()


# Flann
if (USE_SYSTEM_FLANN)
    pkg_config_3rdparty_library(3rdparty_flann flann)
endif ()
if (NOT USE_SYSTEM_FLANN OR NOT 3rdparty_flann_FOUND)
    build_3rdparty_library(3rdparty_flann DIRECTORY flann)
endif ()
set(FLANN_TARGET "3rdparty_flann")
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS "${FLANN_TARGET}")

# Stdgpu
if (BUILD_CUDA_MODULE)
    include(${NNRT_3RDPARTY_DIR}/stdgpu/stdgpu.cmake)
    nnrt_import_3rdparty_library(3rdparty_stdgpu
        INCLUDE_DIRS ${STDGPU_INCLUDE_DIRS}
        LIB_DIR ${STDGPU_LIB_DIR}
        LIBRARIES ${STDGPU_LIBRARIES}
        DEPENDS ext_stdgpu
        )
    list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_stdgpu)
endif ()

# PNG
if (USE_SYSTEM_PNG)
    # ZLIB::ZLIB is automatically included by the PNG package.
    nnrt_find_package_3rdparty_library(3rdparty_libpng
        PACKAGE PNG
        TARGETS PNG::PNG
        )
    if (3rdparty_libpng_FOUND)
        if (NOT BUILD_SHARED_LIBS)
            list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "PNG")
        endif ()
    else ()
        set(USE_SYSTEM_PNG OFF)
    endif ()
endif ()
if (NOT USE_SYSTEM_PNG)
    include(${NNRT_3RDPARTY_DIR}/zlib/zlib.cmake)
    nnrt_import_3rdparty_library(3rdparty_zlib
        HIDDEN
        INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS}
        LIB_DIR ${ZLIB_LIB_DIR}
        LIBRARIES ${ZLIB_LIBRARIES}
        DEPENDS ext_zlib
        )

    include(${NNRT_3RDPARTY_DIR}/libpng/libpng.cmake)
    nnrt_import_3rdparty_library(3rdparty_libpng
        INCLUDE_DIRS ${LIBPNG_INCLUDE_DIRS}
        LIB_DIR ${LIBPNG_LIB_DIR}
        LIBRARIES ${LIBPNG_LIBRARIES}
        DEPENDS ext_libpng
        )
    add_dependencies(ext_libpng ext_zlib)
    target_link_libraries(3rdparty_libpng INTERFACE NNRT::3rdparty_zlib)
endif ()
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_libpng)

#zstr
include(${NNRT_3RDPARTY_DIR}/zstr/zstr.cmake)
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS ext_zstr)

if (SET_UP_U_2_NET)
    include(${NNRT_3RDPARTY_DIR}/U-2-Net/u-2-net.cmake)
endif ()

# TBB
include(${NNRT_3RDPARTY_DIR}/tbb/tbb.cmake)
nnrt_import_3rdparty_library(3rdparty_tbb
    INCLUDE_DIRS ${STATIC_TBB_INCLUDE_DIR}
    LIB_DIR ${STATIC_TBB_LIB_DIR}
    LIBRARIES ${STATIC_TBB_LIBRARIES}
    DEPENDS ext_tbb
    )
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_tbb)

# parallelstl
include(${NNRT_3RDPARTY_DIR}/parallelstl/parallelstl.cmake)
nnrt_import_3rdparty_library(3rdparty_parallelstl
    PUBLIC
    INCLUDE_DIRS ${PARALLELSTL_INCLUDE_DIRS}
    INCLUDE_ALL
    DEPENDS      ext_parallelstl
    )
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_parallelstl)

# MKL/BLAS
if (USE_BLAS)
    if (NOT BUILD_BLAS_FROM_SOURCE)
        find_package(BLAS)
        find_package(LAPACK)
        find_package(LAPACKE)
        if (BLAS_FOUND AND LAPACK_FOUND AND LAPACKE_FOUND)
            message(STATUS "System BLAS/LAPACK/LAPACKE found.")
            list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS
                ${BLAS_LIBRARIES}
                ${LAPACK_LIBRARIES}
                ${LAPACKE_LIBRARIES}
                )
        else ()
            message(STATUS "System BLAS/LAPACK/LAPACKE not found, setting BUILD_BLAS_FROM_SOURCE=ON.")
            set(BUILD_BLAS_FROM_SOURCE ON)
        endif ()
    endif ()

    if (BUILD_BLAS_FROM_SOURCE)
        # Install gfortran first for compiling OpenBLAS/Lapack from source.
        message(STATUS "Building OpenBLAS with LAPACK from source")

        find_program(gfortran_bin "gfortran")
        if (gfortran_bin)
            message(STATUS "gfortran found at ${gfortran}")
        else ()
            message(FATAL_ERROR "gfortran is required to compile LAPACK from source. "
                "On Ubuntu, please install by `apt install gfortran`. "
                "On macOS, please install by `brew install gfortran`. ")
        endif ()

        include(${NNRT_3RDPARTY_DIR}/openblas/openblas.cmake)
        nnrt_import_3rdparty_library(3rdparty_blas
            HIDDEN
            INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR}
            LIB_DIR ${OPENBLAS_LIB_DIR}
            LIBRARIES ${OPENBLAS_LIBRARIES}
            DEPENDS ext_openblas
            )
        # Get gfortran library search directories.
        execute_process(COMMAND ${gfortran_bin} -print-search-dirs
            OUTPUT_VARIABLE gfortran_search_dirs
            RESULT_VARIABLE RET
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        if (RET AND NOT RET EQUAL 0)
            message(FATAL_ERROR "Failed to run `${gfortran_bin} -print-search-dirs`")
        endif ()

        # Parse gfortran library search directories into CMake list.
        string(REGEX MATCH "libraries: =(.*)" match_result ${gfortran_search_dirs})
        if (match_result)
            string(REPLACE ":" ";" gfortran_lib_dirs ${CMAKE_MATCH_1})
        else ()
            message(FATAL_ERROR "Failed to parse gfortran_search_dirs: ${gfortran_search_dirs}")
        endif ()

        if (LINUX_AARCH64 OR APPLE_AARCH64)
            # Find libgfortran.a and libgcc.a inside the gfortran library search
            # directories. This ensures that the library matches the compiler.
            # On ARM64 Ubuntu and ARM64 macOS, libgfortran.a is compiled with `-fPIC`.
            find_library(gfortran_lib NAMES libgfortran.a PATHS ${gfortran_lib_dirs} REQUIRED)
            find_library(gcc_lib NAMES libgcc.a PATHS ${gfortran_lib_dirs} REQUIRED)
            target_link_libraries(3rdparty_blas INTERFACE
                ${gfortran_lib}
                ${gcc_lib}
                )
            if (APPLE_AARCH64)
                # Suppress Apple compiler warnigns.
                target_link_options(3rdparty_blas INTERFACE "-Wl,-no_compact_unwind")
            endif ()
        elseif (UNIX AND NOT APPLE)
            # On Ubuntu 20.04 x86-64, libgfortran.a is not compiled with `-fPIC`.
            # The temporary solution is to link the shared library libgfortran.so.
            # If we distribute a Python wheel, the user's system will also need
            # to have libgfortran.so preinstalled.
            #
            # If you have to link libgfortran.a statically
            # - Read https://gcc.gnu.org/wiki/InstallingGCC
            # - Run `gfortran --version`, e.g. you get 9.3.0
            # - Checkout gcc source code to the corresponding version
            # - Configure with
            #   ${PWD}/../gcc/configure --prefix=${HOME}/gcc-9.3.0 \
            #                           --enable-languages=c,c++,fortran \
            #                           --with-pic --disable-multilib
            # - make install -j$(nproc) # This will take a while
            # - Change this cmake file to libgfortran.a statically.
            # - Link
            #   - libgfortran.a
            #   - libgcc.a
            #   - libquadmath.a
            target_link_libraries(3rdparty_blas INTERFACE gfortran)
        endif ()
        list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_blas)
    endif ()
else ()
    include(${NNRT_3RDPARTY_DIR}/mkl/mkl.cmake)
    # MKL, cuSOLVER, cuBLAS
    # We link MKL statically. For MKL link flags, refer to:
    # https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html
    message(STATUS "Using MKL to support BLAS and LAPACK functionalities.")
    nnrt_import_3rdparty_library(3rdparty_blas
        HIDDEN
        INCLUDE_DIRS ${STATIC_MKL_INCLUDE_DIR}
        LIB_DIR ${STATIC_MKL_LIB_DIR}
        LIBRARIES ${STATIC_MKL_LIBRARIES}
        DEPENDS ext_tbb ext_mkl_include ext_mkl
        )
    if (UNIX)
        target_compile_options(3rdparty_blas INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-m64>")
        target_link_libraries(3rdparty_blas INTERFACE NNRT::3rdparty_threads ${CMAKE_DL_LIBS})
    endif ()
    target_compile_definitions(3rdparty_blas INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:MKL_ILP64>")
    target_compile_definitions(3rdparty_blas INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:MKL_ILP64>")
    list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_blas)
endif ()


if (BUILD_CUDA_MODULE)
    # cuBLAS
    if (WIN32)
        # Nvidia does not provide static libraries for Windows. We don't release
        # pip wheels for Windows with CUDA support at the moment. For the pip
        # wheels to support CUDA on Windows out-of-the-box, we need to either
        # ship the CUDA toolkit with the wheel (e.g. PyTorch can make use of the
        # cudatoolkit conda package), or have a mechanism to locate the CUDA
        # toolkit from the system.
        list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS CUDA::cusolver CUDA::cublas)
    else ()
        # CMake docs   : https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
        # cusolver 11.0: https://docs.nvidia.com/cuda/archive/11.0/cusolver/index.html#static-link-lapack
        # cublas   11.0: https://docs.nvidia.com/cuda/archive/11.0/cublas/index.html#static-library
        # The link order below is important. Theoretically we should use
        # nnrt_find_package_3rdparty_library, but we have to insert
        # liblapack_static.a in the middle of the targets.
        add_library(3rdparty_cublas INTERFACE)
        target_link_libraries(3rdparty_cublas INTERFACE
            CUDA::cusolver_static
            ${CUDAToolkit_LIBRARY_DIR}/liblapack_static.a
            CUDA::cusparse_static
            CUDA::cublas_static
            CUDA::cublasLt_static
            CUDA::culibos
            )
        if (NOT BUILD_SHARED_LIBS)
            # Listed in ${CMAKE_INSTALL_PREFIX}/lib/cmake/NNRT/NNRTTargets.cmake.
            install(TARGETS 3rdparty_cublas EXPORT NNRTTargets)
            list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "CUDAToolkit")
        endif ()
        add_library(NNRT::3rdparty_cublas ALIAS 3rdparty_cublas)
        list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_cublas)
    endif ()



    # MAGMA
    if (NOT USE_SYSTEM_MAGMA)
        include(${NNRT_3RDPARTY_DIR}/magma/magma.cmake)
        nnrt_import_3rdparty_library(3rdparty_magma
            INCLUDE_DIRS ${MAGMA_INCLUDE_DIRS}
            LIB_DIR ${MAGMA_LIB_DIR}
            LIBRARIES ${MAGMA_LIBRARIES}
            DEPENDS ext_magma
            )
        list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_magma)
    else()
        # TODO check USE_SYSTEM_MAGMA case
        pkg_check_modules(MAGMA magma>=2.7.1)
    endif ()
endif ()





# Backward
list(APPEND CMAKE_PREFIX_PATH "${NNRT_3RDPARTY_DIR}/backward-cpp")
find_package(Backward REQUIRED)
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS Backward::Backward)
