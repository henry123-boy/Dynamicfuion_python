set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/3rd-party/CMake)

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

# nnrt_find_package_3rdparty_library(name ...)
#
# Creates an interface library for a find_package dependency.
#
# The function will set ${name}_FOUND to TRUE or FALSE
# indicating whether or not the library could be found.
#
# Valid options:
#    REQUIRED
#        finding the package is required
#    QUIET
#        finding the package is quiet
#    PACKAGE <pkg>
#        the name of the queried package <pkg> forwarded to find_package()
#    TARGETS <target> [<target> ...]
#        the expected targets to be found in <pkg>
#    INCLUDE_DIRS
#        the expected include directory variable names to be found in <pkg>.
#        If <pkg> also defines targets, use them instead and pass them via TARGETS option.
#    LIBRARIES
#        the expected library variable names to be found in <pkg>.
#        If <pkg> also defines targets, use them instead and pass them via TARGETS option.
#
function(nnrt_find_package_3rdparty_library name)
    cmake_parse_arguments(arg "REQUIRED;QUIET" "PACKAGE" "TARGETS;INCLUDE_DIRS;LIBRARIES" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: nnrt_find_package_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_PACKAGE)
        message(FATAL_ERROR "nnrt_find_package_3rdparty_library: Expected value for argument PACKAGE")
    endif()
    set(find_package_args "")
    if(arg_REQUIRED)
        list(APPEND find_package_args "REQUIRED")
    endif()
    if(arg_QUIET)
        list(APPEND find_package_args "QUIET")
    endif()
    find_package(${arg_PACKAGE} ${find_package_args})
    if(${arg_PACKAGE}_FOUND)
        message(STATUS "Using installed third-party library ${name} ${${arg_PACKAGE}_VERSION}")
        add_library(${name} INTERFACE)
        if(arg_TARGETS)
            foreach(target IN LISTS arg_TARGETS)
                if (TARGET ${target})
                    target_link_libraries(${name} INTERFACE ${target})
                else()
                    message(WARNING "Skipping undefined target ${target}")
                endif()
            endforeach()
        endif()
        if(arg_INCLUDE_DIRS)
            foreach(incl IN LISTS arg_INCLUDE_DIRS)
                target_include_directories(${name} INTERFACE ${${incl}})
            endforeach()
        endif()
        if(arg_LIBRARIES)
            foreach(lib IN LISTS arg_LIBRARIES)
                target_link_libraries(${name} INTERFACE ${${lib}})
            endforeach()
        endif()
        if(NOT BUILD_SHARED_LIBS)
            install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
        endif()
        set(${name}_FOUND TRUE PARENT_SCOPE)
        add_library(${PROJECT_NAME}::${name} ALIAS ${name})
    else()
        message(STATUS "Unable to find installed third-party library ${name}")
        set(${name}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# List of linker options for libOpen3D client binaries (eg: pybind) to hide Open3D 3rd
# party dependencies. Only needed with GCC, not AppleClang.
set(NNRT_HIDDEN_3RDPARTY_LINK_OPTIONS)

if (CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
    find_library(LexLIB libl.a)    # test archive in macOS
    if (LexLIB)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_LINK_OPTIONS -load_hidden ${LexLIB})
        check_cxx_source_compiles("int main() {return 0;}" FLAG_load_hidden)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
    endif()
endif()
if (NOT FLAG_load_hidden)
    set(FLAG_load_hidden 0)
endif()

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

# nnrt_import_3rdparty_library(name ...)
#
# Imports a third-party library that has been built independently in a sub project.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface and will be
#        installed, but the library is linked privately.
#    INCLUDE_ALL
#        install all files in the include directories. Default is *.h, *.hpp
#    HIDDEN
#         Symbols from this library will not be exported to client code during
#         linking with Open3D. This is the opposite of the VISIBLE option in
#         nnrt_build_3rdparty_library.  Prefer hiding symbols during building 3rd
#         party libraries, since this option is not supported by the MSVC linker.
#    INCLUDE_DIRS
#        the temporary location where the library headers have been installed.
#        Trailing slashes have the same meaning as with install(DIRECTORY).
#        If your include is "#include <x.hpp>" and the path of the file is
#        "/path/to/libx/x.hpp" then you need to pass "/path/to/libx/"
#        with the trailing "/". If you have "#include <libx/x.hpp>" then you
#        need to pass "/path/to/libx".
#    LIBRARIES
#        the built library name(s). It is assumed that the library is static.
#        If the library is PUBLIC, it will be renamed to Open3D_${name} at
#        install time to prevent name collisions in the install space.
#    LIB_DIR
#        the temporary location of the library. Defaults to
#        CMAKE_ARCHIVE_OUTPUT_DIRECTORY.
#    DEPENDS <target> [<target> ...]
#        targets on which <name> depends on and that must be built before.
#
function(nnrt_import_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;INCLUDE_ALL;HIDDEN" "LIB_DIR" "INCLUDE_DIRS;LIBRARIES;DEPENDS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: nnrt_import_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_LIB_DIR)
        set(arg_LIB_DIR "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    endif()
    add_library(${name} INTERFACE)
    if(arg_INCLUDE_DIRS)
        foreach(incl IN LISTS arg_INCLUDE_DIRS)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM INTERFACE $<BUILD_INTERFACE:${incl_path}>)
            if(arg_PUBLIC OR arg_HEADER)
                if(arg_INCLUDE_ALL)
                    install(DIRECTORY ${incl}
                        DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/nnrt/3rdparty
                        )
                else()
                    install(DIRECTORY ${incl}
                        DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/nnrt/3rdparty
                        FILES_MATCHING
                        PATTERN "*.h"
                        PATTERN "*.hpp"
                        )
                endif()
                target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${NNRT_INSTALL_INCLUDE_DIR}/nnrt/3rdparty>)
            endif()
        endforeach()
    endif()
    if(arg_LIBRARIES)
        list(LENGTH arg_LIBRARIES libcount)
        if(arg_HIDDEN AND NOT arg_PUBLIC AND NOT arg_HEADER)
            set(HIDDEN 1)
        else()
            set(HIDDEN 0)
        endif()
        foreach(arg_LIBRARY IN LISTS arg_LIBRARIES)
            set(library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            if(libcount EQUAL 1)
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}${CMAKE_STATIC_LIBRARY_SUFFIX})
            else()
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}_${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            endif()
            # Apple compiler ld
            target_link_libraries(${name} INTERFACE
                "$<BUILD_INTERFACE:$<$<AND:${HIDDEN},${FLAG_load_hidden}>:-load_hidden >${arg_LIB_DIR}/${library_filename}>")
            if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
                install(FILES ${arg_LIB_DIR}/${library_filename}
                    DESTINATION ${NNRT_INSTALL_LIB_DIR}
                    RENAME ${installed_library_filename}
                )
                target_link_libraries(${name} INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${NNRT_INSTALL_LIB_DIR}/${installed_library_filename}>)
            endif()
            if (HIDDEN)
                # GNU compiler ld
                target_link_options(${name} INTERFACE
                    $<$<CXX_COMPILER_ID:GNU>:LINKER:--exclude-libs,${library_filename}>)
                list(APPEND NNRT_HIDDEN_3RDPARTY_LINK_OPTIONS $<$<CXX_COMPILER_ID:GNU>:LINKER:--exclude-libs,${library_filename}>)
                set(NNRT_HIDDEN_3RDPARTY_LINK_OPTIONS
                    ${NNRT_HIDDEN_3RDPARTY_LINK_OPTIONS} PARENT_SCOPE)
            endif()
        endforeach()
    endif()
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
    endif()
    if(arg_DEPENDS)
        add_dependencies(${name} ${arg_DEPENDS})
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

#
# build_3rdparty_library(name ...)
#
# Builds a third-party library from source
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface, but the library
#        itself is linked privately
#    INCLUDE_ALL
#        install all files in the include directories. Default is *.h, *.hpp
#    DIRECTORY <dir>
#        the library sources are in the subdirectory <dir> of 3rdparty/
#    INCLUDE_DIRS <dir> [<dir> ...]
#        include headers are in the subdirectories <dir>. Trailing slashes
#        have the same meaning as with install(DIRECTORY). <dir> must be
#        relative to the library source directory.
#        If your include is "#include <x.hpp>" and the path of the file is
#        "path/to/libx/x.hpp" then you need to pass "path/to/libx/"
#        with the trailing "/". If you have "#include <libx/x.hpp>" then you
#        need to pass "path/to/libx".
#    SOURCES <src> [<src> ...]
#        the library sources. Can be omitted for header-only libraries.
#        All sources must be relative to the library source directory.
#    LIBS <target> [<target> ...]
#        extra link dependencies
#
function(build_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;INCLUDE_ALL" "DIRECTORY" "INCLUDE_DIRS;SOURCES;LIBS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Invalid syntax: build_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_DIRECTORY)
        set(arg_DIRECTORY "${name}")
    endif()
    if(arg_INCLUDE_DIRS)
        set(include_dirs)
        foreach(incl IN LISTS arg_INCLUDE_DIRS)
            list(APPEND include_dirs "${NNRT_3RDPARTY_DIR}/${arg_DIRECTORY}/${incl}")
        endforeach()
    else()
        set(include_dirs "${NNRT_3RDPARTY_DIR}/${arg_DIRECTORY}/")
    endif()
    message(STATUS "Building library ${name} from source")
    if(arg_SOURCES)
        set(sources)
        foreach(src ${arg_SOURCES})
            list(APPEND sources "${NNRT_3RDPARTY_DIR}/${arg_DIRECTORY}/${src}")
        endforeach()
        add_library(${name} STATIC ${sources})
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM PUBLIC
                $<BUILD_INTERFACE:${incl_path}>
                )
        endforeach()
        target_include_directories(${name} PUBLIC
            $<INSTALL_INTERFACE:${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty>
            )
        nnrt_set_global_properties(${name})
        set_target_properties(${name} PROPERTIES
            OUTPUT_NAME "${PROJECT_NAME}_${name}"
            )
        if(arg_LIBS)
            target_link_libraries(${name} PRIVATE ${arg_LIBS})
        endif()
    else()
        add_library(${name} INTERFACE)
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM INTERFACE
                $<BUILD_INTERFACE:${incl_path}>
                )
        endforeach()
        target_include_directories(${name} INTERFACE
            $<INSTALL_INTERFACE:${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty>
            )
    endif()
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets
            RUNTIME DESTINATION ${NNRT_INSTALL_BIN_DIR}
            ARCHIVE DESTINATION ${NNRT_INSTALL_LIB_DIR}
            LIBRARY DESTINATION ${NNRT_INSTALL_LIB_DIR}
            )
    endif()
    if(arg_PUBLIC OR arg_HEADER)
        foreach(incl IN LISTS include_dirs)
            if(arg_INCLUDE_ALL)
                install(DIRECTORY ${incl}
                    DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    )
            else()
                install(DIRECTORY ${incl}
                    DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    FILES_MATCHING
                    PATTERN "*.h"
                    PATTERN "*.hpp"
                    )
            endif()
        endforeach()
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()
# Convenience function to link against all third-party libraries
# We need this because we create a lot of object libraries to assemble
# the main library
function(nnrt_link_3rdparty_libraries target)
    target_link_libraries(${target} PRIVATE ${NNRT_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${NNRT_3RDPARTY_PUBLIC_TARGETS})
    foreach(dep IN LISTS NNRT_3RDPARTY_HEADER_TARGETS)
        if(TARGET ${dep})
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(def TARGET ${dep} PROPERTY INTERFACE_COMPILE_DEFINITIONS)
            if(def)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${def})
            endif()
        endif()
    endforeach()
endfunction()



# Python
find_package(PythonExecutable REQUIRED) # invokes the module in 3rd-party/CMake

# Threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE) # -pthread instead of -lpthread
find_package(Threads REQUIRED)
list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "Threads")

# OpenMP
if(WITH_OPENMP)
    find_package(OpenMP)
    if(TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Building with OpenMP")
        set(OPENMP_TARGET "OpenMP::OpenMP_CXX")
        list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS "${OPENMP_TARGET}")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "OpenMP")
        endif()
    endif()
endif()


# GLEW
if(USE_SYSTEM_GLEW)
    find_package(GLEW)
    if(TARGET GLEW::GLEW)
        message(STATUS "Using installed third-party library GLEW ${GLEW_VERSION}")
        list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "GLEW")
        set(GLEW_TARGET "GLEW::GLEW")
    else()
        pkg_config_3rdparty_library(3rdparty_glew glew)
        if(3rdparty_glew_FOUND)
            set(GLEW_TARGET "3rdparty_glew")
        else()
            set(USE_SYSTEM_GLEW OFF)
        endif()
    endif()
endif()
if(NOT USE_SYSTEM_GLEW)
    build_3rdparty_library(3rdparty_glew HEADER DIRECTORY glew SOURCES src/glew.c INCLUDE_DIRS include/)
    if(ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(3rdparty_glew PUBLIC GLEW_OSMESA)
    endif()
    if(WIN32)
        target_compile_definitions(3rdparty_glew PUBLIC GLEW_STATIC)
    endif()
    set(GLEW_TARGET "3rdparty_glew")
endif()
list(APPEND NNRT_3RDPARTY_HEADER_TARGETS "${GLEW_TARGET}")
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS "${GLEW_TARGET}")

# OpenGL
find_package(OpenGL REQUIRED)
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS OpenGL::GL OpenGL::GLU)

## Freeglut
find_package(GLUT REQUIRED)
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS GLUT::GLUT)


# Eigen3
if(USE_SYSTEM_EIGEN3)
    find_package(Eigen3)
    if(TARGET Eigen3::Eigen)
        message(STATUS "Using installed third-party library Eigen3 ${EIGEN3_VERSION_STRING}")
        # Eigen3 is a publicly visible dependency, so add it to the list of
        # modules we need to find in the NNRT config script.
        list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "Eigen3")
        set(EIGEN3_TARGET "Eigen3::Eigen")
    else()
        message(STATUS "Unable to find installed third-party library Eigen3")
        set(USE_SYSTEM_EIGEN3 OFF)
    endif()
endif()
if(NOT USE_SYSTEM_EIGEN3)
    build_3rdparty_library(3rdparty_eigen3 PUBLIC DIRECTORY Eigen INCLUDE_DIRS Eigen INCLUDE_ALL)
    set(EIGEN3_TARGET "3rdparty_eigen3")
endif()
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS "${EIGEN3_TARGET}")

# Pybind11
if(USE_SYSTEM_PYBIND11)
    find_package(pybind11)
endif()
if (NOT USE_SYSTEM_PYBIND11 OR NOT TARGET pybind11::module)
    set(USE_SYSTEM_PYBIND11 OFF)
    add_subdirectory(${NNRT_3RDPARTY_DIR}/pybind11)
endif()
if(TARGET pybind11::module)
    set(PYBIND11_TARGET "pybind11::module")
endif()
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS "${PYBIND11_TARGET}")

# TBB
include(${NNRT_3RDPARTY_DIR}/tbb/tbb.cmake)
nnrt_import_3rdparty_library(3rdparty_tbb
    INCLUDE_DIRS ${STATIC_TBB_INCLUDE_DIR}
    LIB_DIR      ${STATIC_TBB_LIB_DIR}
    LIBRARIES    ${STATIC_TBB_LIBRARIES}
    DEPENDS      ext_tbb
)
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_tbb)

# Pytorch

#find_package(Pytorch REQUIRED)
#list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS torch)

# Threads
find_package(Threads REQUIRED)

# Catch2
if(USE_SYSTEM_CATCH2)
    find_package(Catch2)
    if(TARGET Catch2::Catch2)
        message(STATUS "Using installed third-party library Catch2")
        list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "Catch2")
        set(CATCH2_TARGET "Catch2::Catch2")
        list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS Catch2::Catch2)
    else()
        message(STATUS "Unable to find installed third-party library Catch2")
        set(USE_SYSTEM_CATCH2 OFF)
    endif()
endif()
if(NOT USE_SYSTEM_CATCH2)
    include(${NNRT_3RDPARTY_DIR}/Catch2/Catch2.cmake)
    nnrt_import_3rdparty_library(3rdparty_Catch2
        INCLUDE_DIRS ${CATCH2_INCLUDE_DIRS}
        LIB_DIR      ${CATCH2_LIB_DIR}
        LIBRARIES    ${CATCH2_LIBRARIES}
        DEPENDS ext_Catch2
    )
    list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS NNRT::3rdparty_Catch2)
endif()


# Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


# Flann
if(USE_SYSTEM_FLANN)
    pkg_config_3rdparty_library(3rdparty_flann flann)
endif()
if(NOT USE_SYSTEM_FLANN OR NOT 3rdparty_flann_FOUND)
    build_3rdparty_library(3rdparty_flann DIRECTORY flann)
endif()
set(FLANN_TARGET "3rdparty_flann")
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS "${FLANN_TARGET}")

# Open3D
find_package(Open3D REQUIRED)
if(NOT WIN32)
    list(APPEND Open3D_LIBRARIES dl)
endif()
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS ${Open3D_LIBRARIES})

# Stdgpu
if (BUILD_CUDA_MODULE)
    include(${NNRT_3RDPARTY_DIR}/stdgpu/stdgpu.cmake)
    nnrt_import_3rdparty_library(3rdparty_stdgpu
        INCLUDE_DIRS ${STDGPU_INCLUDE_DIRS}
        LIB_DIR      ${STDGPU_LIB_DIR}
        LIBRARIES    ${STDGPU_LIBRARIES}
        DEPENDS      ext_stdgpu
    )
    list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_stdgpu)
endif ()

# PNG
if(USE_SYSTEM_PNG)
    # ZLIB::ZLIB is automatically included by the PNG package.
    nnrt_find_package_3rdparty_library(3rdparty_libpng
        PACKAGE PNG
        TARGETS PNG::PNG
        )
    if(3rdparty_libpng_FOUND)
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "PNG")
        endif()
    else()
        set(USE_SYSTEM_PNG OFF)
    endif()
endif()
if(NOT USE_SYSTEM_PNG)
    include(${NNRT_3RDPARTY_DIR}/zlib/zlib.cmake)
    nnrt_import_3rdparty_library(3rdparty_zlib
        HIDDEN
        INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS}
        LIB_DIR      ${ZLIB_LIB_DIR}
        LIBRARIES    ${ZLIB_LIBRARIES}
        DEPENDS      ext_zlib
    )

    include(${NNRT_3RDPARTY_DIR}/libpng/libpng.cmake)
    nnrt_import_3rdparty_library(3rdparty_libpng
        INCLUDE_DIRS ${LIBPNG_INCLUDE_DIRS}
        LIB_DIR      ${LIBPNG_LIB_DIR}
        LIBRARIES    ${LIBPNG_LIBRARIES}
        DEPENDS      ext_libpng
    )
    add_dependencies(ext_libpng ext_zlib)
    target_link_libraries(3rdparty_libpng INTERFACE NNRT::3rdparty_zlib)
endif()
list(APPEND NNRT_3RDPARTY_PRIVATE_TARGETS NNRT::3rdparty_libpng)

if(SET_UP_U_2_NET)
    include(${NNRT_3RDPARTY_DIR}/U-2-Net/u-2-net.cmake)
endif()


# Backward
list(APPEND CMAKE_PREFIX_PATH "${NNRT_3RDPARTY_DIR}/backward-cpp")
find_package(Backward REQUIRED)
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS Backward::Backward)
