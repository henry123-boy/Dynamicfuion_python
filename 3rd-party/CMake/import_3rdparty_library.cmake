#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 7/17/23.
#  Copyright (c) 2023 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

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
#         linking with NNRT. This is the opposite of the VISIBLE option in
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
#        If the library is PUBLIC, it will be renamed to NNRT_${name} at
#        install time to prevent name collisions in the install space.
#    LIB_DIR
#        the temporary location of the library. Defaults to
#        CMAKE_ARCHIVE_OUTPUT_DIRECTORY.
#    DEPENDS <target> [<target> ...]
#        targets on which <name> depends on and that must be built before.
#
function(nnrt_import_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;INCLUDE_ALL;HIDDEN" "LIB_DIR" "INCLUDE_DIRS;LIBRARIES;DEPENDS" ${ARGN})
    if (arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: nnrt_import_3rdparty_library(${name} ${ARGN})")
    endif ()
    if (NOT arg_LIB_DIR)
        set(arg_LIB_DIR "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    endif ()
    add_library(${name} INTERFACE)
    if (arg_INCLUDE_DIRS)
        foreach (incl IN LISTS arg_INCLUDE_DIRS)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else ()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif ()
            target_include_directories(${name} SYSTEM INTERFACE $<BUILD_INTERFACE:${incl_path}>)
            if (arg_PUBLIC OR arg_HEADER)
                if (arg_INCLUDE_ALL)
                    install(DIRECTORY ${incl}
                        DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/nnrt/3rdparty
                        )
                else ()
                    install(DIRECTORY ${incl}
                        DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/nnrt/3rdparty
                        FILES_MATCHING
                        PATTERN "*.h"
                        PATTERN "*.hpp"
                        )
                endif ()
                target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${NNRT_INSTALL_INCLUDE_DIR}/nnrt/3rdparty>)
            endif ()
        endforeach ()
    endif ()
    if (arg_LIBRARIES)
        list(LENGTH arg_LIBRARIES libcount)
        if (arg_HIDDEN AND NOT arg_PUBLIC AND NOT arg_HEADER)
            set(HIDDEN 1)
        else ()
            set(HIDDEN 0)
        endif ()
        foreach (arg_LIBRARY IN LISTS arg_LIBRARIES)
            set(library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            if (libcount EQUAL 1)
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}${CMAKE_STATIC_LIBRARY_SUFFIX})
            else ()
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}_${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            endif ()
            # Apple compiler ld
            target_link_libraries(${name} INTERFACE
                "$<BUILD_INTERFACE:$<$<AND:${HIDDEN},${FLAG_load_hidden}>:-load_hidden >${arg_LIB_DIR}/${library_filename}>")
            if (NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
                install(FILES ${arg_LIB_DIR}/${library_filename}
                    DESTINATION ${NNRT_INSTALL_LIB_DIR}
                    RENAME ${installed_library_filename}
                    )
                target_link_libraries(${name} INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${NNRT_INSTALL_LIB_DIR}/${installed_library_filename}>)
            endif ()
            if (HIDDEN)
                # GNU compiler ld
                target_link_options(${name} INTERFACE
                    $<$<CXX_COMPILER_ID:GNU>:LINKER:--exclude-libs,${library_filename}>)
                list(APPEND NNRT_HIDDEN_3RDPARTY_LINK_OPTIONS $<$<CXX_COMPILER_ID:GNU>:LINKER:--exclude-libs,${library_filename}>)
                set(NNRT_HIDDEN_3RDPARTY_LINK_OPTIONS
                    ${NNRT_HIDDEN_3RDPARTY_LINK_OPTIONS} PARENT_SCOPE)
            endif ()
        endforeach ()
    endif ()
    if (NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
    endif ()
    if (arg_DEPENDS)
        add_dependencies(${name} ${arg_DEPENDS})
    endif ()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()