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
    if (arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Invalid syntax: build_3rdparty_library(${name} ${ARGN})")
    endif ()
    if (NOT arg_DIRECTORY)
        set(arg_DIRECTORY "${name}")
    endif ()
    if (arg_INCLUDE_DIRS)
        set(include_dirs)
        foreach (incl IN LISTS arg_INCLUDE_DIRS)
            list(APPEND include_dirs "${NNRT_3RDPARTY_DIR}/${arg_DIRECTORY}/${incl}")
        endforeach ()
    else ()
        set(include_dirs "${NNRT_3RDPARTY_DIR}/${arg_DIRECTORY}/")
    endif ()
    message(STATUS "Building library ${name} from source")
    if (arg_SOURCES)
        set(sources)
        foreach (src ${arg_SOURCES})
            list(APPEND sources "${NNRT_3RDPARTY_DIR}/${arg_DIRECTORY}/${src}")
        endforeach ()
        add_library(${name} STATIC ${sources})
        foreach (incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else ()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif ()
            target_include_directories(${name} SYSTEM PUBLIC
                $<BUILD_INTERFACE:${incl_path}>
                )
        endforeach ()
        target_include_directories(${name} PUBLIC
            $<INSTALL_INTERFACE:${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty>
            )
        nnrt_set_global_properties(${name})
        set_target_properties(${name} PROPERTIES
            OUTPUT_NAME "${PROJECT_NAME}_${name}"
            )
        if (arg_LIBS)
            target_link_libraries(${name} PRIVATE ${arg_LIBS})
        endif ()
    else ()
        add_library(${name} INTERFACE)
        foreach (incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else ()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif ()
            target_include_directories(${name} SYSTEM INTERFACE
                $<BUILD_INTERFACE:${incl_path}>
                )
        endforeach ()
        target_include_directories(${name} INTERFACE
            $<INSTALL_INTERFACE:${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty>
            )
    endif ()
    if (NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets
            RUNTIME DESTINATION ${NNRT_INSTALL_BIN_DIR}
            ARCHIVE DESTINATION ${NNRT_INSTALL_LIB_DIR}
            LIBRARY DESTINATION ${NNRT_INSTALL_LIB_DIR}
            )
    endif ()
    if (arg_PUBLIC OR arg_HEADER)
        foreach (incl IN LISTS include_dirs)
            if (arg_INCLUDE_ALL)
                install(DIRECTORY ${incl}
                    DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    )
            else ()
                install(DIRECTORY ${incl}
                    DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    FILES_MATCHING
                    PATTERN "*.h"
                    PATTERN "*.hpp"
                    )
            endif ()
        endforeach ()
    endif ()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()