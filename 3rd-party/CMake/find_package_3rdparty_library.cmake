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
    if (arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: nnrt_find_package_3rdparty_library(${name} ${ARGN})")
    endif ()
    if (NOT arg_PACKAGE)
        message(FATAL_ERROR "nnrt_find_package_3rdparty_library: Expected value for argument PACKAGE")
    endif ()
    set(find_package_args "")
    if (arg_REQUIRED)
        list(APPEND find_package_args "REQUIRED")
    endif ()
    if (arg_QUIET)
        list(APPEND find_package_args "QUIET")
    endif ()
    find_package(${arg_PACKAGE} ${find_package_args})
    if (${arg_PACKAGE}_FOUND)
        message(STATUS "Using installed third-party library ${name} ${${arg_PACKAGE}_VERSION}")
        add_library(${name} INTERFACE)
        if (arg_TARGETS)
            foreach (target IN LISTS arg_TARGETS)
                if (TARGET ${target})
                    target_link_libraries(${name} INTERFACE ${target})
                else ()
                    message(WARNING "Skipping undefined target ${target}")
                endif ()
            endforeach ()
        endif ()
        if (arg_INCLUDE_DIRS)
            foreach (incl IN LISTS arg_INCLUDE_DIRS)
                target_include_directories(${name} INTERFACE ${${incl}})
            endforeach ()
        endif ()
        if (arg_LIBRARIES)
            foreach (lib IN LISTS arg_LIBRARIES)
                target_link_libraries(${name} INTERFACE ${${lib}})
            endforeach ()
        endif ()
        if (NOT BUILD_SHARED_LIBS)
            install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
        endif ()
        set(${name}_FOUND TRUE PARENT_SCOPE)
        add_library(${PROJECT_NAME}::${name} ALIAS ${name})
    else ()
        message(STATUS "Unable to find installed third-party library ${name}")
        set(${name}_FOUND FALSE PARENT_SCOPE)
    endif ()
endfunction()