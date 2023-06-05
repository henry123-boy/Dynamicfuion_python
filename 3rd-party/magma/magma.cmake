#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 6/5/23.
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
include(ExternalProject)
include(ProcessorCount)

if (MSVC)
    # TODO: not sure if MSVC actually still needs the lib prefix manually added with contemporary CMake -- test & remove if unnecessary
    set(lib_name libmagma)
    set(lib_sparse_name libmagma_sparse)
else ()
    set(lib_name magma)
    set(lib_sparse_name magma_sparse)
endif ()

include(NNRTMakeCudaArchitectures)
nnrt_make_cuda_architectures(CUDA_ARCHS)

externalproject_add(
    ext_magma
    PREFIX magma
    URL https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.7.1.tar.gz
    URL_HASH SHA256=d9c8711c047a38cae16efde74bee2eb3333217fd2711e1e9b8606cbbb4ae1a50
    DOWNLOAD_DIR "${NNRT_THIRD_PARTY_DOWNLOAD_DIR}/magma"
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}
    UPDATE_COMMAND ""
)
ExternalProject_Get_Property(ext_magma INSTALL_DIR)

set(MAGMA_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(MAGMA_LIB_DIR ${INSTALL_DIR}/${NNRT_INSTALL_LIB_DIR})
set(MAGMA_LIBRARIES ${lib_name} ${lib_sparse_name})