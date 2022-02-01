### ==================  Python configuration ========================
# Required for building the python package and wheel for python pip

include(get_python_version)
include(get_python_include_dir)
include(get_python_local_site_packages_dir)

get_python_version(${PYTHON_EXECUTABLE} PYTHON_VERSION)
get_python_include_dir(${PYTHON_EXECUTABLE} PYTHON_INCLUDE_DIR)
get_python_local_site_packages_dir(${PYTHON_EXECUTABLE} PYTHON_LOCAL_PACKAGES_DIR)

if (NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "PYTHON_EXECUTABLE should be set in top level CMakeLists.txt")
endif()

# We need to get python version to configure some meta files
message(STATUS "Using Python version: ${PYTHON_VERSION}")

set(PIP_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/pip_package")
set(PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python_package")
set(STUBS_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/stubs")


### ================== Output directory prep ==================

file(MAKE_DIRECTORY ${PIP_PACKAGE_DST_DIR})
# file(MAKE_DIRECTORY ${STUBS_PACKAGE_DST_DIR})


# ==================== set up NNRT C++ library ============================

set(NNRT_SOURCE_FILES
    cpu/graph_proc.h
    cpu/image_proc.h
    cpu/graph_proc.cpp
    cpu/image_proc.cpp
    debugging/debugging.hpp
    core/PlatformIndependence.h

    geometry/kernel/Defines.h
    geometry/WarpableTSDFVoxelGrid.cpp
    geometry/WarpableTSDFVoxelGrid.h
    geometry/kernel/WarpableTSDFVoxelGrid.cpp
    geometry/kernel/WarpableTSDFVoxelGrid.h
    geometry/kernel/WarpableTSDFVoxelGridCPU.cpp
    geometry/kernel/WarpableTSDFVoxelGridCUDA.cu
    geometry/kernel/WarpableTSDFVoxelGridImpl.h

    geometry/GraphWarpField.h
    geometry/GraphWarpField.cpp

    geometry/kernel/Graph.h
    geometry/kernel/Graph.cpp
    geometry/kernel/GraphCPU.cpp
    geometry/kernel/GraphCUDA.cu
    geometry/kernel/GraphImpl.h

    geometry/kernel/WarpUtilities.h
    geometry/kernel/Warp.h
    geometry/kernel/Warp.cpp
    geometry/kernel/WarpCPU.cpp
    geometry/kernel/WarpCUDA.cu
    geometry/kernel/WarpImpl.h
    geometry/kernel/WarpableTSDFVoxelGrid_Analytics.h
    geometry/kernel/WarpableTSDFVoxelGrid_Analytics.cpp
    geometry/kernel/WarpableTSDFVoxelGrid_AnalyticsImpl.h
    geometry/kernel/Comparison.h
    geometry/kernel/Comparison.cpp
    geometry/AnchorComputationMethod.h
    geometry/TransformationMode.h
    geometry/Comparison.h
    geometry/Comparison.cpp
    geometry/kernel/ComparisonCPU.cpp
    geometry/kernel/ComparisonCUDA.cu
    geometry/kernel/ComparisonImpl.h
    geometry/IndexedPointCloud.h

    string/split_string.h
    string/split_string.cpp
    string/join_string_separator.cpp
    string/join_string_separator.h

    core/kernel/KnnUtilities.h
    core/kernel/KnnUtilities_PriorityQueue.h

    core/KeyValuePair.h
    core/DeviceHeap.h
    core/CPU/DeviceHeapCPU.cpp
    core/CUDA/DeviceHeapCUDA.cu
    core/CPU/DeviceHeapCPU.h
    core/CUDA/DeviceHeapCUDA.cuh
    core/Heap.h
    core/Heap.cpp
    core/CPU/HostHeapCPU.h
    core/CPU/HostHeapCPUImpl.h
    core/CPU/HostHeapCPU.cpp
    core/CUDA/HostHeapCUDA.h
    core/CUDA/HostHeapCUDAImpl.cuh
    core/CUDA/HostHeapCUDA.cu
    core/CUDA/HostHeapCUDA.h

    core/KdTree.h
    core/KdTree.cpp
    core/kernel/KdTree.h
    core/kernel/KdTree.cpp
    core/kernel/BuildKdTreeImpl.h
    core/kernel/KdTreeCPU.cpp
    core/kernel/KdTreeCUDA.cu

    core/kernel/KdTreeUtils.cpp
    core/kernel/KdTreeNodeTypes.h

    core/DimensionCount.h

    core/LinearIndex.cpp
    core/LinearIndex.h
    core/kernel/LinearIndex.cpp
    core/kernel/LinearIndex.h
    core/kernel/LinearIndexImpl.h
    core/kernel/LinearIndexCPU.cpp
    core/kernel/LinearIndexCUDA.cu

    core/DeviceSelection.h

    core/TensorManipulationRoutines.h
    core/TensorManipulationRoutines.cpp


)

add_library(nnrt_cpp STATIC
    ${NNRT_SOURCE_FILES}
)

target_include_directories(nnrt_cpp PUBLIC
    ${PYTHON_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

nnrt_link_3rdparty_libraries(nnrt_cpp)
nnrt_set_global_properties(nnrt_cpp)
target_link_libraries(nnrt_cpp PUBLIC CUDA::toolkit)

# ==================== Benchmark Executables =========================================
add_subdirectory(bench)
# ==================== set up NNRT Python binding library ============================
set(PYBIND_SOURCE_FILES
    3rd_party/magic_enum.hpp
    pybind/nnrt_pybind.h
    pybind/nnrt_pybind.cpp
    pybind/geometry/geometry.cpp
    pybind/geometry/geometry.h
    pybind/enum_export.h
)

pybind11_add_module(nnrt
    ${PYBIND_SOURCE_FILES}
)

target_include_directories(nnrt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


target_link_libraries(nnrt PUBLIC nnrt_cpp)

include(ProcessorCount)
set(N_PROCESSORS 1)
ProcessorCount(N)
if(NOT N LESS 2)
    math(EXPR N_PROCESSORS "${N} - 1")
endif()

set(Open3D_DIRECTIVE)
if(Open3D_DIR)
    set(Open3D_DIRECTIVE "Open3D_DIR=${Open3D_DIR}")
endif()

set(Python3_VERSION_DIRECTIVE)
if(DEFINED Python3_VERSION)
    set(Python3_VERSION_DIRECTIVE "Python3_VERSION=${Python3_VERSION}")
endif()

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
    COMMAND ${CMAKE_COMMAND} -E env ${Open3D_DIRECTIVE} ${Python3_VERSION_DIRECTIVE}
    CMAKE_BUILD_PARALLEL_LEVEL=${N_PROCESSORS} CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup_pybind_cmake.py bdist_wheel --bdist-dir ${PYTHON_PACKAGE_DIR} --dist-dir ${PIP_PACKAGE_DST_DIR} --keep-temp
    COMMAND echo "pip wheel created at ${PIP_PACKAGE_DST_DIR}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Use `make install-pip-package` to install pip wheel package to the current
# python environment.
add_custom_target(install-pip-package
    COMMAND ${CMAKE_COMMAND}
    -DPIP_PACKAGE_DST_DIR=${PIP_PACKAGE_DST_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
    DEPENDS pip-package
)


if(NOT DEFINED STUBGEN_EXECUTABLE)
    set(STUBGEN_EXECUTABLE stubgen)
endif()

add_custom_target(install-python-stubs
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${STUBS_PACKAGE_DST_DIR}/nnrt
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt -o ${STUBS_PACKAGE_DST_DIR}/nnrt WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "import nnrt.geometry" >> ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi
    COMMAND ${CMAKE_COMMAND} -E cat ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi ${STUBS_PACKAGE_DST_DIR}/nnrt/nnrt.pyi >> ${STUBS_PACKAGE_DST_DIR}/nnrt/__init__.pyi
    COMMAND ${CMAKE_COMMAND} -E rm -f ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi ${STUBS_PACKAGE_DST_DIR}/nnrt/nnrt.pyi
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.geometry -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${STUBS_PACKAGE_DST_DIR}/nnrt ${PYTHON_LOCAL_PACKAGES_DIR}/nnrt
    COMMAND ${CMAKE_COMMAND} -E echo "copied ${STUBS_PACKAGE_DST_DIR}/nnrt to ${PYTHON_LOCAL_PACKAGES_DIR}"
    DEPENDS install-pip-package
)



add_subdirectory(tests)



