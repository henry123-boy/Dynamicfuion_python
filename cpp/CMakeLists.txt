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
endif ()

# We need to get python version to configure some meta files
message(STATUS "Using Python version: ${PYTHON_VERSION}")

set(PIP_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/pip_package")
set(PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python_package")
set(STUBS_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/stubs")


### ================== Output directory prep ==================

file(MAKE_DIRECTORY ${PIP_PACKAGE_DST_DIR})
# file(MAKE_DIRECTORY ${STUBS_PACKAGE_DST_DIR})


# ==================== set up NNRT C++ library ============================

set(
    NNRT_SOURCE_FILES

    cpu/graph_proc.h
    cpu/image_proc.h
    cpu/graph_proc.cpp
    cpu/image_proc.cpp

    debugging/debugging.hpp

    core/PlatformIndependence.h
    core/PlatformIndependentAtomics.h
    core/PlatformIndependentTuple.h

    core/kernel/KnnUtilities.h
    core/kernel/KnnUtilities_PriorityQueue.h

    core/KeyValuePair.h
    core/heap/DeviceHeap.h
    core/heap/CPU/DeviceHeapCPU.cpp
    core/heap/CUDA/DeviceHeapCUDA.cu
    core/heap/CPU/DeviceHeapCPU.h
    core/heap/CUDA/DeviceHeapCUDA.cuh
    core/heap/Heap.h
    core/heap/Heap.cpp
    core/heap/CPU/HostHeapCPU.h
    core/heap/CPU/HostHeapCPUImpl.h
    core/heap/CPU/HostHeapCPU.cpp
    core/heap/CUDA/HostHeapCUDA.h
    core/heap/CUDA/HostHeapCUDAImpl.cuh
    core/heap/CUDA/HostHeapCUDA.cu
    core/heap/CUDA/HostHeapCUDA.h

    core/functional/Masking.cpp
    core/functional/Masking.h

    core/KdTree.h
    core/KdTree.cpp
    core/kernel/KdTree.h
    core/kernel/KdTree.cpp
    core/kernel/BuildKdTreeImpl.h
    core/kernel/SearchKdTreeImpl.h
    core/kernel/instantiations/KdTreeCPU.cpp
    core/kernel/instantiations/KdTreeCUDA.cu

    core/kernel/KdTreeUtilities.cpp
    core/kernel/KdTreeNodeTypes.h

    core/LinearIndex.cpp
    core/LinearIndex.h
    core/kernel/LinearIndex.cpp
    core/kernel/LinearIndex.h
    core/kernel/LinearIndexImpl.h
    core/kernel/instantiations/LinearIndexCPU.cpp
    core/kernel/instantiations/LinearIndexCUDA.cu

    core/DeviceSelection.h

    core/TensorManipulationRoutines.h
    core/TensorManipulationRoutines.cpp

    core/kernel/HashTableUtilities.h

    core/linalg/Matmul3D.cpp
    core/linalg/Matmul3D.h
    core/linalg/Matmul3D_CPU.cpp
    core/linalg/Matmul3D_CUDA.cpp

    core/linalg/BlasWrapper.h
    core/linalg/LinalgHeadersCUDA.h
    core/linalg/LinalgHeadersCPU.h
    core/linalg/LinalgUtils.h
    core/linalg/LinalgUtils.cpp

    core/PreprocessorStrings.h

    core/functional/Sorting.cpp
    core/functional/Sorting.h
    core/functional/kernel/BubbleSort.h
    core/functional/kernel/Sorting.cpp
    core/functional/kernel/Sorting.h
    core/functional/kernel/SortingImpl.h
    core/functional/kernel/instantiations/SortingCPU.cpp
    core/functional/kernel/instantiations/SortingCUDA.cu

    core/functional/Comparisons.cpp
    core/functional/Comparisons.h
    core/functional/kernel/Comparisons.cpp
    core/functional/kernel/Comparisons.h
    core/functional/kernel/ComparisonsImpl.h
    core/functional/kernel/instantiations/ComparisonsCPU.cpp
    core/functional/kernel/instantiations/ComparisonsCUDA.cu


    string/split_string.h
    string/split_string.cpp
    string/join_string_separator.cpp
    string/join_string_separator.h

    geometry/functional/kernel/Defines.h
    geometry/NonRigidSurfaceVoxelBlockGrid.cpp
    geometry/NonRigidSurfaceVoxelBlockGrid.h

    geometry/kernel/NonRigidSurfaceVoxelBlockGrid.cpp
    geometry/kernel/NonRigidSurfaceVoxelBlockGrid.h
    geometry/kernel/instantiations/NonRigidSurfaceVoxelBlockGridCPU.cpp
    geometry/kernel/instantiations/NonRigidSurfaceVoxelBlockGridCUDA.cu
    geometry/kernel/NonRigidSurfaceVoxelBlockGridImpl.h

    geometry/GraphWarpField.h
    geometry/GraphWarpField.cpp

    geometry/functional/kernel/WarpAnchorComputation.h
    geometry/functional/kernel/WarpAnchorComputation.cpp
    geometry/functional/kernel/instantiations/WarpAnchorComputationCPU.cpp
    geometry/functional/kernel/instantiations/WarpAnchorComputationCUDA.cu
    geometry/functional/kernel/WarpAnchorComputationImpl.h

    geometry/functional/kernel/WarpUtilities.h
    geometry/functional/kernel/Warp3dPointsAndNormals.h
    geometry/functional/kernel/Warp3dPointsAndNormals.cpp
    geometry/functional/kernel/instantiations/Warp3dPointsAndNormalsCPU.cpp
    geometry/functional/kernel/instantiations/Warp3dPointsAndNormalsCUDA.cu
    geometry/functional/kernel/Warp3dPointsAndNormalsImpl.h

    geometry/functional/kernel/Comparison.h
    geometry/functional/kernel/Comparison.cpp
    geometry/functional/kernel/instantiations/ComparisonCPU.cpp
    geometry/functional/kernel/instantiations/ComparisonCUDA.cu
    geometry/functional/kernel/ComparisonImpl.h

    geometry/kernel/Segment.h

    geometry/functional/Downsample3dPoints.h
    geometry/functional/Downsample3dPoints.cpp

    geometry/kernel/PointDownsampling.cpp
    geometry/kernel/PointDownsampling.h
    geometry/kernel/PointDownsamplingImpl.h
    geometry/kernel/instantiations/PointDownsamplingCPU.cpp
    geometry/kernel/instantiations/PointDownsamplingCUDA.cu

    geometry/VoxelBlockGrid.h
    geometry/VoxelBlockGrid.cpp
    geometry/kernel/VoxelGridDtypeDispatch.h

    geometry/functional/AnchorComputationMethod.h
    geometry/TransformationMode.h
    geometry/functional/Comparison.h
    geometry/functional/Comparison.cpp

    geometry/functional/FindKNearestNeighbors.cpp
    geometry/functional/FindKNearestNeighbors.h

    geometry/functional/kernel/PointAggregationBin.h
    geometry/functional/kernel/IndexRaveling.h
    geometry/functional/kernel/PointCollectionExtent.h

    geometry/functional/NormalsOperations.h
    geometry/functional/NormalsOperations.cpp
    geometry/functional/kernel/NormalsOperations.cpp
    geometry/functional/kernel/NormalsOperations.h
    geometry/functional/kernel/NormalsOperationsImpl.h
    geometry/functional/kernel/instantiations/NormalsOperationsCPU.cpp
    geometry/functional/kernel/instantiations/NormalsOperationsCUDA.cu

    geometry/functional/Unproject3dPoints.cpp
    geometry/functional/Unproject3dPoints.h
    geometry/functional/kernel/PointCloud.h
    geometry/functional/kernel/PointCloud.cpp
    geometry/functional/kernel/PointCloudImpl.h
    geometry/kernel/instantiations/PointCloudCPU.cpp
    geometry/kernel/instantiations/PointCloudCUDA.cu

    geometry/functional/Warping.cpp
    geometry/functional/Warping.h
    geometry/functional/WarpAnchorComputation.cpp
    geometry/functional/WarpAnchorComputation.h

    io/TensorIO.cpp
    io/TensorIO.h
    io/SizeVectorIO.cpp
    io/SizeVectorIO.h
    io/DtypeIO.cpp
    io/DtypeIO.h
    io/BlobIO.cpp
    io/BlobIO.h
    io/DeviceIO.cpp
    io/DeviceIO.h
    io/VoxelBlockGridIO.cpp
    io/VoxelBlockGridIO.h
    io/FileStreamSelector.h

    rendering/RasterizeMesh.h
    rendering/RasterizeMesh.cpp
    rendering/functional/kernel/BarycentricCoordinates.h
    rendering/kernel/RasterizeMesh.cpp
    rendering/kernel/RasterizeMesh.h
    rendering/kernel/RasterizeMeshImpl.h
    rendering/kernel/instantiations/RasterizeMeshCPU.cpp
    rendering/kernel/instantiations/RasterizeMeshCUDA.cu
    rendering/kernel/RasterizationConstants.h
    rendering/kernel/CoordinateSystemConversions.h
    rendering/kernel/RayFaceIntersection.h

    rendering/kernel/RasterizeMeshImplCUDA.cuh
    rendering/kernel/GridBitMask.cuh
    rendering/kernel/RasterizeMeshImplCPU.h

    rendering/kernel/ExtractClippedFaceVertices.cpp
    rendering/kernel/ExtractClippedFaceVertices.h
    rendering/kernel/ExtractClippedFaceVerticesImpl.h
    rendering/kernel/instantiations/ExtractClippedFaceVerticesCPU.cpp
    rendering/kernel/instantiations/ExtractClippedFaceVerticesCUDA.cu

    rendering/functional/InterpolateFaceAttributes.cpp
    rendering/functional/InterpolateFaceAttributes.h
    rendering/functional/kernel/InterpolateFaceAttributes.cpp
    rendering/functional/kernel/InterpolateFaceAttributes.h
    rendering/functional/kernel/InterpolateFaceAttributesImpl.h
    rendering/functional/kernel/instantiations/InterpolateFaceAttributesCPU.cpp
    rendering/functional/kernel/instantiations/InterpolateFaceAttributesCUDA.cu
    rendering/functional/kernel/FrontFaceVertexOrder.h

    alignment/DeformableMeshRenderToImageFitter.cpp
    alignment/DeformableMeshRenderToRgbdImageFitter.h

    alignment/functional/WarpedMeshToReferencePointCloud_PointToPlane.cpp
    alignment/functional/WarpedMeshToReferencePointCloud_PointToPlane.h
    alignment/functional/Jacobians.cpp
    alignment/functional/Jacobians.h
    alignment/functional/kernel/Jacobians.cpp
    alignment/functional/kernel/Jacobians.h
    alignment/functional/kernel/JacobiansImpl.h
    alignment/functional/kernel/BarycentricCoordinateJacobians.h
    alignment/functional/kernel/instantiations/JacobiansCPU.cpp
    alignment/functional/kernel/instantiations/JacobiansCUDA.cu
    alignment/functional/kernel/MathTypedefs.h alignment/functional/kernel/ProjectionJacobians.h alignment/functional/kernel/ProjectionJacobians.h core/PlatformIndependentArray.h rendering/EdgeShader.cpp rendering/EdgeShader.h core/PlatformIndependentAlgorithm.h)

add_library(nnrt_cpp STATIC ${NNRT_SOURCE_FILES})

target_include_directories(
    nnrt_cpp PUBLIC

    ${PYTHON_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
nnrt_show_and_abort_on_warning(nnrt_cpp)

nnrt_link_3rdparty_libraries(nnrt_cpp)
nnrt_set_global_properties(nnrt_cpp)
target_link_libraries(nnrt_cpp PUBLIC CUDA::toolkit)

# ==================== Benchmark Executables =========================================
add_subdirectory(bench)
# ==================== set up NNRT Python binding library ============================
set(
    PYBIND_SOURCE_FILES

    3rd_party/magic_enum.hpp
    pybind/nnrt_pybind.h
    pybind/nnrt_pybind.cpp
    pybind/geometry/geometry.cpp
    pybind/geometry/geometry.h
    pybind/geometry/functional/functional.cpp
    pybind/geometry/functional/functional.h
    pybind/enum_export.h
    pybind/core.cpp
    pybind/core.h
    pybind/io.cpp
    pybind/io.h
)

pybind11_add_module(
    nnrt
    ${PYBIND_SOURCE_FILES}
)

target_include_directories(nnrt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(nnrt PUBLIC nnrt_cpp)

include(ProcessorCount)
set(N_PROCESSORS 1)
processorcount(N)
if (NOT N LESS 2)
    math(EXPR N_PROCESSORS "${N} - 1")
endif ()

set(Open3D_DIRECTIVE)
if (Open3D_DIR)
    set(Open3D_DIRECTIVE "Open3D_DIR=${Open3D_DIR}")
endif ()

set(Python3_VERSION_DIRECTIVE)
if (DEFINED Python3_VERSION)
    set(Python3_VERSION_DIRECTIVE "Python3_VERSION=${Python3_VERSION}")
endif ()

set(BUILD_CUDA_MODULE_DIRECTIVE)
if (DEFINED BUILD_CUDA_MODULE)
    set(BUILD_CUDA_MODULE_DIRECTIVE "BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}")
endif ()


# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
    COMMAND ${CMAKE_COMMAND} -E env ${Open3D_DIRECTIVE} ${Python3_VERSION_DIRECTIVE} ${BUILD_CUDA_MODULE_DIRECTIVE}
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


if (NOT DEFINED STUBGEN_EXECUTABLE)
    set(STUBGEN_EXECUTABLE stubgen)
endif ()

add_custom_target(install-python-stubs
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${STUBS_PACKAGE_DST_DIR}/nnrt
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt -o ${STUBS_PACKAGE_DST_DIR}/nnrt WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}

    COMMAND ${CMAKE_COMMAND} -E echo "import nnrt.geometry.functional" >> ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi
    COMMAND ${CMAKE_COMMAND} -E echo "import nnrt.geometry" >> ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi
    COMMAND ${CMAKE_COMMAND} -E echo "import nnrt.core" >> ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi
    COMMAND ${CMAKE_COMMAND} -E cat ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi ${STUBS_PACKAGE_DST_DIR}/nnrt/nnrt.pyi >> ${STUBS_PACKAGE_DST_DIR}/nnrt/__init__.pyi
    COMMAND ${CMAKE_COMMAND} -E rm -f ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi ${STUBS_PACKAGE_DST_DIR}/nnrt/nnrt.pyi

    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.geometry.functional -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.geometry -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.core -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}

    COMMAND ${CMAKE_COMMAND} -E copy_directory ${STUBS_PACKAGE_DST_DIR}/nnrt ${PYTHON_LOCAL_PACKAGES_DIR}/nnrt
    COMMAND ${CMAKE_COMMAND} -E echo "copied ${STUBS_PACKAGE_DST_DIR}/nnrt to ${PYTHON_LOCAL_PACKAGES_DIR}"
    DEPENDS install-pip-package
    )


add_subdirectory(tests)



