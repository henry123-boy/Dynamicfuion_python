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

    # legacy NNRT code
    cpu/graph_proc.h
    cpu/image_proc.h
    cpu/graph_proc.cpp
    cpu/image_proc.cpp

    # new library code (Greg Kramida)

    core/Dispatch.h
    core/PreprocessorStrings.h

    core/platform_independence/Qualifiers.h
    core/platform_independence/Atomics.h
    core/platform_independence/AtomicCounterArray.h
    core/platform_independence/Algorithm.h
    core/platform_independence/Array.h
    core/platform_independence/Tuple.h
    core/platform_independence/Macros.h


    core/KdTree.h
    core/KdTree.cpp

    core/kernel/KnnUtilities.h
    core/kernel/KnnUtilities_PriorityQueue.h
    core/kernel/KdTreeUtilities.cpp

    core/kernel/KdTreeNodeTypes.h
    core/kernel/KdTree.h
    core/kernel/KdTree.cpp
    core/kernel/BuildKdTreeImpl.h
    core/kernel/SearchKdTreeImpl.h
    core/kernel/instantiations/KdTreeCPU.cpp
    core/kernel/instantiations/KdTreeCUDA.cu

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

    core/functional/MaskingImpl.h
    core/functional/Masking.h
    core/functional/instantiations/Masking.cpp

    core/LinearIndex.cpp
    core/LinearIndex.h
    core/kernel/LinearIndex.cpp
    core/kernel/LinearIndex.h
    core/kernel/LinearIndexImpl.h
    core/kernel/instantiations/LinearIndexCPU.cpp
    core/kernel/instantiations/LinearIndexCUDA.cu

    core/DeviceSelection.h
    core/ParallelFor.h

    core/TensorManipulationRoutines.h
    core/TensorManipulationRoutines.cpp

    core/kernel/HashTableUtilities.h

    core/kernel/MathTypedefs.h

    core/linalg/Matmul3D.cpp
    core/linalg/Matmul3D.h
    core/linalg/Matmul3D_CPU.cpp
    core/linalg/Matmul3D_CUDA.cpp

    core/linalg/BlasWrapper.h
    core/linalg/LinalgHeadersCUDA.h
    core/linalg/LinalgHeadersCPU.h
    core/linalg/LinalgUtils.h
    core/linalg/LinalgUtils.cpp
    core/linalg/LapackWrapper.h
    core/linalg/SolveBlockDiagonalCholeskyCPU.cpp
    core/linalg/SolveBlockDiagonalCholesky.h
    core/linalg/SolveBlockDiagonalCholesky.cpp
    core/linalg/SolveBlockDiagonalCholeskyCUDA.cpp
    core/linalg/PointerAggregationForBatchOperationsCPU.h
    core/linalg/Rodrigues.cpp
    core/linalg/Rodrigues.h
    core/linalg/RodriguesImpl.h
    core/linalg/instantiations/RodriguesCPU.cpp
    core/linalg/instantiations/RodriguesCUDA.cu
    core/linalg/KroneckerTensorProduct.h

    core/functional/SortOrder.h
    core/functional/Sorting.cpp
    core/functional/Sorting.h

    core/functional/kernel/SortingAlongLastDimensionByKeyImpl.h
    core/functional/kernel/instantiations/SortingAlongLastDimensionByKeyCUDA.cu
    core/functional/kernel/instantiations/SortingAlongLastDimensionByKeyCPU.cpp

    core/functional/kernel/BubbleSort.h
    core/functional/kernel/Sorting.cpp
    core/functional/kernel/Sorting.h

    core/functional/kernel/SortingAlongLastDimensionImpl.h
    core/functional/kernel/instantiations/SortingAlongLastDimensionCPU.cpp
    core/functional/kernel/instantiations/SortingAlongLastDimensionCUDA.cu

    core/functional/kernel/SortingByColumnImpl.h
    core/functional/kernel/instantiations/SortingByColumnCPU.cpp
    core/functional/kernel/instantiations/SortingByColumnCUDA.cu

    core/functional/Comparisons.cpp
    core/functional/Comparisons.h
    core/functional/kernel/Comparisons.cpp
    core/functional/kernel/Comparisons.h
    core/functional/kernel/ComparisonsImpl.h
    core/functional/kernel/instantiations/ComparisonsCPU.cpp
    core/functional/kernel/instantiations/ComparisonsCUDA.cu

    core/functional/ParallelPrefixScan.cpp
    core/functional/ParallelPrefixScan.h
    core/functional/kernel/ParallelPrefixScanImpl.h
    core/functional/kernel/ParallelPrefixScan.h
    core/functional/kernel/ParallelPrefixScan.cpp

    core/TensorRepresentationConversion.h
    core/GetDType.cpp
    core/GetDType.h

    core/linalg/SolveBlockDiagonalQR.cpp
    core/linalg/SolveBlockDiagonalQR.h
    core/linalg/SolveBlockDiagonalGeneric.h

    core/linalg/SolveBlockDiagonalQR_CPU.cpp
    core/linalg/SolveBlockDiagonalQR_CUDA.cpp

    core/functional/kernel/instantiations/ParallelPrefixScanCPU.cpp
    core/functional/kernel/instantiations/ParallelPrefixScanCUDA.cu

    core/linalg/ZeroOutTriangularBlocks.cpp
    core/linalg/ZeroOutTriangularBlocks.h
    core/linalg/ZeroOutTriangularBlocksImpl.h
    core/linalg/instantiations/ZeroOutTriangularBlocksCPU.cpp
    core/linalg/instantiations/ZeroOutTriangularBlocksCUDA.cu

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

    geometry/HierarchicalGraphWarpField.h
    geometry/HierarchicalGraphWarpField.cpp

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

    geometry/functional/kernel/PointToPlaneDistances.h
    geometry/functional/kernel/PointToPlaneDistances.cpp
    geometry/functional/kernel/instantiations/PointToPlaneDistancesCPU.cpp
    geometry/functional/kernel/instantiations/PointToPlaneDistancesCUDA.cu
    geometry/functional/kernel/PointToPlaneDistancesImpl.h

    geometry/kernel/Segment.h
    geometry/kernel/AxisAlignedBoundingBox.h

    geometry/functional/GeometrySampling.h
    geometry/functional/GeometrySampling.cpp

    geometry/functional/kernel/GeometrySampling.cpp
    geometry/functional/kernel/GeometrySampling.h
    geometry/functional/kernel/GeometrySamplingImpl.h
    geometry/functional/kernel/GeometrySamplingMean.h
    geometry/functional/kernel/GeometrySamplingMean.h
    geometry/functional/kernel/GeometrySamplingMedian.h
    geometry/functional/kernel/GeometrySamplingMedian.h
    geometry/functional/kernel/GeometrySamplingGridBinning.h
    geometry/functional/kernel/instantiations/GeometrySamplingCPU.cpp
    geometry/functional/kernel/instantiations/GeometrySamplingCUDA.cu


    geometry/VoxelBlockGrid.h
    geometry/VoxelBlockGrid.cpp
    geometry/kernel/VoxelGridDtypeDispatch.h

    geometry/functional/AnchorComputationMethod.h
    geometry/TransformationMode.h
    geometry/functional/PointToPlaneDistances.h
    geometry/functional/PointToPlaneDistances.cpp

    geometry/functional/FindKNearestNeighbors.cpp
    geometry/functional/FindKNearestNeighbors.h

    geometry/functional/kernel/PointAggregationBins.h
    geometry/functional/kernel/IndexRaveling.h
    geometry/functional/kernel/PointCollectionExtent.h

    geometry/functional/NormalsOperations.h
    geometry/functional/NormalsOperations.cpp
    geometry/functional/kernel/NormalsOperations.cpp
    geometry/functional/kernel/NormalsOperations.h
    geometry/functional/kernel/NormalsOperationsImpl.h
    geometry/functional/kernel/instantiations/NormalsOperationsCPU.cpp
    geometry/functional/kernel/instantiations/NormalsOperationsCUDA.cu

    geometry/functional/PerspectiveProjection.cpp
    geometry/functional/PerspectiveProjection.h
    geometry/functional/kernel/PerspectiveProjection.h
    geometry/functional/kernel/PerspectiveProjection.cpp
    geometry/functional/kernel/PerspectiveProjectionImpl.h
    geometry/functional/kernel/instantiations/PerspectiveProjectionCPU.cpp
    geometry/functional/kernel/instantiations/PerspectiveProjectionCUDA.cu

    geometry/functional/Warping.cpp
    geometry/functional/Warping.h
    geometry/functional/WarpAnchorComputation.cpp
    geometry/functional/WarpAnchorComputation.h

    geometry/functional/MeshFrom2dTriangle.cpp
    geometry/functional/MeshFrom2dTriangle.h
    geometry/functional/JoinTriangleMeshes.cpp
    geometry/functional/JoinTriangleMeshes.h

    geometry/kernel/HierarchicalGraphWarpField.cpp
    geometry/kernel/HierarchicalGraphWarpField.h
    geometry/RegularizationLayer.h
    geometry/kernel/HierarchicalGraphWarpFieldImpl.h
    geometry/kernel/instantiations/HierarchicalGraphWarpFieldCPU.cpp
    geometry/kernel/instantiations/HierarchicalGraphWarpFieldCUDA.cu

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

    rendering/RasterizeNdcTriangles.h
    rendering/RasterizeNdcTriangles.cpp

    rendering/VertexColorShader.cpp
    rendering/VertexColorShader.h
    rendering/FlatEdgeShader.cpp
    rendering/FlatEdgeShader.h

    rendering/kernel/FlatEdgeShader.cpp
    rendering/kernel/FlatEdgeShader.h
    rendering/kernel/FlatEdgeShaderImpl.h
    rendering/kernel/instantiations/FlatEdgeShaderCPU.cpp
    rendering/kernel/instantiations/FlatEdgeShaderCUDA.cu

    rendering/kernel/RasterizeNdcTriangles.cpp
    rendering/kernel/RasterizeNdcTriangles.h
    rendering/kernel/RasterizeNdcTrianglesImpl.h
    rendering/kernel/instantiations/RasterizeNdcTrianglesCPU.cpp
    rendering/kernel/instantiations/RasterizeNdcTrianglesCUDA.cu
    rendering/kernel/RasterizationConstants.h
    rendering/kernel/CoordinateSystemConversions.h
    rendering/kernel/RayFaceIntersection.h
    rendering/kernel/RasterizeNdcTrianglesCUDA.cuh
    rendering/kernel/GridBitMask.cuh
    rendering/kernel/RasterizeNdcTrianglesImplCPU.h

    rendering/functional/InterpolateVertexAttributes.cpp
    rendering/functional/InterpolateVertexAttributes.h
    rendering/functional/kernel/InterpolateFaceAttributes.cpp
    rendering/functional/kernel/InterpolateFaceAttributes.h
    rendering/functional/kernel/InterpolateFaceAttributesImpl.h
    rendering/functional/kernel/instantiations/InterpolateFaceAttributesCPU.cpp
    rendering/functional/kernel/instantiations/InterpolateFaceAttributesCUDA.cu

    rendering/functional/ExtractFaceVertices.cpp
    rendering/functional/ExtractFaceVertices.h
    rendering/functional/kernel/ExtractClippedFaceVertices.cpp
    rendering/functional/kernel/ExtractClippedFaceVertices.h
    rendering/functional/kernel/ExtractClippedFaceVerticesImpl.h
    rendering/functional/kernel/instantiations/ExtractClippedFaceVerticesCPU.cpp
    rendering/functional/kernel/instantiations/ExtractClippedFaceVerticesCUDA.cu

    rendering/functional/kernel/FrontFaceVertexOrder.h
    rendering/functional/kernel/BarycentricCoordinates.h

    alignment/DeformableMeshToImageFitter.cpp
    alignment/DeformableMeshToImageFitter.h

    alignment/kernel/DeformableMeshToImageFitter.cpp
    alignment/kernel/DeformableMeshToImageFitter.h
    alignment/kernel/DeformableMeshToImageFitterImpl.h
    alignment/kernel/instantiations/DeformableMeshToImageFitterCPU.cpp
    alignment/kernel/instantiations/DeformableMeshToImageFitterCUDA.cu

    alignment/functional/WarpedMeshToReferencePointCloud_PointToPlane.cpp
    alignment/functional/WarpedMeshToReferencePointCloud_PointToPlane.h
    alignment/functional/RasterizedSurfaceJacobians.cpp
    alignment/functional/RasterizedSurfaceJacobians.h
    alignment/functional/WarpedSurfaceJacobians.cpp
    alignment/functional/WarpedSurfaceJacobians.h

    alignment/functional/kernel/WarpedSurfaceJacobians.cpp
    alignment/functional/kernel/WarpedSurfaceJacobians.h
    alignment/functional/kernel/WarpedSurfaceJacobiansImpl.h
    alignment/functional/kernel/BarycentricCoordinateJacobians.h
    alignment/functional/kernel/instantiations/WarpedSurfaceJacobiansCPU.cpp
    alignment/functional/kernel/instantiations/WarpedSurfaceJacobiansCUDA.cu
    alignment/functional/kernel/ProjectionJacobians.h
    alignment/functional/kernel/RasterizedSurfaceJacobians.cpp
    alignment/functional/kernel/RasterizedSurfaceJacobians.h
    alignment/functional/kernel/RasterizedSurfaceJacobiansImpl.h
    alignment/functional/kernel/instantiations/RasterizedSurfaceJacobiansCPU.cpp
    alignment/functional/kernel/instantiations/RasterizedSurfaceJacobiansCUDA.cu
    alignment/functional/PixelVertexAnchorJacobians.cpp
    alignment/functional/PixelVertexAnchorJacobians.h
    alignment/functional/kernel/PixelVertexAnchorJacobians.cpp
    alignment/functional/kernel/PixelVertexAnchorJacobians.h
    alignment/functional/kernel/PixelVertexAnchorJacobiansImpl.h
    alignment/functional/kernel/instantiations/PixelVertexAnchorJacobiansCPU.cpp
    alignment/functional/kernel/instantiations/PixelVertexAnchorJacobiansCUDA.cu
    alignment/IterationMode.h

    alignment/functional/AssociateFacesWithAnchors.cpp
    alignment/functional/AssociateFacesWithAnchors.h
    alignment/functional/FaceNodeAnchors.h

    alignment/functional/kernel/AssociateFacesWithAnchors.cpp
    alignment/functional/kernel/AssociateFacesWithAnchors.h
    alignment/functional/kernel/instantiations/AssociateFacesWithAnchorsCPU.cpp
    alignment/functional/kernel/instantiations/AssociateFacesWithAnchorsCUDA.cu
    alignment/functional/kernel/AssociateFacesWithAnchorsImpl.h

    alignment/functional/ArapJacobian.cpp
    alignment/functional/ArapJacobian.h

    alignment/functional/kernel/ArapJacobian.cpp
    alignment/functional/kernel/ArapJacobian.h
    alignment/functional/kernel/ArapJacobianImpl.h
    alignment/functional/kernel/instantiations/ArapJacobianCPU.cpp
    alignment/functional/kernel/instantiations/ArapJacobianCUDA.cu

    geometry/functional/ComputeDistanceMatrix.cpp
    geometry/functional/ComputeDistanceMatrix.h
    geometry/functional/kernel/ComputeDistanceMatrix.cpp
    geometry/functional/kernel/ComputeDistanceMatrix.h
    geometry/functional/kernel/ComputeDistanceMatrixImpl.h
    geometry/functional/kernel/instantiations/ComputeDistanceMatrixCPU.cpp
    geometry/functional/kernel/instantiations/ComputeDistanceMatrixCUDA.cu

    geometry/functional/TopologicalConversions.cpp
    geometry/functional/TopologicalConversions.h
    geometry/functional/kernel/TopologicalConversions.cpp
    geometry/functional/kernel/TopologicalConversions.h
    geometry/functional/kernel/TopologicalConversionsImpl.h
    geometry/functional/kernel/instantiations/TopologicalConversionsCPU.cpp
    geometry/functional/kernel/instantiations/TopologicalConversionsCUDA.cu

    geometry/WarpField.cpp
    geometry/WarpField.h
    geometry/PlanarGraphWarpField.cpp
    geometry/PlanarGraphWarpField.h

    core/platform_independence/AtomicKeyIndexArray.cuh

    geometry/WarpNodeCoverageComputationMethod.h
    alignment/functional/ArapHessian.cpp
    alignment/functional/ArapHessian.h
    alignment/functional/kernel/ArapHessian.cpp
    alignment/functional/kernel/ArapHessian.h
    alignment/functional/kernel/ArapHessianImpl.h
    alignment/functional/kernel/instantiations/ArapHessianCPU.cpp
    alignment/functional/kernel/instantiations/ArapHessianCUDA.cu

    core/linalg/FactorizeBlocksCholesky.cpp
    core/linalg/FactorizeBlocksCholesky.h
    core/linalg/FactorizeBlocksCholeskyCPU.cpp
    core/linalg/FactorizeBlocksCholeskyCUDA.cpp
    core/linalg/InvertBlocks.cpp
    core/linalg/InvertBlocks.h
    core/linalg/UpLoTriangular.h
    core/linalg/InvertBlocksCPU.cpp
    core/linalg/InvertBlocksCUDA.cu

    core/linalg/MagmaManager.cpp
    core/linalg/MagmaManager.h

    core/CUDAUtils.h
    core/linalg/PointerAggregationForBatchOperationsCUDA.cuh

    core/linalg/SolveBlockSparseArrowheadCholesky.cpp
    core/linalg/SolveBlockSparseArrowheadCholesky.h

    core/linalg/BlockSparseArrowheadMatrix.h

    core/linalg/MatmulBlockSparse.cpp
    core/linalg/MatmulBlockSparse.h
    core/linalg/MatmulBlockSparseImpl.h
    core/linalg/MatmulBlockSparseImpl.h
    core/linalg/instantiations/MatmulBlockSparseCPU.cpp
    core/linalg/instantiations/MatmulBlockSparseCUDA.cu

    core/functional/Tile.cpp
    core/functional/Tile.h
    core/functional/kernel/Tile.cpp
    core/functional/kernel/Tile.h
    core/functional/kernel/TileImpl.h
    core/functional/kernel/instantiations/TileCPU.cpp
    core/functional/kernel/instantiations/TileCUDA.cu

    core/linalg/LinalgKernels.cuh
    core/linalg/BlasAuxiliary.h

    core/linalg/DiagonalBlocks.cpp
    core/linalg/DiagonalBlocks.h
    core/linalg/instantiations/DiagonalBlocksCUDA.cu
    core/linalg/instantiations/DiagonalBlocksCPU.cpp
    core/linalg/DiagonalBlocksImpl.h

    core/linalg/TransposeBlocks.h
    core/linalg/TransposeBlocks.cpp
    core/linalg/TransposeBlocks.cpp
    core/linalg/TransposeBlocksCPU.cpp
    core/linalg/TransposeBlocksCUDA.cu

    core/platform_independence/AtomicTensor.h

    core/linalg/SchurComplement.cpp
    core/linalg/SchurComplement.h

    core/linalg/BlockSums.cpp
    core/linalg/BlockSums.h
    core/linalg/BlockSumsImpl.h
    core/linalg/instantiations/BlockSumsCPU.cpp
    core/linalg/instantiations/BlockSumsCUDA.cu

    core/linalg/SparseBlocks.cpp
    core/linalg/SparseBlocks.h
    core/linalg/SparseBlocks.h
    core/linalg/SparseBlocksImpl.h
    core/linalg/instantiations/SparseBlocksCPU.cpp
    core/linalg/instantiations/SparseBlocksCUDA.cu
    core/linalg/instantiations/SparseBlocksCUDA.cu

    core/linalg/PreconditionDiagonalBlocks.cpp
    core/linalg/PreconditionDiagonalBlocks.h
    core/linalg/PreconditionDiagonalBlocksImpl.h
    core/linalg/instantiations/PreconditionDiagonalBlocksCPU.cpp
    core/linalg/instantiations/PreconditionDiagonalBlocksCUDA.cu

    core/linalg/MatrixPreprocessingOperation.h

    core/linalg/SolveCholesky.cpp
    core/linalg/SolveCholesky.h
    core/linalg/SolveCholeskyCPU.cpp
    core/linalg/SolveCholeskyCUDA.cpp


)

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
    pybind/rendering/rendering.cpp
    pybind/rendering/rendering.h
    pybind/rendering/functional/functional.cpp
    pybind/rendering/functional/functional.h
    pybind/enum_export.h
    pybind/core/core.cpp
    pybind/core/core.h
    pybind/core/linalg/linalg.cpp
    pybind/core/linalg/linalg.h
    pybind/io.cpp
    pybind/io.h
)

pybind11_add_module(
    nnrt
    ${PYBIND_SOURCE_FILES}
)

target_link_directories(nnrt PUBLIC /home/algomorph/Builds/NeuralTracking/cmake-build-debug/magma/lib)
target_link_directories(nnrt PUBLIC /home/algomorph/Builds/NeuralTracking/cmake-build-debug)
target_link_libraries(nnrt PUBLIC magma)
target_link_libraries(nnrt PUBLIC /home/algomorph/Builds/NeuralTracking/cmake-build-debug/magma/lib/libmagma.a)


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
    COMMAND ${CMAKE_COMMAND} -E echo "import nnrt.rendering.functional" >> ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi
    COMMAND ${CMAKE_COMMAND} -E echo "import nnrt.rendering" >> ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi
    COMMAND ${CMAKE_COMMAND} -E echo "import nnrt.core.linalg" >> ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi
    COMMAND ${CMAKE_COMMAND} -E echo "import nnrt.core" >> ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi
    COMMAND ${CMAKE_COMMAND} -E cat ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi ${STUBS_PACKAGE_DST_DIR}/nnrt/nnrt.pyi >> ${STUBS_PACKAGE_DST_DIR}/nnrt/__init__.pyi
    COMMAND ${CMAKE_COMMAND} -E rm -f ${STUBS_PACKAGE_DST_DIR}/nnrt/import.pyi ${STUBS_PACKAGE_DST_DIR}/nnrt/nnrt.pyi

    #TODO: import structure here needs to be fixed to work hierarchically, i.e. when you go one level deep to nnrt.geometry.functional,
    # IDEs are unable to parse .functional (since it's not inside the geometry.pyi file)
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.geometry.functional -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.geometry -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.rendering.functional -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.rendering -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.core.linalg -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}
    COMMAND ${STUBGEN_EXECUTABLE} -m nnrt.core -o ${STUBS_PACKAGE_DST_DIR} WORKING_DIRECTORY ${PYTHON_PACKAGE_DIR}

    COMMAND ${CMAKE_COMMAND} -E copy_directory ${STUBS_PACKAGE_DST_DIR}/nnrt ${PYTHON_LOCAL_PACKAGES_DIR}/nnrt
    COMMAND ${CMAKE_COMMAND} -E echo "copied ${STUBS_PACKAGE_DST_DIR}/nnrt to ${PYTHON_LOCAL_PACKAGES_DIR}"
    DEPENDS install-pip-package
    )


add_subdirectory(tests)

