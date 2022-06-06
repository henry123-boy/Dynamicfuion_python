#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph).
#  Copyright (c) 2021 Gregory Kramida
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

from ext_argparse import ParameterEnum, Parameter
from enum import Enum

from typing import Type

from data.presets import FrameSequencePreset


class VisualizationMode(Enum):
    NONE = 0
    CANONICAL_MESH = 1
    WARPED_MESH = 2
    POINT_CLOUD_TRACKING = 3
    COMBINED = 4


class SourceImageMode(Enum):
    IMAGE_ONLY = 0
    RENDERED_ONLY = 1
    RENDERED_WITH_PREVIOUS_FRAME_OVERLAY = 2


class GraphGenerationMode(Enum):
    FIRST_FRAME_EXTRACTED_MESH = 0
    FIRST_FRAME_DEPTH_IMAGE = 1
    FIRST_FRAME_LOADED_GRAPH = 2


class AnchorComputationMode(Enum):
    EUCLIDEAN = 0
    SHORTEST_PATH = 1
    PRECOMPUTED = 2


class TrackingSpanMode(Enum):
    FIRST_TO_CURRENT = 0
    PREVIOUS_TO_CURRENT = 1
    KEYFRAME_TO_CURRENT = 2


class MeshExtractionWeightThresholdingMode(Enum):
    CONSTANT = 0
    RAMP_UP_TO_CONSTANT = 1


class TrackingParameters(ParameterEnum):
    source_image_mode = \
        Parameter(default=SourceImageMode.IMAGE_ONLY, arg_type=SourceImageMode,
                  arg_help="How to generate the image source RGBD image pair for tracking/alignment toward the target "
                           "image (next RGBD image pair in the sequence.)")
    graph_generation_mode = \
        Parameter(default=GraphGenerationMode.FIRST_FRAME_DEPTH_IMAGE, arg_type=GraphGenerationMode,
                  arg_help="Method used to generate the graph inside the moving structures in the scene (i.e. a motion "
                           "proxy data structure that is used to store and play back the estimated surface motion).")
    pixel_anchor_computation_mode = \
        Parameter(default=AnchorComputationMode.PRECOMPUTED, arg_type=AnchorComputationMode,
                  arg_help="Method used to assign graph nodes as anchors to each pixel and compute their weights, "
                           "which control the influence of the graph on the estimated surface.")
    tracking_span_mode = \
        Parameter(default=TrackingSpanMode.KEYFRAME_TO_CURRENT, arg_type=TrackingSpanMode,
                  arg_help="Interval over which to perform alignment for tracking objects. FIRST_TO_CURRENT mode will "
                           "make the program track between the first frame and each incoming sequence frame. "
                           "PREVIOUS_TO_CURRENT will make it track between each consecutive pair of frames."
                           "KEYFRAME_TO_CURRENT will track between the latest keyframe and the current frame, keyframes"
                           "being sampled sparsely from the sequence.")

    keyframe_interval = \
        Parameter(default=50, arg_type=int,
                  arg_help="When KEYFRAME_TO_CURRENT tracking_span_mode is used, controls the uniform intervals "
                           "between which keyframes are sampled.")

    mesh_extraction_weight_thresholding_mode = \
        Parameter(default=MeshExtractionWeightThresholdingMode.RAMP_UP_TO_CONSTANT,
                  arg_type=MeshExtractionWeightThresholdingMode,
                  arg_help="Mesh extraction from TSDF for both the canonical and the forward-warped mesh will be guided"
                           " by this schema. When set to `CONSTANT`, the `mesh_extraction_weight_threshold` will "
                           " dictate directly what is the minimum voxel weight for the voxel to be considered for "
                           " extracting mesh geometry from. Per the weighted averaging integration schema, with each "
                           " time a voxel has been integrated new data into, its weight increases by one. When set to"
                           " `RAMP_UP_TO_CONSTANT, weight 0 will be considered for all voxels in the first frame,"
                           " then the threshold will increase by one with each frame until reaching the specified "
                           " threshold.")

    mesh_extraction_weight_threshold = \
        Parameter(default=10, arg_type=int,
                  arg_help="The weight threshold used to control which voxels are considered for extracting the "
                           "canonical and forward-warped mesh, see `mesh_extraction_weight_thresholding_mode` for "
                           "details.")


class IntegrationParameters(ParameterEnum):
    anchor_node_count = \
        Parameter(default=4, arg_type=int,
                  arg_help="Number of nodes used as \"anchors\" for a point on a surface. The anchors are used to"
                           " control the motion of the point they've been assigned to control by linearly blending"
                           " the transformation matrices associated with each node.")
    fusion_minimum_valid_anchor_count = \
        Parameter(default=3, arg_type=int,
                  arg_help="TSDF voxels which have fewer than this number of valid anchors will not have any new data "
                           "fused in from an incoming RGBD image pair. Valid anchors for a specific voxel are graph "
                           "nodes that are closer than a specific distance threshold from this voxel.")


class VerbosityParameters(ParameterEnum):
    print_frame_info = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Print number and source (e.g. file paths) of the current RGBD frame in the sequence before "
                           "processing it.")
    print_intrinsics = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Print the camera intrinsic matrix before processing the sequence.")
    print_cuda_memory_info = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Print CUDA memory information before processing each frame.")
    print_total_runtime = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Print the total runtime, in seconds, after processing the entire sequence.")


class VisualizationParameters(ParameterEnum):
    visualization_mode = \
        Parameter(default=VisualizationMode.NONE, arg_type=VisualizationMode,
                  arg_help="Controls extra visualization during the runtime of the fusion program.")


# TODO: Remove the "_to_disk" suffixes where present
class LoggingParameters(ParameterEnum):
    record_visualization_to_disk = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Record the visualization result to disk (as a movie). [WARNING: CURRENTLY, BROKEN]")
    record_canonical_meshes_to_disk = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Record canonical (reference) meshes to disk, i.e. the TSDF with all acquired geometry in "
                           "the time reference frame of the sequence start.")
    record_warped_meshes_to_disk = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Record the warped (deformed) mesh to disk at each frame.")
    record_rendered_warped_mesh = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Record the rendering of the warped (deformed) mesh to the camera plane at each frame.")
    record_gn_point_clouds = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Record the source point cloud being aligned at each Gauss-Newton iteration.")
    record_source_and_target_point_clouds = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Record the point clouds generated from source and target RGB-D frame before processing "
                           "each new frame in the sequence")
    record_correspondences = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Record the source points that have unfiltered correspondences, together with their "
                           "correspondences in the target frame. Not that this is different from just recording "
                           "the source and target point clouds, since correspondences are floating point coordinates "
                           "that are destinations of each source vector.")
    record_graph_transformations = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Record node original positions, rotations, and translations at each frame after non-rigid "
                           "alignment.")
    record_frameviewer_metadata = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Enables setting up settings for the frameviewer app, in order for the (potentially masked)"
                           "input frames to be viewed concurrently with the output in the visualizer app")


class TelemetryParameters(ParameterEnum):
    verbosity: Type[VerbosityParameters] = VerbosityParameters
    visualization: Type[VisualizationParameters] = VisualizationParameters
    logging: Type[LoggingParameters] = LoggingParameters


class InputDataParameters(ParameterEnum):
    sequence_preset = \
        Parameter(default=FrameSequencePreset.BERLIN_SOD_MASKS, arg_type=FrameSequencePreset,
                  arg_help="Which sequence preset to use during the run.")
    run_until_frame = \
        Parameter(default=-1, arg_type=int, arg_help="Stop processing sequence before the specified frame index. "
                                                     "When set to -1, processes sequence to the end.")
    start_at_frame = \
        Parameter(default=-1, arg_type=int, arg_help="Start processing sequence before the specified frame index. "
                                                     "When set to -1, processes sequence from the start.")


class FusionParameters(ParameterEnum):
    input_data: Type[InputDataParameters] = InputDataParameters
    tracking: Type[TrackingParameters] = TrackingParameters
    integration: Type[IntegrationParameters] = IntegrationParameters
    telemetry: Type[TelemetryParameters] = TelemetryParameters
