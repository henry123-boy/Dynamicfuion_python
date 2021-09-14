from ext_argparse import ParameterEnum, Parameter
from enum import Enum

from data import FrameSequencePreset


class VisualizationMode(Enum):
    NONE = 0
    CANONICAL_MESH = 1
    WARPED_MESH = 2
    POINT_CLOUD_TRACKING = 3
    COMBINED = 4


class SourceImageMode(Enum):
    REUSE_PREVIOUS_FRAME = 0
    RENDERED_ONLY = 1
    RENDERED_WITH_PREVIOUS_FRAME_OVERLAY = 2


class TransformationMode(Enum):
    QUATERNIONS = 0
    MATRICES = 1


class GraphGenerationMode(Enum):
    FIRST_FRAME_EXTRACTED_MESH = 0
    FIRST_FRAME_DEPTH_IMAGE = 1
    FIRST_FRAME_LOADED_GRAPH = 2


class AnchorComputationMode(Enum):
    EUCLIDEAN = 0
    SHORTEST_PATH = 1
    PRECOMPUTED = 2


class TrackingSpanMode(Enum):
    ZERO_TO_T = 0
    T_MINUS_ONE_TO_T = 1


class TrackingParameters(ParameterEnum):
    source_image_mode = \
        Parameter(default=SourceImageMode.REUSE_PREVIOUS_FRAME, arg_type=SourceImageMode,
                  arg_help="How to generate the image source RGBD image pair for tracking/alignment toward the target "
                           "image (next RGBD image pair in the sequence.)")
    graph_generation_mode = \
        Parameter(default=GraphGenerationMode.FIRST_FRAME_DEPTH_IMAGE, arg_type=GraphGenerationMode,
                  arg_help="Method used to generate the graph inside the moving structures in the scene (i.e. a motion "
                           "proxy data structure that is used to store and play back the estimated surface motion).")
    pixel_anchor_computation_mode = \
        Parameter(default=AnchorComputationMode.PRECOMPUTED, arg_type=AnchorComputationMode,
                  arg_help="Method used to assign graph nodes as anchors to each pixel and compute their weights, which "
                           "control the influence of the graph on the estimated surface.")
    tracking_span_mode = \
        Parameter(default=TrackingSpanMode.ZERO_TO_T, arg_type=TrackingSpanMode,
                  arg_help="Interval over which to perform alignment for tracking objects. ZERO_TO_T mode will make "
                           "the program track between the first frame and each incoming sequence frame. "
                           "T_MINUS_ONE_TO_T will make it track between each consecutive pair of frames.")


class IntegrationParameters(ParameterEnum):
    anchor_node_count = \
        Parameter(default=4, arg_type=int,
                  arg_help="Number of nodes used as anchors for point on a surface.")
    fusion_minimum_valid_anchor_count = \
        Parameter(default=3, arg_type=int,
                  arg_help="TSDF voxels which have fewer than this number of valid anchors will not have any new data "
                           "fused in from an incoming RGBD image pair. Valid anchors for a specific voxel are graph "
                           "nodes that are closer than a specific distance threshold from this voxel.")
    transformation_mode = \
        Parameter(default=TransformationMode.MATRICES, arg_type=TrackingSpanMode,
                  arg_help="During fusion/integration, transformations (including rotations) can be handled in multiple "
                           "ways mathematically. Use this setting to dictate how.")

    voxel_anchor_computation_mode = \
        Parameter(default=AnchorComputationMode.EUCLIDEAN, arg_type=AnchorComputationMode,
                  arg_help="Stipulates how nodes are assigned as anchors to individual TSDF voxels. SHORTEST_PATH "
                           "mode will add the euclidean distance from the voxel to the nearest node to the shortest "
                           "path distance via this node to the target node.")


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


class VisualizationParameters(ParameterEnum):
    visualization_mode = \
        Parameter(default=VisualizationMode.NONE, arg_type=VisualizationMode,
                  arg_help="Controls extra visualization during the runtime of the fusion program.")


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


class TelemetryParameters(ParameterEnum):
    verbosity = VerbosityParameters
    visualization = VisualizationParameters
    logging = LoggingParameters


class FusionParameters(ParameterEnum):
    sequence_preset = \
        Parameter(default=FrameSequencePreset.BERLIN_50_SOD_MASKS, arg_type=FrameSequencePreset,
                  arg_help="Which sequence preset to use during the run.")
    tracking = TrackingParameters
    integration = IntegrationParameters
    telemetry = TelemetryParameters
