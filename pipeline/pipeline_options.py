from enum import Enum
import sys
import options
from data import FrameSequenceDataset, FrameSequencePreset


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
    FIRST_FRAME_LOADED_GRAPH = 1


class AnchorComputationMode(Enum):
    EUCLIDEAN = 0
    SHORTEST_PATH = 1


# **** BEHAVIOR *****

# Tracking
source_image_mode: SourceImageMode = SourceImageMode.REUSE_PREVIOUS_FRAME
# We will overwrite the default value in options.py / settings.py
options.use_mask = True
options.gn_max_nodes = 3000
graph_generation_mode = GraphGenerationMode.FIRST_FRAME_LOADED_GRAPH
pixel_anchor_computation_mode = AnchorComputationMode.EUCLIDEAN

# Integration
anchor_node_count = 4  # used for initial graph generation, mesh warping, and integration
fusion_minimum_valid_anchor_count = 3
# TODO setting to tune maximum invalid node count
transformation_mode = TransformationMode.MATRICES
voxel_anchor_computation_mode = AnchorComputationMode.EUCLIDEAN

# **** TELEMETRY *****

# verbosity options
print_frame_info = True
print_intrinsics = False
print_cuda_memory_info = False

# visualization options
visualization_mode: VisualizationMode = VisualizationMode.NONE

# logging options
record_visualization_to_disk = False
record_canonical_meshes_to_disk = True
record_warped_meshes_to_disk = True
record_rendered_warped_mesh = False
record_gn_point_clouds = True

# **** DATASET *****

# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_50.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_STATIC.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_ROTATION_Z.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_OFFSET_X.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_OFFSET_XY.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_ROTATION_Z.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_STRETCH_Y.value
# sequence: FrameSequenceDataset = FrameSequencePreset.RED_SHORTS_40.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_SOD_MASKS.value
sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_50_SOD_MASKS.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_3_SOD_MASKS.value


def print_pipeline_options(stdout=sys.stdout):
    original_stdout = sys.stdout
    sys.stdout = stdout
    print("PIPELINE OPTIONS (pipeline_options.py):")

    # **** BEHAVIOR *****

    # Tracking
    print("source_image_mode:", source_image_mode)
    # overridden from options
    print("use_mask (overridden from options.py):", options.use_mask)
    print("gn_max_nodes (overridden from options.py):", options.gn_max_nodes)
    print("graph_generation_mode:", graph_generation_mode)
    print("pixel_anchor_computation_mode:", pixel_anchor_computation_mode)

    # Integration
    print("anchor_node_count", anchor_node_count)
    print("fusion_minimum_valid_anchor_count", fusion_minimum_valid_anchor_count)
    print("transformation_mode", transformation_mode)
    print("voxel_anchor_computation_mode", voxel_anchor_computation_mode)

    # **** TELEMETRY *****

    # verbosity options
    print("print_frame_info", print_frame_info)
    print("print_intrinsics", print_intrinsics)
    print("print_cuda_memory_info", print_cuda_memory_info)

    # visualization options
    print("visualization_mode", visualization_mode)

    # logging options
    print("record_visualization_to_disk", record_visualization_to_disk)
    print("record_canonical_meshes_to_disk", record_canonical_meshes_to_disk)
    print("record_warped_meshes_to_disk", record_warped_meshes_to_disk)
    print("record_rendered_warped_mesh", record_rendered_warped_mesh)
    print("record_gn_point_clouds", record_gn_point_clouds)

    # **** DATASET *****
    print("sequence id:", sequence.sequence_id)
    print("sequence segment:", sequence.segment_name)
    print("sequence split:", sequence.split)
    print("sequence folder:", sequence.get_sequence_directory())
    print("sequence range:", sequence.start_frame_index, "to", sequence.start_frame_index + len(sequence))

    sys.stdout = original_stdout


def save_pipeline_options_to_file(file_path: str) -> None:
    with open(file_path, 'w') as file:
        print_pipeline_options(file)
