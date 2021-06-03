from enum import Enum
import options
from data import FrameSequenceDataset, FrameSequencePreset


class VisualizationMode(Enum):
    NONE = 0
    CANONICAL_MESH = 1
    WARPED_MESH = 2
    POINT_CLOUD_TRACKING = 3


class SourceImageMode(Enum):
    REUSE_PREVIOUS_FRAME = 0
    RENDERED_ONLY = 1
    RENDERED_WITH_PREVIOUS_FRAME_OVERLAY = 2


class TransformationMode(Enum):
    QUATERNIONS = 0
    MATRICES = 1


# **** BEHAVIOR *****

# Preprocessing
# TODO: the below settings should be part of dataset class tree!
far_clip_distance = 2400  # mm, non-positive value for no clipping
input_image_size = (480, 640)
mask_clip_lower_threshold = 150

# Tracking
source_image_mode: SourceImageMode = SourceImageMode.REUSE_PREVIOUS_FRAME
# We will overwrite the default value in options.py / settings.py
options.use_mask = True
options.gn_max_nodes = 3000

# Integration
anchor_node_count = 4  # also used for initial graph generation
transformation_mode = TransformationMode.MATRICES


# **** TELEMETRY *****

# verbosity options
print_frame_info = True
print_intrinsics = False
print_gpu_memory_info = False
print_voxel_fusion_statistics = True

# visualization options
visualization_mode: VisualizationMode = VisualizationMode.WARPED_MESH

# logging options
record_visualization_to_disk = False

# **** DATASET *****

# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_50.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_STATIC.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_ROTATION_Z.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_OFFSET_X.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_OFFSET_XY.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_ROTATION_Z.value
# sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_STRETCH_Y.value
# sequence: FrameSequenceDataset = FrameSequencePreset.RED_SHORTS_40.value
# sequence: FrameSequenceDataset = FrameSequencePreset.RED_SHORTS_40_SOD_MASKS.value
sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_SOD_MASKS.value
