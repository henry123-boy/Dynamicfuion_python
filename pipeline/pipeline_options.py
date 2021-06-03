from enum import Enum
import options
from data import FrameSequenceDataset, FrameSequencePreset


class VisualizationMode(Enum):
    NONE = 0
    CANONCIAL_MESH = 1
    WARPED_MESH = 2
    POINT_CLOUD_TRACKING = 3


class SourceImageMode(Enum):
    REUSE_DATASET = 0
    RENDERED_ONLY = 1
    RENDERD_WITH_PREVIOUS_IMAGE_OVERLAY = 2


# We will overwrite the default value in options.py / settings.py
options.use_mask = True
options.gn_max_nodes = 3000

# TODO: the below settings should be part of dataset class tree!
far_clip_distance = 2400  # mm, non-positive value for no clipping
input_image_size = (480, 640)
mask_clip_lower_threshold = 150

# **** TELEMETRY *****

# verbosity options
print_frame_info = True
print_intrinsics = False
print_gpu_memory_info = False
print_voxel_fusion_statistics = False

# visualization options
visualization_mode: VisualizationMode = VisualizationMode.WARPED_MESH

# logging options
record_visualization_to_disk = False

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

anchor_node_count = 4
source_image_mode: SourceImageMode = SourceImageMode.REUSE_DATASET