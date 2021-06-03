from enum import Enum
import options


class VisualizationMode(Enum):
    NONE = 0
    CANONCIAL_MESH = 1
    WARPED_MESH = 2
    POINT_CLOUD_TRACKING = 3


# We will overwrite the default value in options.py / settings.py
options.use_mask = True
options.gn_max_nodes = 3000

# TODO: the below settings should be part of dataset class tree!
far_clip_distance = 2400  # mm, non-positive value for no clipping
input_image_size = (480, 640)
mask_clip_lower_threshold = 50

# **** TELEMETRY *****

# verbosity options
print_frame_info = True
print_intrinsics = False
print_gpu_memory_info = False
print_voxel_fusion_statistics = False

# visualization options
visualization_mode: VisualizationMode = VisualizationMode.POINT_CLOUD_TRACKING

# logging options
record_visualization_to_disk = False
