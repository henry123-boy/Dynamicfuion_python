from enum import Enum
import cv2
import numpy as np


class FrameImageType(Enum):
    COLOR = 1
    DEPTH = 2
    MASK = 3


def generate_frame_image_path(index: int, frame_image_type: FrameImageType, input_folder: str) -> str:
    filename_prefix_by_frame_image_type = {
        FrameImageType.COLOR: "color",
        FrameImageType.DEPTH: "depth",
        FrameImageType.MASK: "sod"
    }
    extension_by_frame_image_type = {
        FrameImageType.COLOR: "jpg",
        FrameImageType.DEPTH: "png",
        FrameImageType.MASK: "png"
    }
    return f"{input_folder:s}/{filename_prefix_by_frame_image_type[frame_image_type]:s}/{index:06d}.{extension_by_frame_image_type[frame_image_type]:s}"


def load_frame_numpy_raw_image(index: int, frame_image_type: FrameImageType, input_folder: str) -> np.ndarray:
    return cv2.imread(generate_frame_image_path(index, frame_image_type, input_folder), cv2.IMREAD_UNCHANGED)


def load_mask_numpy_image(index: int, input_folder: str) -> np.ndarray:
    return load_frame_numpy_raw_image(index, FrameImageType.MASK, input_folder)


def load_depth_numpy_image(index: int, input_folder: str, conversion_factor=0.001) -> np.ndarray:
    return load_frame_numpy_raw_image(index, FrameImageType.DEPTH, input_folder).astype(np.float32) * conversion_factor


def load_color_numpy_image(index: int, input_folder: str) -> np.ndarray:
    return load_frame_numpy_raw_image(index, FrameImageType.COLOR, input_folder)
