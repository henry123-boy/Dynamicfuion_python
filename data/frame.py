from enum import Enum
import typing
import os
import re

import cv2
import numpy as np
import open3d as o3d

import options
from abc import ABC, abstractmethod, ABCMeta


class DataSplit(Enum):
    TEST = "test"
    VALIDATION = "val"
    TRAIN = "train"


class DatasetType(Enum):
    DEEP_DEFORM = 0
    LOCAL = 1
    CUSTOM = 2


def make_frame_file_name_mask(name: str, extension: str) -> str:
    """
    Compile a mask based on specified frame file name and extension
    :param name: name of the file (excluding the extension), e.g. if the whole file is "frame-000000.color.png",
    the name is "frame-000000.color". The function expects the name to have the index as the ONLY continuous,
    0-padded digit within the name.
    :param extension: extension, e.g. ".png" or ".jpg"
    :return:
    """
    search_result = re.search(r"\d+", name)
    span = search_result.span()
    name_mask = name[:span[0]] + "{:0" + str(span[1] - span[0]) + "d}" + name[span[1]:] + extension
    return name_mask


class GenericDataset:
    def __init__(self, sequence_id: typing.Union[None, int] = None, split: typing.Union[None, DataSplit] = None,
                 base_dataset_type: DatasetType = DatasetType.DEEP_DEFORM,
                 has_masks: bool = False, custom_frame_directory: typing.Union[None, str] = None,
                 masks_subfolder: typing.Union[None, str] = None):
        """
        Define a single-frame dataset.
        :param sequence_id: 0-based index of the sequence
        :param split: which data split to use
        :param base_dataset_type: determines the base directory of the dataset.
         DatasetType.DEEP_DEFORM will use options.base_dataset_dir,
         DatasetType.LOCAL will use "example_data" folder within the repo,
         DatasetType.CUSTOM requires custom_frame_folder to be set up
        :param has_masks: whether the dataset has or doesn't have masks.
        """
        self.sequence_id = sequence_id
        self.split = split
        self._base_dataset_type = base_dataset_type
        folder_switcher = {
            DatasetType.DEEP_DEFORM: options.dataset_base_directory,
            DatasetType.LOCAL: "example_data",
            DatasetType.CUSTOM: custom_frame_directory
        }
        self._base_data_directory = folder_switcher[base_dataset_type]
        self._has_masks = has_masks
        if self._base_data_directory is None:
            raise ValueError("For base_dataset_type == DatasetType.CUSTOM, custom_base_data_folder should point to the frame data location")

        self._mask_image_filename_mask = None
        self._mask_frame_directory = None

        if base_dataset_type == DatasetType.CUSTOM:
            self._sequence_directory = self._base_data_directory
            self._color_frame_directory = self._base_data_directory
            self._depth_frame_directory = self._base_data_directory
            if has_masks:
                self._mask_frame_directory = self._base_data_directory

            # attempt to infer color, depth, and mask image paths from file names and extensions
            filenames = os.listdir(self._base_data_directory)
            color_frame_found = False
            depth_frame_found = False
            mask_frame_found = False
            intrinsics_file_found = False
            self._color_image_filename_mask = None
            self._depth_image_filename_mask = None
            self._mask_image_filename_mask = None

            def looks_like_color_image(_name: str, _extension: str) -> bool:
                return _extension in [".png", ".jpg", ".jpeg"] and ("color" in _name or "rgb" in _name)

            def looks_like_depth_image(_name: str, _extension: str) -> bool:
                return _extension == ".png" and "depth" in _name

            def looks_like_mask_image(_name: str, _extension: str) -> bool:
                return _extension == ".png" and "mask" in _name

            for filename in filenames:
                parts = os.path.splitext(filename)
                name = parts[0]
                extension = parts[1]
                if not color_frame_found and looks_like_color_image(name, extension):
                    color_frame_found = True
                    name_mask = make_frame_file_name_mask(name, extension)
                    self._color_image_filename_mask = os.path.join(self._base_data_directory, name_mask)
                elif not depth_frame_found and looks_like_depth_image(name, extension):
                    depth_frame_found = True
                    name_mask = make_frame_file_name_mask(name, extension)
                    self._depth_image_filename_mask = os.path.join(self._base_data_directory, name_mask)
                elif not mask_frame_found and has_masks and looks_like_mask_image(name, extension):
                    mask_frame_found = True
                    name_mask = make_frame_file_name_mask(name, extension)
                    self._mask_image_filename_mask = os.path.join(self._base_data_directory, name_mask)
                elif not intrinsics_file_found and extension == ".txt" and "ntrinsics" in name:
                    intrinsics_file_found = True
                    self._intrinsics_file_path = os.path.join(self._base_data_directory, filename)

            def search_subfolders_for_image_data(potential_subfolders: typing.List[str], expected_extensions: typing.List[str]) -> typing.Union[
                None, str]:
                four_digit_pattern = re.compile(r"\d{4}")
                for subfolder in potential_subfolders:
                    potential_directory = os.path.join(self._base_data_directory, subfolder)
                    if os.path.isdir(potential_directory):
                        _filenames = os.listdir(potential_directory)
                        for _filename in _filenames:
                            _parts = os.path.splitext(_filename)
                            _name = _parts[0]
                            _extension = _parts[1]
                            if _extension in expected_extensions and re.match(four_digit_pattern, _name) is not None:
                                _name_mask = make_frame_file_name_mask(_name, _extension)
                                return os.path.join(potential_directory, _name_mask)

            if self._depth_image_filename_mask is None:
                potential_depth_subfolders = ["depth", "depth_images", "depth_frames"]
                self._depth_image_filename_mask = search_subfolders_for_image_data(potential_depth_subfolders, [".png"])
                if self._depth_image_filename_mask is None:
                    raise ValueError(f"Could not find any depth frame data in {self._base_data_directory}")

            if self._color_image_filename_mask is None:
                potential_color_subfolders = ["color", "color_images", "color_frames"]
                self._color_image_filename_mask = search_subfolders_for_image_data(potential_color_subfolders, [".jpg", ".png"])
                if self._color_image_filename_mask is None:
                    raise ValueError(f"Could not find any color frame data in {self._base_data_directory}")

            if has_masks and self._mask_image_filename_mask is None:
                if masks_subfolder is None:
                    potential_mask_subfolders = ["mask", "masks", "mask_images", "omask"]
                else:
                    potential_mask_subfolders = [masks_subfolder]
                self._mask_image_filename_mask = search_subfolders_for_image_data(potential_mask_subfolders, [".png"])
                if self._mask_image_filename_mask is None:
                    raise ValueError(f"Could not find any mask frame data in {self._base_data_directory}")

        else:
            if sequence_id is None or split is None:
                raise ValueError(f"A dataset of type DatasetType.DEEP_DEFORM requires an integer sequence_id "
                                 f"and split of type DataSplit. Got sequence id {str(sequence_id)} and split {str(split)}.")
            self._sequence_directory = os.path.join(self._base_data_directory, "{:s}/seq{:03d}".format(self.split.value, self.sequence_id))
            self._color_frame_directory = os.path.join(self._sequence_directory, "color")
            self._depth_frame_directory = os.path.join(self._sequence_directory, "depth")
            self._color_image_filename_mask = os.path.join(self._color_frame_directory, "{:06d}.jpg")
            self._depth_image_filename_mask = os.path.join(self._depth_frame_directory, "{:06d}.png")
            if has_masks:
                if masks_subfolder is None:
                    self._mask_frame_directory = os.path.join(self._sequence_directory, "mask")
                else:
                    self._mask_frame_directory = os.path.join(self._sequence_directory, masks_subfolder)
                first_filename = os.listdir(self._mask_frame_directory)[0]
                parts = os.path.splitext(first_filename)
                name = parts[0]
                extension = parts[1]
                name_mask = make_frame_file_name_mask(name, extension)
                self._mask_image_filename_mask = os.path.join(self._mask_frame_directory, name_mask)
            self._intrinsics_file_path = os.path.join(self._sequence_directory, "intrinsics.txt")

    def get_sequence_directory(self) -> str:
        return self._sequence_directory

    def get_color_frame_directory(self) -> str:
        return self._color_frame_directory

    def get_depth_frame_directory(self) -> str:
        return self._depth_frame_directory

    def get_mask_frame_directory(self) -> str:
        return self._mask_frame_directory

    def get_intrinsics_path(self) -> str:
        return self._intrinsics_file_path


class FrameDataset(metaclass=ABCMeta):
    @abstractmethod
    def get_color_image_path(self) -> str:
        pass

    @abstractmethod
    def get_depth_image_path(self) -> str:
        pass

    @abstractmethod
    def get_mask_image_path(self) -> str:
        pass

    def load_color_image_numpy(self) -> np.ndarray:
        return cv2.imread(self.get_color_image_path(), cv2.IMREAD_UNCHANGED)

    def load_depth_image_numpy(self) -> np.ndarray:
        return cv2.imread(self.get_depth_image_path(), cv2.IMREAD_UNCHANGED)

    def load_mask_image_numpy(self) -> np.ndarray:
        return cv2.imread(self.get_mask_image_path(), cv2.IMREAD_UNCHANGED)

    def load_color_image_open3d(self, device: o3d.core.Device) -> o3d.t.geometry.Image:
        return o3d.t.geometry.Image.from_legacy_image(o3d.io.read_image(self.get_color_image_path()), device=device)

    def load_depth_image_open3d(self, device: o3d.core.Device) -> o3d.t.geometry.Image:
        return o3d.t.geometry.Image.from_legacy_image(o3d.io.read_image(self.get_depth_image_path()), device=device)

    def load_mask_image_open3d(self, device: o3d.core.Device) -> o3d.t.geometry.Image:
        return o3d.t.geometry.Image.from_legacy_image(o3d.io.read_image(self.get_mask_image_path()), device=device)


class StandaloneFrameDataset(FrameDataset, GenericDataset):
    def __init__(self, frame_index: int,
                 sequence_id: typing.Union[None, int] = None,
                 split: typing.Union[None, DataSplit] = None,
                 base_dataset_type: DatasetType = DatasetType.DEEP_DEFORM,
                 has_masks: bool = False, custom_frame_directory: typing.Union[None, str] = None,
                 masks_subfolder: typing.Union[None, str] = None):
        """
        Define a single-frame dataset.
        :param frame_index: 0-based index of the frame
        :param sequence_id: 0-based index of the sequence
        :param split: which data split to use
        :param base_dataset_type: determines the base directory of the dataset.
         DatasetType.DEEP_DEFORM will use options.base_dataset_dir,
         DatasetType.LOCAL will use "example_data" folder within the repo,
         DatasetType.CUSTOM requires custom_frame_folder to be set up
        :param has_masks: whether the dataset has or doesn't have masks.
        """
        super(StandaloneFrameDataset, self).__init__(sequence_id, split, base_dataset_type, has_masks,
                                                     custom_frame_directory, masks_subfolder)
        self.frame_index = frame_index

    def get_color_image_path(self) -> str:
        return self._color_image_filename_mask.format(self.frame_index)

    def get_depth_image_path(self) -> str:
        return self._depth_image_filename_mask.format(self.frame_index)

    def get_mask_image_path(self) -> str:
        if self._has_masks:
            return self._mask_image_filename_mask.format(self.frame_index)
        else:
            raise ValueError("Trying to retrieve mask path, but the current dataset is defined to have no masks!")


class SequenceFrameDataset(FrameDataset):
    def __init__(self, frame_index: int, color_frame_path: str, depth_frame_path: str, mask_frame_path: typing.Union[None, str] = None):
        self.frame_index = frame_index
        self.color_image_path = color_frame_path
        self.depth_image_path = depth_frame_path
        self.mask_image_path = mask_frame_path

    def get_color_image_path(self) -> str:
        return self.color_image_path

    def get_depth_image_path(self) -> str:
        return self.depth_image_path

    def get_mask_image_path(self) -> str:
        if self.mask_image_path is not None:
            return self.mask_image_path
        else:
            raise ValueError("Trying to retrieve mask path, but the current dataset is defined to have no masks!")
