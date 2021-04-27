from enum import Enum
import typing
import os
import re

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
    name_mask = name[:span[0]] + "{:0" + str(span[1] - span[0]) + "d}" + extension
    return name_mask


class GenericDataset:
    def __init__(self, sequence_id: typing.Union[None, int] = None, split: typing.Union[None, DataSplit] = None,
                 base_dataset_type: DatasetType = DatasetType.DEEP_DEFORM,
                 has_masks: bool = False, custom_frame_directory: typing.Union[None, str] = None):
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
            for filename in filenames:
                parts = os.path.splitext(filename)
                name = parts[0]
                extension = parts[1]
                if not color_frame_found and extension in [".png", ".jpg", ".jpeg"] and ("color" in name or "rgb" in name):
                    color_frame_found = True
                    name_mask = make_frame_file_name_mask(name, extension)
                    self._color_image_filename_mask = os.path.join(self._base_data_directory, name_mask)
                elif not depth_frame_found and extension == ".png" and "depth" in name:
                    depth_frame_found = True
                    name_mask = make_frame_file_name_mask(name, extension)
                    self._depth_image_filename_mask = os.path.join(self._base_data_directory, name_mask)
                elif not mask_frame_found and has_masks and extension == ".png" and "mask" in name:
                    mask_frame_found = True
                    name_mask = make_frame_file_name_mask(name, extension)
                    self._mask_image_filename_mask = os.path.join(self._base_data_directory, name_mask)
                elif not intrinsics_file_found and extension == ".txt" and "ntrinsics" in name:
                    intrinsics_file_found = True
                    self._intrinsics_file_path = os.path.join(self._base_data_directory, filename)
                else:
                    break

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
                self._mask_frame_directory = os.path.join(self._sequence_directory, "mask")
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


class StandaloneFrameDataset(GenericDataset, FrameDataset):
    def __init__(self, frame_index: int,
                 sequence_id: typing.Union[None, int] = None,
                 split: typing.Union[None, DataSplit] = None,
                 base_dataset_type: DatasetType = DatasetType.DEEP_DEFORM,
                 has_masks: bool = False, custom_frame_directory: typing.Union[None, str] = None):
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
        super().__init__(sequence_id, split, base_dataset_type, has_masks, custom_frame_directory)
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
