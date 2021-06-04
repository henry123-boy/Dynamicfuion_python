import typing
import os
from collections import Sequence
from enum import Enum
from data.frame import GenericDataset, DatasetType, DataSplit, SequenceFrameDataset


class FrameSequenceDataset(GenericDataset, typing.Sequence[SequenceFrameDataset]):

    def __init__(self, sequence_id: typing.Union[None, int] = None,
                 split: typing.Union[None, DataSplit] = None,
                 start_frame_index: int = 0,
                 frame_count: typing.Union[None, int] = None,
                 base_dataset_type: DatasetType = DatasetType.DEEP_DEFORM,
                 has_masks: bool = False,
                 segment_name: typing.Union[None, str] = None,
                 custom_frame_directory: typing.Union[None, str] = None,
                 masks_subfolder: typing.Union[None, str] = None):
        """
        Define a frame pair dataset.
        :param start_frame_index: 0-based index of the start frame
        :param frame_count: total number of frames to process.
        "None" means "go to highest  index possible"
        :param sequence_id: 0-based index of the sequence
        :param split: which data split to use
        :param base_dataset_type: determines the base directory of the dataset.
        DatasetType.DEEP_DEFORM will use options.base_dataset_dir,
        DatasetType.LOCAL will use "example_data" folder within the repo,
        DatasetType.CUSTOM requires custom_frame_folder to be set up
        :param has_masks: whether the dataset has or doesn't have masks.
        :param segment_name: specify segment name if graph is available for more than one segment in the dataset (e.g. Shirt0)
        (not necessary if there is only one segment)
        """
        super().__init__(sequence_id, split, base_dataset_type, has_masks, custom_frame_directory, masks_subfolder)
        self.graph_filename = None

        if self._base_dataset_type is not DatasetType.CUSTOM:
            # assumes add_in_graph_data.py script has already been successfully run on the sequence.
            # also assumes standard folder structure for DeepDeform & DeepDeformGraph datasets.
            graph_edges_dir = os.path.join(self.get_sequence_directory(), "graph_edges")
            parts = os.path.splitext(os.listdir(graph_edges_dir)[0])[0].split('_')
            if segment_name is None:
                segment_name = parts[1]
            if has_masks:
                mask_filename_mask_with_segment = os.path.join(self._mask_frame_directory, "{:06d}_" + segment_name + ".png")
                if os.path.isfile(mask_filename_mask_with_segment.format(0)):
                    self._mask_image_filename_mask = os.path.join(self._mask_frame_directory, "{:06d}_" + segment_name + ".png")
        self.segment_name = segment_name
        self.start_frame_index = start_frame_index

        if frame_count is None:
            frame_on_disk_count = 0
            depth_image_filename = self._depth_image_filename_mask.format(frame_on_disk_count)
            while os.path.isfile(depth_image_filename):
                frame_on_disk_count += 1
                depth_image_filename = self._depth_image_filename_mask.format(frame_on_disk_count)
            if start_frame_index >= frame_on_disk_count:
                raise ValueError(f"Specified sequence start, {start_frame_index:d}, is greater or equal than "
                                 f"the total count of frames found on disk {frame_on_disk_count:d}")
            self.frame_count = frame_on_disk_count - start_frame_index
        else:
            self.frame_count = frame_count

        self._next_frame_index = self.start_frame_index
        self._end_before_index = self.start_frame_index + self.frame_count

    def get_frame_at(self, index) -> SequenceFrameDataset:
        mask_image_path = None if not self._has_masks else self._mask_image_filename_mask.format(index)
        return SequenceFrameDataset(index,
                                    self._color_image_filename_mask.format(index),
                                    self._depth_image_filename_mask.format(index),
                                    mask_image_path)

    def get_next_frame(self) -> typing.Union[None, SequenceFrameDataset]:
        if self.has_more_frames():
            frame = self.get_frame_at(self._next_frame_index)
            self._next_frame_index += 1
            return frame

    def get_next_frame_index(self):
        return self._next_frame_index

    def has_more_frames(self):
        return self._next_frame_index < self._end_before_index

    def rewind(self):
        self._next_frame_index = self.start_frame_index

    def __len__(self):
        return self.frame_count

    def __repr__(self):
        return f"<{self.__class__.__name__:s}. Sequence directory: " \
               f"{self._sequence_directory:s}. From frame {self.start_frame_index}" \
               f" to frame {self._end_before_index - 1}. Next frame frame: {self._next_frame_index}>"

    # TODO: fix regular for-loop iteration for sequences: why does the iteration not respect len(sequence)??! how to
    #  make it start from start_frame_index, not 0, every time?

    def __getitem__(self, index):
        """Get a list item"""
        return self.get_frame_at(index)


class StaticFrameSequenceDataset(FrameSequenceDataset):
    def get_frame_at(self, index) -> SequenceFrameDataset:
        mask_image_path = None if not self._has_masks else self._mask_image_filename_mask.format(0)
        return SequenceFrameDataset(index,
                                    self._color_image_filename_mask.format(0),
                                    self._depth_image_filename_mask.format(0),
                                    mask_image_path)
