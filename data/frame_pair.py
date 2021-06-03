import typing
import os
from enum import Enum
from data.frame import GenericDataset, DatasetType, DataSplit


class FramePairDataset(GenericDataset):

    def __init__(self, source_frame_index: int, target_frame_index: int,
                 sequence_id: typing.Union[None, int] = None,
                 split: typing.Union[None, DataSplit] = None,
                 base_dataset_type: DatasetType = DatasetType.DEEP_DEFORM,
                 has_masks: bool = False,
                 segment_name: typing.Union[None, str] = None,
                 custom_frame_directory: typing.Union[None, str] = None,
                 masks_subfolder: typing.Union[None, str] = None):
        """
        Define a frame pair dataset.
        :param source_frame_index: 0-based index of the source frame
        :param target_frame_index: 0-based index of the target frame
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
            # TODO: make flexible-enough to load even when graph data is not available and signify this by storing some
            #  value in a field
            parts = os.path.splitext(os.listdir(graph_edges_dir)[0])[0].split('_')
            if segment_name is None:
                segment_name = parts[1]
            if has_masks:
                self._mask_image_filename_mask = os.path.join(self._mask_frame_directory, "{:06d}_" + segment_name + ".png")
            self.graph_filename = f"{parts[0]:s}_{segment_name}_{source_frame_index:06d}_{target_frame_index:06d}_geodesic_0.05"

        self.source_frame_index = source_frame_index
        self.target_frame_index = target_frame_index
        self.segment_name = segment_name

    def get_source_color_image_path(self) -> str:
        return self._color_image_filename_mask.format(self.source_frame_index)

    def get_target_color_image_path(self) -> str:
        return self._color_image_filename_mask.format(self.target_frame_index)

    def get_source_depth_image_path(self) -> str:
        return self._depth_image_filename_mask.format(self.source_frame_index)

    def get_target_depth_image_path(self) -> str:
        return self._depth_image_filename_mask.format(self.target_frame_index)

    def get_source_mask_image_path(self) -> str:
        if self._has_masks:
            return self._mask_image_filename_mask.format(self.source_frame_index)
        else:
            raise ValueError("Trying to retrieve mask path, but the current dataset is defined to have no masks!")

    def get_target_mask_image_path(self) -> str:
        if self._has_masks:
            return self._mask_image_filename_mask.format(self.target_frame_index)
        else:
            raise ValueError("Trying to retrieve mask path, but the current dataset is defined to have no masks!")
        




