import typing
import os

import open3d as o3d
import open3d.core as o3c

from data import DeformDataset, StaticCenterCrop
from data.frame import GenericDataset, DatasetType, DataSplit, SequenceFrameDataset
from nnrt.geometry import GraphWarpField


class FrameSequenceDataset(GenericDataset, typing.Sequence[SequenceFrameDataset]):

    def __init__(self, sequence_id: typing.Union[None, int] = None,
                 split: typing.Union[None, DataSplit] = None,
                 start_frame_index: int = 0,
                 frame_count: typing.Union[None, int] = None,
                 base_dataset_type: DatasetType = DatasetType.DEEP_DEFORM,
                 has_masks: bool = False,
                 segment_name: typing.Union[None, str] = None,
                 custom_frame_directory: typing.Union[None, str] = None,
                 masks_subfolder: typing.Union[None, str] = None,
                 far_clipping_distance: float = 0.0,
                 mask_lower_threshold: int = 250):
        """
        Define a frame pair dataset.
        :param start_frame_index: 0-based index of the start frame
        :param frame_count: total number of frames to process.
        "None" means "go to highest  index possible"
        :param sequence_id: 0-based index of the sequence
        :param split: which data split to use
        :param base_dataset_type: determines the base directory of the dataset.
        DatasetType.DEEP_DEFORM will use options.base_dataset_dir and assume Deep Deform + graph data database structure,
         DatasetType.LOCAL will use "example_data" folder within the repo and assume Deep Deform + graph data database structure,
         DatasetType.CUSTOM requires custom_frame_folder to be set to the root of the sequence on disk.
        :param has_masks: whether the dataset has or doesn't have masks.
        :param segment_name: specify segment name if graph is available for more than one segment in the dataset (e.g. Shirt0)
        (not necessary if there is only one segment)
        :param custom_frame_directory: Root of the sequence on disk for CUSTOM base_dataset_type
        :param masks_subfolder: an optional subfolder where to look for masks ("masks" is the default assumed).
        Used for any dataset type.
        :param far_clipping_distance: used for clipping depth pixels when reading the depth sequence. Set in meters.
        Set to 0 or a negative value for no far clipping.
        """
        super().__init__(sequence_id, split, base_dataset_type, has_masks, custom_frame_directory, masks_subfolder,
                         far_clipping_distance, mask_lower_threshold)
        self.segment_name = segment_name
        self.start_frame_index = start_frame_index
        self.frame_count = frame_count
        self._next_frame_index = None
        self._end_before_index = None
        self._resolution = None

    def load(self):
        super(FrameSequenceDataset, self).load()
        if self._base_dataset_type is not DatasetType.CUSTOM:
            # assumes add_in_graph_data.py script has already been successfully run on the sequence.
            # also assumes standard folder structure for DeepDeform & DeepDeformGraph datasets.
            graph_edges_dir = os.path.join(self.get_sequence_directory(), "graph_edges")
            parts = os.path.splitext(os.listdir(graph_edges_dir)[0])[0].split('_')
            if self.segment_name is None:
                segment_name = parts[1]
            if self.has_masks():
                mask_filename_mask_with_segment = os.path.join(self._mask_frame_directory,
                                                               "{:06d}_" + segment_name + ".png")
                if os.path.isfile(mask_filename_mask_with_segment.format(0)):
                    self._mask_image_filename_mask = os.path.join(self._mask_frame_directory,
                                                                  "{:06d}_" + segment_name + ".png")

        if self.frame_count is None:
            frame_on_disk_count = 0
            depth_image_filename = self._depth_image_filename_mask.format(frame_on_disk_count)
            while os.path.isfile(depth_image_filename):
                frame_on_disk_count += 1
                depth_image_filename = self._depth_image_filename_mask.format(frame_on_disk_count)
            if self.start_frame_index >= frame_on_disk_count:
                raise ValueError(f"Specified sequence start, {self.start_frame_index:d}, is greater or equal than "
                                 f"the total count of frames found on disk {frame_on_disk_count:d}")
            self.frame_count = frame_on_disk_count - self.start_frame_index

        self._next_frame_index = self.start_frame_index
        self._end_before_index = self.start_frame_index + self.frame_count

        first_frame_image = o3d.io.read_image(self._color_image_filename_mask.format(self.start_frame_index))
        image_dims = first_frame_image.get_max_bound()  # array([ width, height])
        self._resolution = (int(image_dims[1]), int(image_dims[0]))  # (height, width)

    def get_frame_at(self, index) -> SequenceFrameDataset:
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")

        mask_image_path = None if not self._has_masks else self._mask_image_filename_mask.format(index)

        return SequenceFrameDataset(index,
                                    self._color_image_filename_mask.format(index),
                                    self._depth_image_filename_mask.format(index),
                                    mask_image_path)

    def get_next_frame(self) -> typing.Union[None, SequenceFrameDataset]:
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")

        if self.has_more_frames():
            frame = self.get_frame_at(self._next_frame_index)
            self._next_frame_index += 1
            return frame

    def advance_to_frame(self, index):
        if index < self.start_frame_index or index >= self._end_before_index:
            raise ValueError(f"Provided index not within frame range, {self.start_frame_index, self._end_before_index}")
        else:
            self._next_frame_index = index

    def get_current_graph_name(self):
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")

        graph_edges_dir = os.path.join(self.get_sequence_directory(), "graph_edges")
        graph_filename = None
        current_frame_index = self._next_frame_index - 1
        for filename in os.listdir(graph_edges_dir):
            filename_sans_ext = os.path.splitext(filename)[0]
            parts = filename_sans_ext.split('_')
            start_frame = int(parts[2])
            if start_frame == current_frame_index:
                graph_filename = filename_sans_ext
                break
        return graph_filename

    def get_current_pixel_anchors_and_weights(self, cropper: typing.Union[None, StaticCenterCrop] = None):
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")

        graph_filename = self.get_current_graph_name()
        if graph_filename is None:
            return None, None

        pixel_anchors, pixel_weights = \
            DeformDataset.load_anchors_and_weights_from_sequence_directory_and_graph_filename(
                self.get_sequence_directory(), graph_filename, cropper
            )

        return pixel_anchors, pixel_weights

    def get_current_frame_graph_warp_field(self, device: o3c.Device) -> typing.Union[None, GraphWarpField]:
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")

        if self._base_dataset_type == DatasetType.CUSTOM:
            return None

        # TODO: make flexible-enough to load even when graph data is not available and signify this by storing some
        #  value in a field

        graph_filename = self.get_current_graph_name()
        if graph_filename is None:
            return None

        graph_nodes, graph_edges, graph_edge_weights, _, graph_clusters = \
            DeformDataset.load_graph_data(self.get_sequence_directory(), graph_filename, False)

        nodes_o3d = o3c.Tensor(graph_nodes, device=device)
        edges_o3d = o3c.Tensor(graph_edges, device=device)
        edge_weights_o3d = o3c.Tensor(graph_edge_weights, device=device)
        clusters_o3d = o3c.Tensor(graph_clusters.flatten(), device=device)
        graph = GraphWarpField(nodes_o3d, edges_o3d, edge_weights_o3d, clusters_o3d)

        return graph

    def get_next_frame_index(self):
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")

        return self._next_frame_index

    def has_more_frames(self):
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")

        return self._next_frame_index < self._end_before_index

    def rewind(self):
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")
        self._next_frame_index = self.start_frame_index

    @property
    def resolution(self):
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")
        return self._resolution

    def __len__(self):
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")
        return self.frame_count

    def __repr__(self):
        return f"<{self.__class__.__name__:s}. Loaded: {self._loaded}. Sequence directory: " \
               f"{self._sequence_directory:s}. From frame {self.start_frame_index}" \
               f" to frame {self._end_before_index - 1}. Next frame frame: {self._next_frame_index}>"

    # TODO: fix regular for-loop iteration for sequences: why does the iteration not respect len(sequence)??! how to
    #  make it start from start_frame_index, not 0, every time?

    def __getitem__(self, index):
        """Get a list item"""
        return self.get_frame_at(index)


class StaticFrameSequenceDataset(FrameSequenceDataset):
    def get_frame_at(self, index) -> SequenceFrameDataset:
        if not self._loaded:
            raise ValueError("Before a dataset can be used, it has to be loaded with the .load() method.")

        mask_image_path = None if not self._has_masks else self._mask_image_filename_mask.format(0)
        return SequenceFrameDataset(index,
                                    self._color_image_filename_mask.format(0),
                                    self._depth_image_filename_mask.format(0),
                                    mask_image_path)
