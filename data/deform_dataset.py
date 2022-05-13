import os
import json
import typing

from torch.utils.data import Dataset
import torch
import numpy as np
import skimage.io
from multipledispatch import dispatch

from data.io import load_flow, load_graph_nodes_or_deformations, load_graph_edges, load_graph_edges_weights, \
    load_graph_clusters, load_int_image, load_float_image
import image_processing

from data.cropping import StaticCenterCrop


class DeformDataset(Dataset):
    def __init__(self, dataset_base_dir, labels_filename,
                 input_width, input_height, max_boundary_distance):
        self.dataset_base_dir = dataset_base_dir
        self.labels_path = os.path.join(self.dataset_base_dir, labels_filename + ".json")

        self.input_width = input_width
        self.input_height = input_height

        self.max_boundary_distance = max_boundary_distance

        self.cropper = None

        self._load()

    def _load(self):
        with open(self.labels_path) as f:
            self.labels = json.loads(f.read())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.labels[index]

        src_color_image_path = os.path.join(self.dataset_base_dir, data["source_color"])
        src_depth_image_path = os.path.join(self.dataset_base_dir, data["source_depth"])
        tgt_color_image_path = os.path.join(self.dataset_base_dir, data["target_color"])
        tgt_depth_image_path = os.path.join(self.dataset_base_dir, data["target_depth"])
        graph_nodes_path = os.path.join(self.dataset_base_dir, data["graph_nodes"])
        graph_edges_path = os.path.join(self.dataset_base_dir, data["graph_edges"])
        graph_edges_weights_path = os.path.join(self.dataset_base_dir, data["graph_edges_weights"])
        graph_node_deformations_path = os.path.join(self.dataset_base_dir, data["graph_node_deformations"])
        graph_clusters_path = os.path.join(self.dataset_base_dir, data["graph_clusters"])
        pixel_anchors_path = os.path.join(self.dataset_base_dir, data["pixel_anchors"])
        pixel_weights_path = os.path.join(self.dataset_base_dir, data["pixel_weights"])
        optical_flow_image_path = os.path.join(self.dataset_base_dir, data["optical_flow"])
        scene_flow_image_path = os.path.join(self.dataset_base_dir, data["scene_flow"])

        # Load source, target image and flow.
        source, _, cropper = DeformDataset.load_image(
            src_color_image_path, src_depth_image_path, data["intrinsics"], self.input_height, self.input_width
        )
        target, target_boundary_mask, _ = DeformDataset.load_image(
            tgt_color_image_path, tgt_depth_image_path, data["intrinsics"], self.input_height, self.input_width, cropper=cropper,
            max_boundary_distance=self.max_boundary_distance, compute_boundary_mask=True
        )

        optical_flow_gt, optical_flow_mask, scene_flow_gt, scene_flow_mask = DeformDataset.load_flow(
            optical_flow_image_path, scene_flow_image_path, cropper
        )

        # Load/compute graph.
        graph_nodes, graph_edges, graph_edges_weights, graph_node_deformations, graph_clusters = \
            DeformDataset.load_graph_data(
                graph_nodes_path, graph_edges_path, graph_edges_weights_path, graph_node_deformations_path, graph_clusters_path
            )
        pixel_anchors, pixel_weights = DeformDataset.load_anchors_and_weights(pixel_anchors_path, pixel_weights_path, cropper)

        # Compute groundtruth transformation for graph canonical_node_positions.
        num_nodes = graph_nodes.shape[0]

        # Check that flow mask is valid for at least one pixel.
        assert np.sum(optical_flow_mask) > 0, "Zero flow mask for sample: " + json.dumps(data)

        # Store intrinsics.
        fx = data["intrinsics"]["fx"]
        fy = data["intrinsics"]["fy"]
        cx = data["intrinsics"]["cx"]
        cy = data["intrinsics"]["cy"]

        fx, fy, cx, cy = image_processing.modify_intrinsics_due_to_cropping(
            fx, fy, cx, cy, self.input_height, self.input_width, original_h=480, original_w=640
        )

        intrinsics = np.zeros(4, dtype=np.float32)
        intrinsics[0] = fx
        intrinsics[1] = fy
        intrinsics[2] = cx
        intrinsics[3] = cy

        return {
            "source": source,
            "target": target,
            "target_boundary_mask": target_boundary_mask,
            "optical_flow_gt": optical_flow_gt,
            "optical_flow_mask": optical_flow_mask,
            "scene_flow_gt": scene_flow_gt,
            "scene_flow_mask": scene_flow_mask,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "graph_edges_weights": graph_edges_weights,
            "graph_node_deformations": graph_node_deformations,
            "graph_clusters": graph_clusters,
            "pixel_anchors": pixel_anchors,
            "pixel_weights": pixel_weights,
            "num_nodes": np.array(num_nodes, dtype=np.int64),
            "intrinsics": intrinsics,
            "index": np.array(index, dtype=np.int32)
        }

    def get_metadata(self, index):
        return self.labels[index]

    @staticmethod
    def prepare_pytorch_input(color_image: np.ndarray, depth_or_point_image: np.ndarray,
                              intrinsics: dict, input_height: int, input_width: int, cropper=None,
                              max_boundary_dist: float = 0.1, compute_boundary_mask: bool = False) \
            -> typing.Tuple[np.ndarray, typing.Union[np.ndarray, None], StaticCenterCrop]:
        # Backproject depth image.
        if len(depth_or_point_image.shape) == 2:
            depth_or_point_image = image_processing.backproject_depth(depth_or_point_image,
                                                                      intrinsics["fx"], intrinsics["fy"],
                                                                      intrinsics["cx"], intrinsics["cy"])  # (h, w, 3)
            point_image = depth_or_point_image.astype(np.float32)
        else:
            assert len(depth_or_point_image.shape) == 3 and depth_or_point_image.shape[2] == 3
            point_image = depth_or_point_image

        image_size = color_image.shape[:2]

        # Crop, since we need it to be divisible by 64
        if cropper is None:
            cropper = StaticCenterCrop(image_size, (input_height, input_width))

        color_image = cropper(color_image)
        point_image = cropper(point_image)

        # Construct the final image by converting uint RGB to float RGB
        # and stitching RGB+XYZ in the first axis.
        image_rgbxyz = np.zeros((6, input_height, input_width), dtype=np.float32)

        image_rgbxyz[:3, :, :] = np.moveaxis(color_image, -1, 0) / 255.0  # (3, h, w)

        assert np.max(image_rgbxyz[:3, :, :]) <= 1.0, np.max(image_rgbxyz[:3, :, :])
        image_rgbxyz[3:, :, :] = np.moveaxis(point_image, -1, 0)  # (3, h, w)

        if not compute_boundary_mask:
            return image_rgbxyz, None, cropper
        else:
            assert max_boundary_dist
            boundary_mask = image_processing.compute_boundary_mask(depth_or_point_image, max_boundary_dist)
            return image_rgbxyz, boundary_mask, cropper

    @staticmethod
    def load_image(
            color_image_path, depth_image_path,
            intrinsics, input_height, input_width, cropper=None,
            max_boundary_distance=0.1, compute_boundary_mask=False):
        # Load images.
        color_image = skimage.io.imread(color_image_path)  # (h, w, 3)
        depth_image = skimage.io.imread(depth_image_path)  # (h, w)
        return DeformDataset.prepare_pytorch_input(
            color_image, depth_image, intrinsics, input_height, input_width, cropper,
            max_boundary_distance, compute_boundary_mask)

    @staticmethod
    def load_flow(optical_flow_image_path, scene_flow_image_path, cropper):
        # Load flow images.
        optical_flow_image = load_flow(optical_flow_image_path)  # (2, h, w)
        scene_flow_image = load_flow(scene_flow_image_path)  # (3, h, w)

        # Temporarily move axis for cropping
        optical_flow_image = np.moveaxis(optical_flow_image, 0, -1)  # (h, w, 2)
        scene_flow_image = np.moveaxis(scene_flow_image, 0, -1)  # (h, w, 3)

        # Crop for dimensions to be divisible by 64
        optical_flow_image = cropper(optical_flow_image)
        scene_flow_image = cropper(scene_flow_image)

        # Compute flow mask.
        optical_flow_mask = np.isfinite(optical_flow_image)  # (h, w, 2)
        optical_flow_mask = np.logical_and(optical_flow_mask[..., 0], optical_flow_mask[..., 1])  # (h, w)
        optical_flow_mask = optical_flow_mask[..., np.newaxis]  # (h, w, 1)
        optical_flow_mask = np.repeat(optical_flow_mask, 2, axis=2)  # (h, w, 2)

        scene_flow_mask = np.isfinite(scene_flow_image)  # (h, w, 3)
        scene_flow_mask = np.logical_and(scene_flow_mask[..., 0], scene_flow_mask[..., 1], scene_flow_mask[..., 2])  # (h, w)
        scene_flow_mask = scene_flow_mask[..., np.newaxis]  # (h, w, 1)
        scene_flow_mask = np.repeat(scene_flow_mask, 3, axis=2)  # (h, w, 3)

        # set invalid pixels to zero in the flow image
        optical_flow_image[optical_flow_mask == False] = 0.0
        scene_flow_image[scene_flow_mask == False] = 0.0

        # put channels back in first axis
        optical_flow_image = np.moveaxis(optical_flow_image, -1, 0).astype(np.float32)  # (2, h, w)
        optical_flow_mask = np.moveaxis(optical_flow_mask, -1, 0).astype(np.int64)  # (2, h, w)

        scene_flow_image = np.moveaxis(scene_flow_image, -1, 0).astype(np.float32)  # (3, h, w)
        scene_flow_mask = np.moveaxis(scene_flow_mask, -1, 0).astype(np.int64)  # (3, h, w)

        return optical_flow_image, optical_flow_mask, scene_flow_image, scene_flow_mask

    @staticmethod
    def load_anchors_and_weights(pixel_anchors_path: str, pixel_weights_path: str,
                                 cropper: typing.Union[StaticCenterCrop, None]) -> typing.Tuple[np.ndarray, np.ndarray]:
        if cropper is not None:
            pixel_anchors = cropper(load_int_image(pixel_anchors_path))
            pixel_weights = cropper(load_float_image(pixel_weights_path))
        else:
            pixel_anchors = load_int_image(pixel_anchors_path)
            pixel_weights = load_float_image(pixel_weights_path)

        assert np.isfinite(pixel_weights).all(), pixel_weights
        return pixel_anchors, pixel_weights

    @staticmethod
    def load_anchors_and_weights_from_sequence_directory_and_graph_filename(sequence_directory: str, graph_filename: str,
                                                                            cropper: typing.Union[StaticCenterCrop, None]):
        pixel_anchors_path = os.path.join(sequence_directory, "pixel_anchors", graph_filename + ".bin")
        pixel_weights_path = os.path.join(sequence_directory, "pixel_weights", graph_filename + ".bin")
        return DeformDataset.load_anchors_and_weights(pixel_anchors_path, pixel_weights_path, cropper)

    @staticmethod
    @dispatch(str, str, str, object, str)
    def load_graph_data(
            graph_nodes_path: str, graph_edges_path: str, graph_edges_weights_path: str,
            graph_node_deformations_path: typing.Union[None, str], graph_clusters_path: str
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, typing.Union[np.ndarray, None], np.ndarray]:
        # Load data.
        graph_nodes = load_graph_nodes_or_deformations(graph_nodes_path)
        graph_edges = load_graph_edges(graph_edges_path)
        graph_edges_weights = load_graph_edges_weights(graph_edges_weights_path)
        graph_node_deformations = load_graph_nodes_or_deformations(graph_node_deformations_path) \
            if graph_node_deformations_path is not None else None
        graph_clusters = load_graph_clusters(graph_clusters_path)

        assert np.isfinite(graph_edges_weights).all(), graph_edges_weights

        if graph_node_deformations is not None:
            assert np.isfinite(graph_node_deformations).all(), graph_node_deformations
            assert graph_node_deformations.shape[1] == 3
            assert graph_node_deformations.dtype == np.float32

        return graph_nodes, graph_edges, graph_edges_weights, graph_node_deformations, graph_clusters

    @staticmethod
    @dispatch(str, str, bool)
    def load_graph_data(sequence_directory: str, graph_filename: str, load_deformations: bool):
        graph_nodes_path = os.path.join(sequence_directory, "graph_nodes", graph_filename + ".bin")
        graph_edges_path = os.path.join(sequence_directory, "graph_edges", graph_filename + ".bin")
        graph_edges_weights_path = os.path.join(sequence_directory, "graph_edges_weights", graph_filename + ".bin")
        graph_node_deformations_path = None if not load_deformations \
            else os.path.join(sequence_directory, "graph_node_deformations", graph_filename + ".bin")
        graph_clusters_path = os.path.join(sequence_directory, "graph_clusters", graph_filename + ".bin")

        return DeformDataset.load_graph_data(graph_nodes_path, graph_edges_path, graph_edges_weights_path,
                                             graph_node_deformations_path, graph_clusters_path)

    @staticmethod
    def collate_with_padding(batch):
        batch_size = len(batch)

        # Compute max number of canonical_node_positions.
        item_keys = 0
        max_num_nodes = 0
        for sample_idx in range(batch_size):
            item_keys = batch[sample_idx].keys()
            num_nodes = batch[sample_idx]["num_nodes"]
            if num_nodes > max_num_nodes:
                max_num_nodes = num_nodes

        # Convert merged parts into torch tensors.
        # We pad graph canonical_node_positions, edges and deformation ground truth with zeros.
        batch_converted = {}

        for key in item_keys:
            if key == "graph_nodes" or key == "graph_edges" or \
                    key == "graph_edges_weights" or key == "graph_node_deformations" or \
                    key == "graph_clusters":
                batched_sample = torch.zeros((batch_size, max_num_nodes, batch[0][key].shape[1]), dtype=torch.from_numpy(batch[0][key]).dtype)
                for sample_idx in range(batch_size):
                    batched_sample[sample_idx, :batch[sample_idx][key].shape[0], :] = torch.from_numpy(batch[sample_idx][key])
                batch_converted[key] = batched_sample
            else:
                batched_sample = torch.zeros((batch_size, *batch[0][key].shape), dtype=torch.from_numpy(batch[0][key]).dtype)
                for sample_idx in range(batch_size):
                    batched_sample[sample_idx] = torch.from_numpy(batch[sample_idx][key])
                batch_converted[key] = batched_sample

        return [
            batch_converted["source"],
            batch_converted["target"],
            batch_converted["target_boundary_mask"],
            batch_converted["optical_flow_gt"],
            batch_converted["optical_flow_mask"],
            batch_converted["scene_flow_gt"],
            batch_converted["scene_flow_mask"],
            batch_converted["graph_nodes"],
            batch_converted["graph_edges"],
            batch_converted["graph_edges_weights"],
            batch_converted["graph_node_deformations"],
            batch_converted["graph_clusters"],
            batch_converted["pixel_anchors"],
            batch_converted["pixel_weights"],
            batch_converted["num_nodes"],
            batch_converted["intrinsics"],
            batch_converted["index"]
        ]
