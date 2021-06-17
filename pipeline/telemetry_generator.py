import os
import typing

import numpy as np
import open3d as o3d

from data import SequenceFrameDataset
from pipeline.graph import DeformationGraphNumpy
from utils.viz.fusion_visualization_recorder import FusionVisualizationRecorder
from pynvml import *
from pipeline.pipeline_options import VisualizationMode
import utils.viz.tracking as tracking_viz


class TelemetryGenerator:
    def __init__(self,
                 record_visualization_to_disk: bool,
                 print_cuda_memory_info: bool,
                 print_frame_info: bool,
                 visualization_mode: VisualizationMode,
                 output_directory: str):
        # set up mesh recorder
        self.mesh_video_recorder = None
        self.record_visualization_to_disk = record_visualization_to_disk
        self.visualization_mode = visualization_mode
        if self.record_visualization_to_disk:
            # TODO fix recording, it currently doesn't seem to work. Perhaps use something other than Open3D for rendering,
            #  e.g. PyTorch3D, and ffmpeg-python for recording instead of OpenCV
            self.mesh_video_recorder = FusionVisualizationRecorder(
                output_video_path=os.path.join(output_directory, "mesh_visualization.mkv"),
                front=[0, 0, -1], lookat=[0, 0, 1.5],
                up=[0, -1.0, 0], zoom=0.7
            )
        self.print_gpu_memory_info = print_cuda_memory_info
        if self.print_gpu_memory_info:
            nvmlInit()
        self.print_frame_info = print_frame_info

    def print_cuda_memory_info_if_needed(self):
        if self.print_gpu_memory_info:
            device_handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(device_handle)
            print(f'total    : {info.total}')
            print(f'free     : {info.free}')
            print(f'used     : {info.used}')

    def process_canonical_mesh(self, canonical_mesh: o3d.geometry.TriangleMesh) -> None:
        if self.visualization_mode == VisualizationMode.CANONICAL_MESH:
            if self.record_visualization_to_disk:
                # FIXME: mesh_video_recorder is broken, fix
                self.mesh_video_recorder.capture_frame([canonical_mesh])
            else:
                o3d.visualization.draw_geometries([canonical_mesh],
                                                  front=[0, 0, -1],
                                                  lookat=[0, 0, 1.5],
                                                  up=[0, -1.0, 0],
                                                  zoom=0.7)

    def process_warped_mesh(self, warped_mesh: o3d.geometry.TriangleMesh) -> None:
        if self.visualization_mode == VisualizationMode.WARPED_MESH:
            if self.record_visualization_to_disk:
                self.mesh_video_recorder.capture_frame([warped_mesh])
            else:
                o3d.visualization.draw_geometries([warped_mesh],
                                                  front=[0, 0, -1],
                                                  lookat=[0, 0, 1.5],
                                                  up=[0, -1.0, 0],
                                                  zoom=0.7)

    def process_alignment_viz(self, deform_net_data: dict,
                              tracking_image_height,
                              tracking_image_width,
                              source_rgbxyz: np.ndarray,
                              target_rgbxyz: np.ndarray,
                              pixel_anchors: np.ndarray,
                              pixel_weights: np.ndarray,
                              graph: DeformationGraphNumpy,
                              additional_geometry: typing.List = []):
        if self.visualization_mode == VisualizationMode.POINT_CLOUD_TRACKING:
            node_count = len(graph.nodes)
            rotations_pred = deform_net_data["node_rotations"].view(node_count, 3, 3).cpu().numpy()
            translations_pred = deform_net_data["node_translations"].view(node_count, 3).cpu().numpy()

            # TODO: not sure what the mask prediction can be useful for except in visualization so far...
            mask_pred = deform_net_data["mask_pred"]
            assert mask_pred is not None, "Make sure use_mask=True in options.py"
            mask_pred = mask_pred.view(-1, tracking_image_height, tracking_image_width).cpu().numpy()
            # Compute mask gt for mask baseline
            _, source_points, valid_source_points, target_matches, valid_target_matches, valid_correspondences, _, _ \
                = deform_net_data["correspondence_info"]

            target_matches = target_matches.view(-1, tracking_image_height, tracking_image_width).cpu().numpy()
            valid_source_points = valid_source_points.view(-1, tracking_image_height, tracking_image_width).cpu().numpy()
            valid_correspondences = valid_correspondences.view(-1, tracking_image_height, tracking_image_width).cpu().numpy()

            additional_geometry = [item for item in additional_geometry if item is not None]
            tracking_viz.visualize_tracking(source_rgbxyz, target_rgbxyz, pixel_anchors, pixel_weights,
                                            graph.get_warped_nodes(), graph.edges,
                                            rotations_pred, translations_pred, mask_pred,
                                            valid_source_points, valid_correspondences, target_matches, additional_geometry)

    def process_result_visualization_and_logging(self,
                                                 canonical_mesh: typing.Union[None, o3d.geometry.TriangleMesh],
                                                 warped_mesh: typing.Union[None, o3d.geometry.TriangleMesh],
                                                 deform_net_data: dict,
                                                 tracking_image_height: int, tracking_image_width: int,
                                                 source_rgbxyz: np.ndarray, target_rgbxyz: np.ndarray,
                                                 pixel_anchors: np.ndarray, pixel_weights: np.ndarray,
                                                 graph: DeformationGraphNumpy):
        if self.visualization_mode == VisualizationMode.CANONICAL_MESH:
            if canonical_mesh is None:
                raise ArgumentError("Expecting a TriangleMesh as canonical_mesh argument.")
            self.process_canonical_mesh(canonical_mesh)
        elif self.visualization_mode == VisualizationMode.WARPED_MESH:
            if canonical_mesh is None:
                raise ArgumentError("Expecting a TriangleMesh as warped_mesh argument.")
            self.process_warped_mesh(warped_mesh)
        elif self.visualization_mode == VisualizationMode.POINT_CLOUD_TRACKING:
            self.process_alignment_viz(deform_net_data,
                                       tracking_image_height,
                                       tracking_image_width,
                                       source_rgbxyz,
                                       target_rgbxyz,
                                       pixel_anchors,
                                       pixel_weights,
                                       graph)
        elif self.visualization_mode == VisualizationMode.COMBINED:
            raise NotImplementedError("TODO")

    def print_frame_info_if_needed(self, current_frame: SequenceFrameDataset):
        if self.print_frame_info:
            print("Processing frame:", current_frame.frame_index)
            print("Color path:", current_frame.color_image_path)
            print("Depth path:", current_frame.depth_image_path)
