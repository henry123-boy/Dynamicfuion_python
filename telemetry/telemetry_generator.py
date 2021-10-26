from typing import Union, List
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from data import SequenceFrameDataset
from warp_field.graph import DeformationGraphNumpy
from telemetry.visualization.fusion_visualization_recorder import FusionVisualizationRecorder
import pynvml
import os
import telemetry.visualization.tracking as tracking_viz
from settings.fusion import VisualizationMode
from settings import Parameters
import ext_argparse


class TelemetryGenerator:
    def __init__(self,
                 record_visualization_to_disk: bool,
                 record_framewise_canonical_mesh: bool,
                 record_framewise_warped_mesh: bool,
                 record_rendered_warped_mesh: bool,
                 record_gn_point_clouds: bool,
                 record_source_and_target_point_clouds: bool,
                 record_graph_transformations: bool,
                 print_cuda_memory_info: bool,
                 print_frame_info: bool,
                 visualization_mode: VisualizationMode,
                 output_directory: str,
                 record_over_run_time: datetime = datetime.now()):

        save_all_parameters_to_file = True

        # set up mesh recorder
        self.mesh_video_recorder = None
        self.record_visualization_to_disk = record_visualization_to_disk
        self.record_framewise_canonical_mesh = record_framewise_canonical_mesh
        self.record_framewise_warped_mesh = record_framewise_warped_mesh
        self.record_rendered_mesh = record_rendered_warped_mesh
        self.record_gn_point_clouds = record_gn_point_clouds
        self.record_source_and_target_point_clouds = record_source_and_target_point_clouds
        self.record_graph_transformations = record_graph_transformations
        self.visualization_mode = visualization_mode
        self.output_directory = os.path.join(output_directory, record_over_run_time.strftime("%y-%m-%d-%H-%M-%S"))
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        if save_all_parameters_to_file:
            settings_path = Path(os.path.join(self.output_directory, "settings.yaml"))
            ext_argparse.dump(Parameters, settings_path)

        if self.record_visualization_to_disk:
            # TODO fix recording, it currently doesn't seem to work. Perhaps use something other than Open3D for rendering,
            #  e.g. PyTorch3D, and ffmpeg-python for recording instead of OpenCV
            self.mesh_video_recorder = FusionVisualizationRecorder(
                output_video_path=os.path.join(self.output_directory, "mesh_visualization.mkv"),
                front=[0, 0, -1], lookat=[0, 0, 1.5],
                up=[0, -1.0, 0], zoom=0.7
            )
        self.print_gpu_memory_info = print_cuda_memory_info
        if self.print_gpu_memory_info:
            pynvml.nvmlInit()
        self.print_frame_info = print_frame_info
        self.frame_output_directory = os.path.join(self.output_directory, "frame_output")
        if not os.path.exists(self.frame_output_directory):
            os.makedirs(self.frame_output_directory)

        # Note: if you want to store GN point clouds in a different directory, uncomment here and use
        # self.deformed_points_output_directory in the process_gn_point_cloud method instead of self.frame_output_directory
        #
        # self.deformed_points_output_directory = os.path.join(self.output_directory, "gn_deformed_points")
        # if self.record_gn_point_clouds and not os.path.exists(self.deformed_points_output_directory):
        #     os.makedirs(self.deformed_points_output_directory)
        self.frame_index = 0

    def set_frame_index(self, frame_index):
        self.frame_index = frame_index

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
                              additional_geometry: List = []):
        if self.visualization_mode == VisualizationMode.POINT_CLOUD_TRACKING:
            node_count = len(graph.nodes)
            rotations_pred = deform_net_data["node_rotations"].view(node_count, 3, 3).cpu().numpy()
            translations_pred = deform_net_data["node_translations"].view(node_count, 3).cpu().numpy()

            # TODO: not sure what the mask prediction can be useful for except in visualization so far...
            mask_pred = deform_net_data["mask_pred"]
            assert mask_pred is not None, "Make sure use_mask is used / set to true in settings."
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

    def process_rendering_result(self, color_image, depth_image, frame_index):
        if self.record_rendered_mesh:
            cv2.imwrite(os.path.join(self.frame_output_directory, f"{frame_index:06d}_rendered_color.jpg"), color_image)
            cv2.imwrite(os.path.join(self.frame_output_directory, f"{frame_index:06d}_rendered_depth.png"), depth_image)

    def process_result_visualization_and_logging(self,
                                                 canonical_mesh: Union[None, o3d.geometry.TriangleMesh],
                                                 warped_mesh: Union[None, o3d.geometry.TriangleMesh],
                                                 deform_net_data: dict,
                                                 tracking_image_height: int, tracking_image_width: int,
                                                 source_rgbxyz: np.ndarray, target_rgbxyz: np.ndarray,
                                                 pixel_anchors: np.ndarray, pixel_weights: np.ndarray,
                                                 graph: DeformationGraphNumpy):
        if self.visualization_mode == VisualizationMode.CANONICAL_MESH:
            self.process_canonical_mesh(canonical_mesh)
        elif self.visualization_mode == VisualizationMode.WARPED_MESH:
            self.process_warped_mesh(warped_mesh)
        elif self.visualization_mode == VisualizationMode.POINT_CLOUD_TRACKING and deform_net_data is not None:
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

    def record_meshes_to_disk_if_needed(self,
                                        canonical_mesh: Union[None, o3d.geometry.TriangleMesh],
                                        warped_mesh: Union[None, o3d.geometry.TriangleMesh]):
        if self.record_framewise_canonical_mesh:
            o3d.io.write_triangle_mesh(os.path.join(self.frame_output_directory, f"{self.frame_index:06d}_canonical_mesh.ply"),
                                       canonical_mesh)
        if self.record_framewise_warped_mesh:
            o3d.io.write_triangle_mesh(os.path.join(self.frame_output_directory, f"{self.frame_index:06d}_warped_mesh.ply"),
                                       warped_mesh)

    def print_frame_info_if_needed(self, current_frame: SequenceFrameDataset):
        if self.print_frame_info:
            print("Processing frame:", current_frame.frame_index)
            print("Color path:", current_frame.color_image_path)
            print("Depth path:", current_frame.depth_image_path)

    def process_gn_point_cloud(self, deformed_points: torch.Tensor, source_colors: torch.Tensor, gauss_newton_iteration):
        if self.record_gn_point_clouds:
            path = os.path.join(self.frame_output_directory,
                                f"{self.frame_index:06d}_deformed_points_iter_{gauss_newton_iteration:03d}.npy")
            np.save(path, np.concatenate((source_colors.cpu().detach().numpy().reshape(-1, 3),
                                          deformed_points.cpu().detach().numpy().reshape(-1, 3)), axis=1))

    def process_source_and_target_point_clouds(self, source_rgbxyz, target_rgbxyz):
        if self.record_source_and_target_point_clouds:
            source_path = os.path.join(self.frame_output_directory, f"{self.frame_index:06d}_source_rgbxyz.npy")
            np.save(source_path, source_rgbxyz.reshape(6, -1).T)
            target_path = os.path.join(self.frame_output_directory, f"{self.frame_index:06d}_target_rgbxyz.npy")
            np.save(target_path, target_rgbxyz.reshape(6, -1).T)

    def process_graph_transformation(self, graph: DeformationGraphNumpy):
        if self.record_graph_transformations:
            nodes_path = os.path.join(self.frame_output_directory, f"{self.frame_index:06d}_nodes.npy")
            np.save(nodes_path, graph.nodes)
            edges_path = os.path.join(self.frame_output_directory, f"{self.frame_index:06d}_edges.npy")
            np.save(edges_path, graph.edges)
            rotations_path = os.path.join(self.frame_output_directory, f"{self.frame_index:06d}_node_rotations.npy")
            np.save(rotations_path, graph.rotations_mat)
            translations_path = os.path.join(self.frame_output_directory, f"{self.frame_index:06d}_node_translations.npy")
            np.save(translations_path, graph.translations_vec)
