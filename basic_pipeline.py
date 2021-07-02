#!/usr/bin/python3

# experimental fusion pipeline based on original NNRT code
# Copyright 2021 Gregory Kramida

# stdlib
import sys
import typing
from enum import Enum
import cProfile
import argparse

# 3rd-party
import numpy as np
import open3d as o3d
import open3d.core as o3c
from dq3d import dualquat, quat

from scipy.spatial.transform.rotation import Rotation

# local
import nnrt

import options
from alignment.interface import run_non_rigid_alignment
from data import camera
from data import *
from pipeline.numba_cuda.preprocessing import cuda_compute_normal
from pipeline.numpy_cpu.preprocessing import cpu_compute_normal
from pipeline.rendering.pytorch3d_renderer import PyTorch3DRenderer
import utils.image
import utils.voxel_grid
from alignment.deform_net import DeformNet
from alignment.default import load_default_nnrt_network
from pipeline.graph import DeformationGraphNumpy, build_deformation_graph_from_mesh
import pipeline.pipeline_options as po
from pipeline.telemetry_generator import TelemetryGenerator

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


class FusionPipeline:
    def __init__(self):
        # === preprocess options & initialize telemetry ===
        self.extracted_framewise_canonical_mesh_needed = \
            po.source_image_mode != po.SourceImageMode.REUSE_PREVIOUS_FRAME or \
            po.visualization_mode == po.VisualizationMode.CANONICAL_MESH or \
            po.visualization_mode == po.VisualizationMode.WARPED_MESH or \
            po.record_canonical_meshes_to_disk

        self.framewise_warped_mesh_needed = \
            po.source_image_mode != po.SourceImageMode.REUSE_PREVIOUS_FRAME or \
            po.visualization_mode == po.VisualizationMode.WARPED_MESH or \
            po.record_warped_meshes_to_disk or po.record_rendered_warped_mesh

        self.telemetry_generator = TelemetryGenerator(po.record_visualization_to_disk,
                                                      po.record_canonical_meshes_to_disk,
                                                      po.record_warped_meshes_to_disk,
                                                      po.record_rendered_warped_mesh,
                                                      po.record_gn_point_clouds,
                                                      po.print_cuda_memory_info,
                                                      po.print_frame_info,
                                                      po.visualization_mode, options.output_directory)

        # === load alignment network, configure device ===
        self.deform_net: DeformNet = load_default_nnrt_network(self.telemetry_generator)
        self.device = o3d.core.Device('cuda:0')

        # === initialize structures ===
        self.graph: typing.Union[DeformationGraphNumpy, None] = None
        self.volume = utils.voxel_grid.make_default_tsdf_voxel_grid(self.device)

        #####################################################################################################
        # region === dataset, intrinsics & extrinsics in various shapes, sizes, and colors ===
        #####################################################################################################
        self.sequence: FrameSequenceDataset = po.sequence
        first_frame = self.sequence.get_frame_at(0)

        intrinsics_open3d_cpu = camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(
            self.sequence.get_intrinsics_path(),
            first_frame.get_depth_image_path())
        self.fx, self.fy, self.cx, self.cy = camera.extract_intrinsic_projection_parameters(intrinsics_open3d_cpu)
        self.intrinsics_dict = camera.intrinsic_projection_parameters_as_dict(intrinsics_open3d_cpu)
        if po.print_intrinsics:
            camera.print_intrinsic_projection_parameters(intrinsics_open3d_cpu)

        self.intrinsics_open3d_device = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix,
                                                        o3d.core.Dtype.Float32, self.device)
        self.extrinsics_open3d_device = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32, self.device)
        # endregion

    def extract_and_warp_canonical_mesh_if_necessary(self):
        # TODO: try to speed up by using the extracted CUDA-based mesh directly (and converting to torch tensors via
        #  dlpack for rendering where this can be done).
        #  Conversion to legacy mesh can be delegated to before visualization, and only we're trying to visualize one of these meshes
        #  The first step is to provide warping for the o3d.t.geometry.TriangleMesh (see graph.py).
        #  This may involve augmenting the Open3D extension in the local C++/CUDA code.
        canonical_mesh: typing.Union[None, o3d.geometry.TriangleMesh] = None
        if self.extracted_framewise_canonical_mesh_needed:
            canonical_mesh = self.volume.extract_surface_mesh(0).to_legacy_triangle_mesh()

        warped_mesh: typing.Union[None, o3d.geometry.TriangleMesh] = None
        # TODO: perform topological graph update
        if self.framewise_warped_mesh_needed:
            if po.transformation_mode == po.TransformationMode.QUATERNIONS:
                warped_mesh = self.graph.warp_mesh_dq(canonical_mesh, options.node_coverage)
            else:
                warped_mesh = self.graph.warp_mesh_mat(canonical_mesh, options.node_coverage)
        return canonical_mesh, warped_mesh

    def run(self) -> int:
        telemetry_generator = self.telemetry_generator
        deform_net = self.deform_net
        device = self.device
        volume = self.volume
        sequence = self.sequence

        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        #####################################################################################################
        # region === initialize loop data structures ===
        #####################################################################################################

        renderer = PyTorch3DRenderer(self.sequence.resolution, device, self.intrinsics_open3d_device)

        previous_depth_image_np: typing.Union[None, np.ndarray] = None
        previous_color_image_np: typing.Union[None, np.ndarray] = None
        previous_mask_image_np: typing.Union[None, np.ndarray] = None

        canonical_mesh: typing.Union[None, o3d.geometry.TriangleMesh] = None
        warped_mesh: typing.Union[None, o3d.geometry.TriangleMesh] = None

        while sequence.has_more_frames():
            current_frame = sequence.get_next_frame()
            self.telemetry_generator.set_frame_index(current_frame.frame_index)
            #####################################################################################################
            # region ===== grab images, mask / clip if necessary, transfer to GPU versions for Open3D ===========
            #####################################################################################################
            telemetry_generator.print_frame_info_if_needed(current_frame)
            telemetry_generator.print_cuda_memory_info_if_needed()

            depth_image_open3d_legacy = o3d.io.read_image(current_frame.depth_image_path)
            depth_image_np = np.array(depth_image_open3d_legacy)

            color_image_open3d_legacy = o3d.io.read_image(current_frame.color_image_path)
            color_image_np = np.array(color_image_open3d_legacy)

            # limit the number of nodes & clusters by cutting at depth
            if sequence.far_clipping_distance_mm > 0:
                color_image_np[depth_image_np > sequence.far_clipping_distance_mm] = 0
                depth_image_np[depth_image_np > sequence.far_clipping_distance_mm] = 0

            # limit the number of nodes & clusters by masking out a segment
            if sequence.has_masks():
                mask_image_open3d_legacy = o3d.io.read_image(current_frame.mask_image_path)
                mask_image_np = np.array(mask_image_open3d_legacy)
                color_image_np[mask_image_np < sequence.mask_lower_threshold] = 0
                depth_image_np[mask_image_np < sequence.mask_lower_threshold] = 0

            mask_image_np = depth_image_np != 0

            depth_image_open3d = o3d.t.geometry.Image(o3c.Tensor(depth_image_np, device=device))
            color_image_open3d = o3d.t.geometry.Image(o3c.Tensor(color_image_np, device=device))

            # endregion
            if current_frame.frame_index == sequence.start_frame_index:
                volume.integrate(depth_image_open3d, color_image_open3d, self.intrinsics_open3d_device, self.extrinsics_open3d_device, options.depth_scale, 3.0)
                volume.activate_sleeve_blocks()
                volume.activate_sleeve_blocks()
                canonical_mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()

                # === Construct initial deformation graph
                if po.graph_generation_mode == po.GraphGenerationMode.FIRST_FRAME_EXTRACTED_MESH:
                    self.graph = build_deformation_graph_from_mesh(canonical_mesh, options.node_coverage,
                                                                   erosion_iteration_count=10,
                                                                   neighbor_count=8)
                elif po.graph_generation_mode == po.GraphGenerationMode.FIRST_FRAME_LOADED_GRAPH:
                    self.graph = sequence.get_current_frame_graph()
                    if self.graph is None:
                        raise ValueError(f"Could not load graph for frame {current_frame.frame_index}.")
                else:
                    raise NotImplementedError(f"graph generation mode {po.graph_generation_mode.name} not implemented.")
                canonical_mesh, warped_mesh = self.extract_and_warp_canonical_mesh_if_necessary()
            else:

                #####################################################################################################
                # region ===== prepare source point cloud & RGB image ====
                #####################################################################################################
                if po.source_image_mode == po.SourceImageMode.REUSE_PREVIOUS_FRAME:
                    source_depth = previous_depth_image_np
                    source_color = previous_color_image_np
                else:
                    source_depth, source_color = renderer.render_mesh(warped_mesh, depth_scale=options.depth_scale)
                    source_depth = source_depth.astype(np.uint16)
                    telemetry_generator.process_rendering_result(source_color, source_depth, current_frame.frame_index)

                    # flip channels, i.e. RGB<-->BGR
                    source_color = cv2.cvtColor(source_color, cv2.COLOR_BGR2RGB)
                    if po.source_image_mode == po.SourceImageMode.RENDERED_WITH_PREVIOUS_FRAME_OVERLAY:
                        # re-use pixel data from previous frame
                        source_depth[previous_mask_image_np] = previous_depth_image_np[previous_mask_image_np]
                        source_color[previous_mask_image_np] = previous_color_image_np[previous_mask_image_np]

                source_point_image = image_utils.backproject_depth(source_depth, fx, fy, cx, cy, depth_scale=options.depth_scale)  # (h, w, 3)

                source_rgbxyz, _, cropper = DeformDataset.prepare_pytorch_input(
                    source_color, source_point_image, self.intrinsics_dict,
                    options.alignment_image_height, options.alignment_image_width
                )
                # endregion
                #####################################################################################################
                # region === prepare target point cloud, RGB image, normal map, pixel anchors, and pixel weights ====
                #####################################################################################################
                # TODO: replace options.depth_scale by a calibration property read from disk for each dataset
                target_point_image = image_utils.backproject_depth(depth_image_np, fx, fy, cx, cy, depth_scale=options.depth_scale)  # (h, w, 3)
                target_rgbxyz, _, _ = DeformDataset.prepare_pytorch_input(
                    color_image_np, target_point_image, self.intrinsics_dict,
                    options.alignment_image_height, options.alignment_image_width,
                    cropper=cropper
                )
                if device.get_type() == o3c.Device.CUDA:
                    target_normal_map = cuda_compute_normal(target_point_image)
                else:
                    target_normal_map = cpu_compute_normal(target_point_image)
                target_normal_map_o3d = o3c.Tensor(target_normal_map, dtype=o3c.Dtype.Float32, device=device)

                if po.pixel_anchor_computation_mode == po.AnchorComputationMode.EUCLIDEAN:
                    pixel_anchors, pixel_weights = nnrt.compute_pixel_anchors_euclidean(
                        self.graph.nodes, source_point_image, options.node_coverage
                    )
                else:
                    pixel_anchors, pixel_weights = nnrt.compute_pixel_anchors_shortest_path(
                        source_point_image, self.graph.nodes, self.graph.edges, po.anchor_node_count, options.node_coverage
                    )

                pixel_anchors = cropper(pixel_anchors)
                pixel_weights = cropper(pixel_weights)
                # endregion
                #####################################################################################################
                # region === adjust intrinsic / projection parameters due to cropping ====
                #####################################################################################################
                fx_cropped, fy_cropped, cx_cropped, cy_cropped = utils.image.modify_intrinsics_due_to_cropping(
                    fx, fy, cx, cy, options.alignment_image_height, options.alignment_image_width, original_h=cropper.h, original_w=cropper.w
                )
                cropped_intrinsics_numpy = np.array([fx_cropped, fy_cropped, cx_cropped, cy_cropped], dtype=np.float32)
                # endregion

                #####################################################################################################
                # region === run the motion prediction & optimization ====
                #####################################################################################################

                deform_net_data = run_non_rigid_alignment(deform_net, source_rgbxyz, target_rgbxyz, pixel_anchors,
                                                          pixel_weights, self.graph, cropped_intrinsics_numpy, device)

                # Get some of the results
                node_count = len(self.graph.nodes)
                rotations_pred = deform_net_data["node_rotations"].view(node_count, 3, 3).cpu().numpy()
                translations_pred = deform_net_data["node_translations"].view(node_count, 3).cpu().numpy()

                # endregion
                #####################################################################################################
                # region === fuse alignment ====
                #####################################################################################################
                # use the resulting frame transformation predictions to update the global, cumulative node transformations

                for rotation, translation, i_node in zip(rotations_pred, translations_pred, np.arange(0, node_count)):
                    node_position = self.graph.nodes[i_node]
                    current_rotation = self.graph.rotations_mat[i_node] = \
                        rotation.dot(self.graph.rotations_mat[i_node])
                    current_translation = self.graph.translations_vec[i_node] = \
                        translation + self.graph.translations_vec[i_node]

                    translation_global = node_position + current_translation - current_rotation.dot(node_position)
                    self.graph.transformations_dq[i_node] = dualquat(quat(current_rotation), translation_global)

                # prepare data for Open3D integration
                nodes_o3d = o3c.Tensor(self.graph.nodes, dtype=o3c.Dtype.Float32, device=device)

                if po.transformation_mode == po.TransformationMode.QUATERNIONS:
                    # prepare data for Open3D integration
                    node_dual_quaternions = np.array([np.concatenate((dq.real.data, dq.dual.data)) for dq in self.graph.transformations_dq])
                    node_dual_quaternions_o3d = o3c.Tensor(node_dual_quaternions, dtype=o3c.Dtype.Float32, device=device)
                    cos_voxel_ray_to_normal = volume.integrate_warped_euclidean_dq(
                        depth_image_open3d, color_image_open3d, target_normal_map_o3d,
                        self.intrinsics_open3d_device, self.extrinsics_open3d_device,
                        nodes_o3d, node_dual_quaternions_o3d, options.node_coverage,
                        anchor_count=po.anchor_node_count, minimum_valid_anchor_count=po.fusion_minimum_valid_anchor_count,
                        depth_scale=options.depth_scale, depth_max=3.0)
                elif po.transformation_mode == po.TransformationMode.MATRICES:
                    node_rotations_o3d = o3c.Tensor(self.graph.rotations_mat, dtype=o3c.Dtype.Float32, device=device)
                    node_translations_o3d = o3c.Tensor(self.graph.translations_vec, dtype=o3c.Dtype.Float32, device=device)
                    cos_voxel_ray_to_normal = volume.integrate_warped_euclidean_mat(
                        depth_image_open3d, color_image_open3d, target_normal_map_o3d,
                        self.intrinsics_open3d_device, self.extrinsics_open3d_device,
                        nodes_o3d, node_rotations_o3d, node_translations_o3d, options.node_coverage,
                        anchor_count=po.anchor_node_count, minimum_valid_anchor_count=po.fusion_minimum_valid_anchor_count,
                        depth_scale=options.depth_scale, depth_max=3.0)
                else:
                    raise ValueError("Unsupported motion blending mode")
                # TODO: not sure how the cos_voxel_ray_to_normal can be useful after the integrate_warped operation.
                #  Check BaldrLector's NeuralTracking fork code.
                # endregion
                #####################################################################################################

                telemetry_generator.process_result_visualization_and_logging(
                    canonical_mesh, warped_mesh,
                    deform_net_data,
                    options.alignment_image_height,
                    options.alignment_image_width,
                    source_rgbxyz, target_rgbxyz,
                    pixel_anchors, pixel_weights,
                    self.graph
                )
                canonical_mesh, warped_mesh = self.extract_and_warp_canonical_mesh_if_necessary()

            previous_color_image_np = color_image_np
            previous_depth_image_np = depth_image_np
            previous_mask_image_np = mask_image_np

        return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Basic Fusion Pipeline based on Neural Non-Rigid Tracking + Fusion4D + Open3D spatial hashing")
    parser.add_argument("--profile", action='store_true')
    args = parser.parse_args()
    pipeline = FusionPipeline()
    if args.profile:
        cProfile.run('pipeline.run()')
    else:
        sys.exit(pipeline.run())
