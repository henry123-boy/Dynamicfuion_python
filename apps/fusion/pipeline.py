#!/usr/bin/python3

# experimental tsdf_management tsdf_management based on original NNRT code
# Copyright 2021 Gregory Kramida
from typing import Union, List, Tuple
import timeit

# 3rd-party
import numpy
import numpy as np
import open3d.core as o3c
import open3d as o3d
import cv2
import torch.utils.dlpack as torch_dlpack

# local
import nnrt

from alignment.deform_net import DeformNet
from alignment.default import load_default_nnrt_network
from alignment.interface import \
    run_non_rigid_alignment  # temporarily out-of-order here due to some CuPy 10 / CUDA 11.4 problems
from apps.create_graph_data import build_graph_warp_field_from_depth_image
from data import camera
from data import *
from image_processing.numba_cuda.preprocessing import cuda_compute_normal
from image_processing.numpy_cpu.preprocessing import cpu_compute_normal
import image_processing
from rendering.pytorch3d_renderer import PyTorch3DRenderer
import tsdf.default_voxel_grid as default_tsdf
from warp_field.graph_warp_field import build_deformation_graph_from_mesh
from settings.fusion import SourceImageMode, VisualizationMode, \
    AnchorComputationMode, TrackingSpanMode, GraphGenerationMode, MeshExtractionWeightThresholdingMode
from settings import Parameters
from telemetry.telemetry_generator import TelemetryGenerator

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


# NOTE: computation is currently performed using matrix math, not dual quaternions like in
# DynamicFusion (Newcombe et al.). For an incomplete dual-quaternion implementation, see commit f5923da (<2022.02.01)


class FusionPipeline:

    def __init__(self):
        FusionPipeline.perform_complex_logic_parameter_validation()

        viz_parameters = Parameters.fusion.telemetry.visualization
        log_parameters = Parameters.fusion.telemetry.logging
        verbosity_parameters = Parameters.fusion.telemetry.verbosity
        tracking_parameters = Parameters.fusion.tracking

        # === preprocess options & initialize telemetry ===
        # TODO: this logic needs to be handled in the TelemetryGenerator constructor. The flags from it can be checked
        #  here to see if certain operations will need to be done to produce input for the telemetry generator.
        self.extracted_framewise_canonical_mesh_needed = \
            tracking_parameters.source_image_mode.value != SourceImageMode.IMAGE_ONLY or \
            viz_parameters.visualization_mode.value in [VisualizationMode.CANONICAL_MESH,
                                                        VisualizationMode.WARPED_MESH] or \
            log_parameters.record_canonical_meshes_to_disk.value

        self.framewise_warped_mesh_needed = \
            (tracking_parameters.tracking_span_mode.value == TrackingSpanMode.PREVIOUS_TO_CURRENT
             and tracking_parameters.source_image_mode.value != SourceImageMode.IMAGE_ONLY) or \
            viz_parameters.visualization_mode.value == VisualizationMode.WARPED_MESH or \
            log_parameters.record_warped_meshes_to_disk.value or log_parameters.record_rendered_warped_mesh.value

        self.telemetry_generator = TelemetryGenerator(log_parameters.record_visualization_to_disk.value,
                                                      log_parameters.record_canonical_meshes_to_disk.value,
                                                      log_parameters.record_warped_meshes_to_disk.value,
                                                      log_parameters.record_rendered_warped_mesh.value,
                                                      log_parameters.record_gn_point_clouds.value,
                                                      log_parameters.record_source_and_target_point_clouds.value,
                                                      log_parameters.record_correspondences.value,
                                                      log_parameters.record_graph_transformations.value,
                                                      log_parameters.record_frameviewer_metadata.value,
                                                      verbosity_parameters.print_cuda_memory_info.value,
                                                      verbosity_parameters.print_frame_info.value,
                                                      viz_parameters.visualization_mode.value,
                                                      Parameters.path.output_directory.value)

        # === load alignment network, configure device ===
        self.deform_net: DeformNet = load_default_nnrt_network(o3c.Device.CUDA,
                                                               log_parameters.record_gn_point_clouds.value)
        self.device = o3d.core.Device('cuda:0')

        # === initialize structures ===
        self.active_graph: Union[nnrt.geometry.GraphWarpField, None] = None
        self.keyframe_graphs: List[nnrt.geometry.GraphWarpField] = []
        self.volume = default_tsdf.make_default_tsdf_voxel_grid(self.device)

        #####################################################################################################
        # region === dataset, intrinsics & extrinsics in various shapes, sizes, and colors ===
        #####################################################################################################
        self.sequence: FrameSequenceDataset = Parameters.fusion.input_data.sequence_preset.value.value
        self.sequence.load()
        self.start_at_frame_index = max(self.sequence.start_frame_index,
                                        Parameters.fusion.input_data.start_at_frame.value)
        first_frame = self.sequence.get_frame_at(self.start_at_frame_index)

        intrinsics_open3d_cpu, self.intrinsic_matrix_np = camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(
            self.sequence.get_intrinsics_path(), first_frame.get_depth_image_path())
        self.fx, self.fy, self.cx, self.cy = camera.extract_intrinsic_projection_parameters(intrinsics_open3d_cpu)
        self.intrinsics_dict = camera.intrinsic_projection_parameters_as_dict(intrinsics_open3d_cpu)

        if verbosity_parameters.print_intrinsics.value:
            camera.print_intrinsic_projection_parameters(intrinsics_open3d_cpu)

        self.intrinsics_open3d_device = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix,
                                                        o3d.core.Dtype.Float32, self.device)
        self.extrinsics_open3d_device = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32, self.device)
        # TODO: initialize cropper here -- you can already get the first frame
        self.cropper: Union[None, StaticCenterCrop] = None
        # endregion

    def extract_and_warp_canonical_mesh_if_necessary(self, frame_index: int, graph: nnrt.geometry.GraphWarpField) -> \
            Tuple[Union[None, o3d.t.geometry.TriangleMesh], Union[None, o3d.t.geometry.TriangleMesh]]:
        mesh_extraction_weight_threshold = self.determine_mesh_extraction_threshold(frame_index)
        canonical_mesh: Union[None, o3d.t.geometry.TriangleMesh] = None
        if self.extracted_framewise_canonical_mesh_needed:
            canonical_mesh = self.volume.extract_surface_mesh(-1, mesh_extraction_weight_threshold)

        warped_mesh: Union[None, o3d.t.geometry.TriangleMesh] = None

        if self.framewise_warped_mesh_needed or self.frame_is_keyframe(frame_index):
            warped_mesh = graph.warp_mesh(canonical_mesh)

        return canonical_mesh, warped_mesh

    def determine_mesh_extraction_threshold(self, frame_index: int) -> int:
        frame_count = frame_index - self.start_at_frame_index
        tracking_parameters = Parameters.fusion.tracking
        if tracking_parameters.mesh_extraction_weight_thresholding_mode.value == \
                MeshExtractionWeightThresholdingMode.CONSTANT:
            return tracking_parameters.mesh_extraction_weight_threshold.value
        else:
            if frame_count < tracking_parameters.mesh_extraction_weight_threshold.value:
                return frame_count
            else:
                return tracking_parameters.mesh_extraction_weight_threshold.value

    def frame_is_keyframe(self, frame_index: int) -> bool:
        frame_count = frame_index - self.start_at_frame_index
        tracking_parameters = Parameters.fusion.tracking
        return tracking_parameters.tracking_span_mode.value == TrackingSpanMode.KEYFRAME_TO_CURRENT and \
               frame_count % tracking_parameters.keyframe_interval.value == 0

    def run(self) -> int:
        start_time = timeit.default_timer()

        verbosity_parameters = Parameters.fusion.telemetry.verbosity
        tracking_parameters = Parameters.fusion.tracking
        integration_parameters = Parameters.fusion.integration
        deform_net_parameters = Parameters.deform_net
        alignment_parameters = Parameters.alignment
        graph_parameters = Parameters.graph

        node_coverage = graph_parameters.node_coverage.value

        depth_scale = deform_net_parameters.depth_scale.value
        alignment_image_width = alignment_parameters.image_width.value
        alignment_image_height = alignment_parameters.image_height.value

        telemetry_generator = self.telemetry_generator
        device = self.device
        volume = self.volume

        sequence = self.sequence

        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        #####################################################################################################
        # region === initialize loop data structures ===
        #####################################################################################################

        renderer = PyTorch3DRenderer(self.sequence.resolution, device, self.intrinsics_open3d_device)

        saved_depth_image_np: Union[None, np.ndarray] = None
        saved_color_image_np: Union[None, np.ndarray] = None
        saved_mask_image_np: Union[None, np.ndarray] = None

        canonical_mesh: Union[None, o3d.geometry.TriangleMesh] = None
        warped_mesh: Union[None, o3d.geometry.TriangleMesh] = None
        keyframe_mesh: Union[None, o3d.geometry.TriangleMesh] = None

        saved_pixel_anchors: Union[None, o3d.core.Tensor] = None
        saved_pixel_weights: Union[None, o3d.core.Tensor] = None

        # process sequence start/end bound parameters
        check_for_end_frame = Parameters.fusion.input_data.run_until_frame.value != -1
        start_frame_index = sequence.start_frame_index
        if Parameters.fusion.input_data.start_at_frame.value != -1:
            start_frame_index = Parameters.fusion.input_data.start_at_frame.value
            sequence.advance_to_frame(start_frame_index)

        # save info into file in output in order to sync a frameviewer when viewing results in visualizer --
        # to observe both input and output
        self.telemetry_generator.save_info_for_frameviewer(sequence)

        while sequence.has_more_frames():
            current_frame = sequence.get_next_frame()
            if check_for_end_frame and current_frame.frame_index >= Parameters.fusion.input_data.run_until_frame.value:
                break
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

            current_frame_is_keyframe = self.frame_is_keyframe(current_frame.frame_index)

            # endregion
            if current_frame.frame_index == start_frame_index:
                # ====== initial (canonical) frame ======
                self.cropper = \
                    StaticCenterCrop(depth_image_np.shape[:2], (alignment_image_height, alignment_image_width))
                # region =============== FIRST FRAME PROCESSING / GRAPH INITIALIZATION ================================
                volume.integrate(depth_image_open3d, color_image_open3d, self.intrinsics_open3d_device,
                                 self.extrinsics_open3d_device, depth_scale, sequence.far_clipping_distance)
                # TODO: remove these calls after implementing proper block activation inside the IntegrateWarped____
                #  C++ functions
                # __DEBUG
                # volume.activate_sleeve_blocks()
                # volume.activate_sleeve_blocks()

                saved_pixel_anchors, saved_pixel_weights = \
                    self.initialize_graph_and_anchors(volume, device, sequence, current_frame.frame_index,
                                                      depth_image_np, mask_image_np)
                # cache first-frame images in every possible scenario
                saved_color_image_np = color_image_np
                saved_depth_image_np = depth_image_np
                saved_mask_image_np = mask_image_np

                # TODO: save initial meshes somehow specially maybe (line below will extract)?
                # canonical_mesh, warped_mesh = self.extract_and_warp_canonical_mesh_if_necessary()
                # endregion
            else:
                # ====== any subsequent frame ======
                #####################################################################################################
                # region ===== prepare source point cloud & RGB image for non-rigid alignment  ====
                #####################################################################################################
                #  when we track 0-to-t, we force reusing original frame for the source.
                if tracking_parameters.source_image_mode.value is SourceImageMode.IMAGE_ONLY:
                    source_depth = saved_depth_image_np
                    source_color = saved_color_image_np
                else:
                    # @formatter:off
                    source_mesh = warped_mesh \
                        if tracking_parameters.tracking_span_mode.value is TrackingSpanMode.PREVIOUS_TO_CURRENT else (
                        keyframe_mesh if tracking_parameters.tracking_span_mode.value is
                                         TrackingSpanMode.KEYFRAME_TO_CURRENT else canonical_mesh
                    )
                    # @formatter:on
                    source_depth, source_color = renderer.render_mesh_legacy(source_mesh, depth_scale=depth_scale)
                    source_depth = source_depth.astype(np.uint16)
                    telemetry_generator.process_rendering_result(source_color, source_depth, current_frame.frame_index)

                    # flip channels, i.e. RGB<-->BGR
                    source_color = cv2.cvtColor(source_color, cv2.COLOR_BGR2RGB)
                    if tracking_parameters.source_image_mode.value == \
                            SourceImageMode.RENDERED_WITH_PREVIOUS_FRAME_OVERLAY:
                        # re-use pixel data from previous frame
                        source_depth[saved_mask_image_np] = saved_depth_image_np[saved_mask_image_np]
                        source_color[saved_mask_image_np] = saved_color_image_np[saved_mask_image_np]

                source_point_image_np = image_processing.backproject_depth(source_depth, fx, fy, cx, cy,
                                                                           depth_scale=depth_scale)  # (h, w, 3)

                source_rgbxyz, _, cropper = DeformDataset.prepare_pytorch_input(
                    source_color, source_point_image_np, self.intrinsics_dict,
                    alignment_image_height, alignment_image_width
                )
                # endregion
                #####################################################################################################
                # region === prepare target point cloud, RGB image, normal map ====
                #####################################################################################################
                # TODO: replace options.depth_scale by a calibration/intrinsic property read from disk for each dataset,
                #  like InfiniTAM
                target_point_image = image_processing.backproject_depth(depth_image_np, fx, fy, cx, cy,
                                                                        depth_scale=depth_scale)  # (h, w, 3)
                target_rgbxyz, _, _ = DeformDataset.prepare_pytorch_input(
                    color_image_np, target_point_image, self.intrinsics_dict,
                    alignment_image_height, alignment_image_width, cropper=cropper
                )
                self.telemetry_generator.process_source_and_target_point_clouds(source_rgbxyz, target_rgbxyz)
                if device.get_type() == o3c.Device.CUDA:
                    target_normal_map = cuda_compute_normal(target_point_image)
                else:
                    target_normal_map = cpu_compute_normal(target_point_image)
                target_normal_map_o3d = o3c.Tensor(target_normal_map, dtype=o3c.Dtype.Float32, device=device)
                # endregion
                ########################################################################################################
                # region prepare pixel anchors, and pixel weights
                ########################################################################################################
                if current_frame_is_keyframe:
                    saved_pixel_anchors, saved_pixel_weights = self.compute_pixel_anchors(source_point_image_np, device)

                if tracking_parameters.tracking_span_mode.value is TrackingSpanMode.PREVIOUS_TO_CURRENT:
                    pixel_anchors, pixel_weights = \
                        self.compute_pixel_anchors(source_point_image_np, device)
                else:
                    pixel_anchors = saved_pixel_anchors
                    pixel_weights = saved_pixel_weights
                # endregion
                #####################################################################################################
                # region === adjust intrinsic / projection parameters due to cropping ====
                #####################################################################################################
                # TODO: we assume entire sequence has same resolution. Just like making cropper in another TODO,
                #   cropped intrinsics should be pre-computed in the __init__ method.
                fx_cropped, fy_cropped, cx_cropped, cy_cropped = image_processing.modify_intrinsics_due_to_cropping(
                    fx, fy, cx, cy, alignment_image_height, alignment_image_width, original_h=cropper.h,
                    original_w=cropper.w
                )
                cropped_intrinsics_numpy = np.array([fx_cropped, fy_cropped, cx_cropped, cy_cropped], dtype=np.float32)
                # endregion

                #####################################################################################################
                # region === run the motion prediction & optimization (non-rigid alignment) ====
                #####################################################################################################

                deform_net_data = run_non_rigid_alignment(self.deform_net, source_rgbxyz, target_rgbxyz, pixel_anchors,
                                                          pixel_weights, self.active_graph, cropped_intrinsics_numpy,
                                                          device)
                telemetry_generator.process_gn_point_clouds(deform_net_data["gn_point_clouds"])
                telemetry_generator.process_correspondences(deform_net_data["correspondence_info"],
                                                            deform_net_data["mask_pred"])

                # Get predicted node rotations & translations, reshape as needed (drop batch dimension)
                node_count = len(self.active_graph.nodes)
                rotations_pred = deform_net_data["node_rotations"].view(node_count, 3, 3)
                translations_pred = deform_net_data["node_translations"].view(node_count, 3)

                # endregion
                #####################################################################################################
                # region === fuse/integrate aligned data into the canonical/reference TSDF volume ====
                #####################################################################################################

                # use the resulting frame transformation predictions to update the global,
                # cumulative node transformations
                node_rotation_predictions = o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(rotations_pred))
                node_translation_predictions = o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(translations_pred))
                graph_for_integration = None
                if tracking_parameters.tracking_span_mode.value is TrackingSpanMode.FIRST_TO_CURRENT:
                    self.active_graph.rotations = node_rotation_predictions
                    self.active_graph.translations = node_translation_predictions
                    graph_for_integration = self.active_graph
                elif tracking_parameters.tracking_span_mode.value is TrackingSpanMode.PREVIOUS_TO_CURRENT:
                    self.active_graph.rotations = nnrt.core.matmul3d(self.active_graph.rotations,
                                                                     node_rotation_predictions)
                    self.active_graph.translations += o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(translations_pred))
                    graph_for_integration = self.active_graph
                elif tracking_parameters.tracking_span_mode.value is TrackingSpanMode.KEYFRAME_TO_CURRENT:
                    self.active_graph.rotations = node_rotation_predictions
                    self.active_graph.translations = node_translation_predictions
                    if len(self.keyframe_graphs) == 0:
                        graph_for_integration = self.active_graph
                    else:
                        graph_for_integration = self.keyframe_graphs[-1].clone()
                        graph_for_integration.rotations = \
                            nnrt.core.matmul3d(graph_for_integration.rotations, self.active_graph.rotations)
                        graph_for_integration.translations += self.active_graph.translations
                    if current_frame_is_keyframe:
                        self.active_graph = graph_for_integration.apply_transformations()
                        self.keyframe_graphs.append(graph_for_integration)

                # handle logging/vis of the graph data
                telemetry_generator.process_graph_transformation(graph_for_integration)

                cos_voxel_ray_to_normal = volume.integrate_warped(
                    depth_image_open3d, color_image_open3d, target_normal_map_o3d,
                    self.intrinsics_open3d_device, self.extrinsics_open3d_device, graph_for_integration,
                    depth_scale=depth_scale, depth_max=3.0)

                # TODO: not sure how the cos_voxel_ray_to_normal can be useful after the integrate_warped operation.
                #  Check BaldrLector's NeuralTracking fork code.
                # endregion
                #####################################################################################################
                canonical_mesh, warped_mesh = \
                    self.extract_and_warp_canonical_mesh_if_necessary(current_frame.frame_index, graph_for_integration)
                if current_frame_is_keyframe:
                    keyframe_mesh = warped_mesh

                telemetry_generator.process_result_visualization_and_logging(
                    canonical_mesh, warped_mesh,
                    deform_net_data,
                    alignment_image_height, alignment_image_width,
                    source_rgbxyz, target_rgbxyz,
                    pixel_anchors, pixel_weights,
                    graph_for_integration
                )
                telemetry_generator.record_meshes_to_disk_if_needed(canonical_mesh, warped_mesh)

            # ==== determine if we need to cache the current color & depth image
            # TODO: optimize by caching projected point cloud as well
            if tracking_parameters.tracking_span_mode.value == TrackingSpanMode.PREVIOUS_TO_CURRENT \
                    or current_frame_is_keyframe:
                saved_color_image_np = color_image_np
                saved_depth_image_np = depth_image_np
                saved_mask_image_np = mask_image_np

        if verbosity_parameters.print_total_runtime.value:
            end_time = timeit.default_timer()
            print(f"Total runtime (in seconds, with graph generation, "
                  f"without model + TSDF grid initialization): {end_time - start_time}")
        return PROGRAM_EXIT_SUCCESS

    def initialize_graph_and_anchors(self, volume: nnrt.geometry.WarpableTSDFVoxelGrid,
                                     device: o3c.Device, sequence: FrameSequenceDataset,
                                     frame_index: int,
                                     depth_image_np: numpy.ndarray, mask_image_np: numpy.ndarray) \
            -> Tuple[Union[None, o3c.Tensor], Union[None, o3c.Tensor]]:
        tracking_parameters = Parameters.fusion.tracking
        integration_parameters = Parameters.fusion.integration
        graph_parameters = Parameters.graph
        node_coverage = Parameters.graph.node_coverage.value
        deform_net_parameters = Parameters.deform_net
        precomputed_pixel_anchors = None
        precomputed_pixel_weights = None
        if tracking_parameters.graph_generation_mode.value == GraphGenerationMode.FIRST_FRAME_EXTRACTED_MESH:
            canonical_mesh_legacy: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(-1, 0).to_legacy()
            canonical_mesh = o3d.t.geometry.TrinagleMesh.from_legacy(canonical_mesh_legacy, device=device)
            self.active_graph = build_deformation_graph_from_mesh(
                canonical_mesh, node_coverage, erosion_iteration_count=10, neighbor_count=8,
                minimum_valid_anchor_count=integration_parameters.fusion_minimum_valid_anchor_count.value
            )
        elif tracking_parameters.graph_generation_mode.value == GraphGenerationMode.FIRST_FRAME_LOADED_GRAPH:
            self.active_graph = sequence.get_current_frame_graph_warp_field(device)
            if self.active_graph is None:
                raise ValueError(f"Could not load graph for frame {frame_index}.")
        elif tracking_parameters.graph_generation_mode.value == GraphGenerationMode.FIRST_FRAME_DEPTH_IMAGE:
            self.active_graph, _, precomputed_pixel_anchors, precomputed_pixel_weights = \
                build_graph_warp_field_from_depth_image(
                    depth_image_np, mask_image_np,
                    intrinsic_matrix=self.intrinsic_matrix_np, device=device,
                    max_triangle_distance=graph_parameters.graph_max_triangle_distance.value,
                    depth_scale_reciprocal=deform_net_parameters.depth_scale.value,
                    erosion_num_iterations=graph_parameters.graph_erosion_num_iterations.value,
                    erosion_min_neighbors=graph_parameters.graph_erosion_min_neighbors.value,
                    remove_nodes_with_too_few_neighbors=graph_parameters.graph_remove_nodes_with_too_few_neighbors.value,
                    use_only_valid_vertices=graph_parameters.graph_use_only_valid_vertices.value,
                    sample_random_shuffle=graph_parameters.graph_sample_random_shuffle.value,
                    neighbor_count=graph_parameters.graph_neighbor_count.value,
                    enforce_neighbor_count=graph_parameters.graph_enforce_neighbor_count.value,
                    node_coverage=node_coverage,
                    minimum_valid_anchor_count=integration_parameters.fusion_minimum_valid_anchor_count.value
                )
        else:
            raise NotImplementedError(
                f"graph generation mode {tracking_parameters.graph_generation_mode.value.name} not implemented.")

        if tracking_parameters.pixel_anchor_computation_mode.value == AnchorComputationMode.PRECOMPUTED:
            precomputed_pixel_anchors, precomputed_pixel_weights = sequence.get_current_pixel_anchors_and_weights()

        if tracking_parameters.tracking_span_mode.value != TrackingSpanMode.PREVIOUS_TO_CURRENT and \
                (precomputed_pixel_anchors is None or precomputed_pixel_weights is None):
            depth_scale = deform_net_parameters.depth_scale.value
            source_point_image = image_processing.backproject_depth(depth_image_np, self.fx, self.fy, self.cx, self.cy,
                                                                    depth_scale=depth_scale)  # (h, w, 3)
            return self.compute_pixel_anchors(source_point_image, device)

        precomputed_pixel_anchors = self.cropper(precomputed_pixel_anchors)
        precomputed_pixel_weights = self.cropper(precomputed_pixel_weights)
        return o3c.Tensor(precomputed_pixel_anchors, device=device), o3c.Tensor(precomputed_pixel_weights,
                                                                                device=device)

    def compute_pixel_anchors(self, source_point_image: np.ndarray, device: o3c.Device) \
            -> Tuple[o3c.Tensor, o3c.Tensor]:
        tracking_parameters = Parameters.fusion.tracking
        node_coverage = Parameters.graph.node_coverage.value
        source_point_image_o3d = o3c.Tensor(source_point_image, device=device)

        if tracking_parameters.pixel_anchor_computation_mode.value == AnchorComputationMode.EUCLIDEAN:
            pixel_anchors, pixel_weights = \
                nnrt.geometry.compute_anchors_and_weights_euclidean(
                    source_point_image_o3d, self.active_graph.nodes, 4, 0, node_coverage
                )
        elif tracking_parameters.pixel_anchor_computation_mode.value == AnchorComputationMode.SHORTEST_PATH:
            pixel_anchors, pixel_weights = \
                nnrt.geometry.compute_anchors_and_weights_shortest_path(
                    source_point_image_o3d, self.active_graph.nodes, self.active_graph.edges, 4, node_coverage
                )
        else:
            raise NotImplementedError(
                f"{AnchorComputationMode.__name__:s} '{tracking_parameters.pixel_anchor_computation_mode.name:s}' not "
                f"implemented for computation of pixel anchors & weights."
            )
        # adjust anchor & weight maps to alignment input resolution
        pixel_anchors = self.cropper(pixel_anchors)
        pixel_weights = self.cropper(pixel_weights)
        return pixel_anchors, pixel_weights

    @staticmethod
    def perform_complex_logic_parameter_validation():
        tracking_parameters = Parameters.fusion.tracking
        if tracking_parameters.pixel_anchor_computation_mode.value == AnchorComputationMode.PRECOMPUTED and \
                tracking_parameters.tracking_span_mode.value is not TrackingSpanMode.FIRST_TO_CURRENT:
            raise ValueError(f"Illegal value: {AnchorComputationMode.__name__:s} "
                             f"{AnchorComputationMode.PRECOMPUTED} for pixel anchors is only allowed when "
                             f"{TrackingSpanMode.__name__} is set to {TrackingSpanMode.FIRST_TO_CURRENT}")
