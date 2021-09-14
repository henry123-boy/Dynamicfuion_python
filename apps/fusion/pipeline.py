#!/usr/bin/python3

# experimental tsdf_management tsdf_management based on original NNRT code
# Copyright 2021 Gregory Kramida

from alignment.interface import run_non_rigid_alignment  # temporarily out-of-order here due to some CuPy 10 / CUDA 11.4 problems

# 3rd-party
import open3d.core as o3c
from dq3d import dualquat, quat

# local
import nnrt

from data import camera
from data import *
from image_processing.numba_cuda.preprocessing import cuda_compute_normal
from image_processing.numpy_cpu.preprocessing import cpu_compute_normal
from rendering.pytorch3d_renderer import PyTorch3DRenderer
import image_processing.image_processing2
import tsdf_management.default_voxel_grid as default_tsdf
from alignment.deform_net import DeformNet
from alignment.default import load_default_nnrt_network
from warp_field.graph import DeformationGraphNumpy, build_deformation_graph_from_mesh
import settings.settings_fusion as sf
from telemetry.telemetry_generator import TelemetryGenerator
from apps.create_graph_data import build_deformation_graph_from_depth_image

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


class FusionPipeline:
    def __init__(self):
        # === preprocess options & initialize telemetry ===
        self.extracted_framewise_canonical_mesh_needed = \
            sf.source_image_mode != sf.SourceImageMode.REUSE_PREVIOUS_FRAME or \
            sf.visualization_mode == sf.VisualizationMode.CANONICAL_MESH or \
            sf.visualization_mode == sf.VisualizationMode.WARPED_MESH or \
            sf.record_canonical_meshes_to_disk

        self.framewise_warped_mesh_needed = \
            sf.source_image_mode != sf.SourceImageMode.REUSE_PREVIOUS_FRAME or \
            sf.visualization_mode == sf.VisualizationMode.WARPED_MESH or \
            sf.record_warped_meshes_to_disk or sf.record_rendered_warped_mesh

        self.telemetry_generator = TelemetryGenerator(sf.record_visualization_to_disk,
                                                      sf.record_canonical_meshes_to_disk,
                                                      sf.record_warped_meshes_to_disk,
                                                      sf.record_rendered_warped_mesh,
                                                      sf.record_gn_point_clouds,
                                                      sf.record_source_and_target_point_clouds,
                                                      sf.print_cuda_memory_info,
                                                      sf.print_frame_info,
                                                      sf.visualization_mode, settings_general.output_directory)

        # === load alignment network, configure device ===
        self.deform_net: DeformNet = load_default_nnrt_network(o3c.Device.CUDA, self.telemetry_generator)
        self.device = o3d.core.Device('cuda:0')

        # === initialize structures ===
        self.graph: typing.Union[DeformationGraphNumpy, None] = None
        self.volume = default_tsdf.make_default_tsdf_voxel_grid(self.device)

        #####################################################################################################
        # region === dataset, intrinsics & extrinsics in various shapes, sizes, and colors ===
        #####################################################################################################
        self.sequence: FrameSequenceDataset = sf.sequence
        first_frame = self.sequence.get_frame_at(0)

        intrinsics_open3d_cpu, self.intrinsic_matrix_np = camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(
            self.sequence.get_intrinsics_path(), first_frame.get_depth_image_path())
        self.fx, self.fy, self.cx, self.cy = camera.extract_intrinsic_projection_parameters(intrinsics_open3d_cpu)
        self.intrinsics_dict = camera.intrinsic_projection_parameters_as_dict(intrinsics_open3d_cpu)

        if sf.print_intrinsics:
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
            canonical_mesh = self.volume.extract_surface_mesh(-1, 0).to_legacy_triangle_mesh()

        warped_mesh: typing.Union[None, o3d.geometry.TriangleMesh] = None
        # TODO: perform topological graph update
        if self.framewise_warped_mesh_needed:
            if sf.transformation_mode == sf.TransformationMode.QUATERNIONS:
                warped_mesh = self.graph.warp_mesh_dq(canonical_mesh, settings_general.node_coverage)
            else:
                warped_mesh = self.graph.warp_mesh_mat(canonical_mesh, settings_general.node_coverage)
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

        precomputed_anchors = None
        precomputed_weights = None

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
                volume.integrate(depth_image_open3d, color_image_open3d, self.intrinsics_open3d_device, self.extrinsics_open3d_device,
                                 settings_general.depth_scale, 3.0)
                # TODO: remove these calls after implementing proper block activation inside the IntegrateWarped____ C++ functions
                volume.activate_sleeve_blocks()
                volume.activate_sleeve_blocks()

                # === Construct initial deformation graph
                if sf.pixel_anchor_computation_mode == sf.AnchorComputationMode.PRECOMPUTED:
                    precomputed_anchors, precomputed_weights = sequence.get_current_pixel_anchors_and_weights()
                if sf.graph_generation_mode == sf.GraphGenerationMode.FIRST_FRAME_EXTRACTED_MESH:
                    canonical_mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(-1, 0).to_legacy_triangle_mesh()
                    self.graph = build_deformation_graph_from_mesh(canonical_mesh, settings_general.node_coverage,
                                                                   erosion_iteration_count=10,
                                                                   neighbor_count=8)
                elif sf.graph_generation_mode == sf.GraphGenerationMode.FIRST_FRAME_LOADED_GRAPH:
                    self.graph = sequence.get_current_frame_graph()
                    if self.graph is None:
                        raise ValueError(f"Could not load graph for frame {current_frame.frame_index}.")
                elif sf.graph_generation_mode == sf.GraphGenerationMode.FIRST_FRAME_DEPTH_IMAGE:
                    self.graph, _, precomputed_anchors, precomputed_weights = \
                        build_deformation_graph_from_depth_image(
                            depth_image_np, mask_image_np, intrinsic_matrix=self.intrinsic_matrix_np,
                            max_triangle_distance=settings_general.graph_max_triangle_distance, depth_scale_reciprocal=settings_general.depth_scale,
                            erosion_num_iterations=settings_general.graph_erosion_num_iterations,
                            erosion_min_neighbors=settings_general.graph_erosion_min_neighbors,
                            remove_nodes_with_too_few_neighbors=settings_general.graph_remove_nodes_with_too_few_neighbors,
                            use_only_valid_vertices=settings_general.graph_use_only_valid_vertices,
                            sample_random_shuffle=settings_general.graph_sample_random_shuffle,
                            neighbor_count=settings_general.graph_neighbor_count, enforce_neighbor_count=settings_general.graph_enforce_neighbor_count
                        )
                else:
                    raise NotImplementedError(f"graph generation mode {sf.graph_generation_mode.name} not implemented.")
                canonical_mesh, warped_mesh = self.extract_and_warp_canonical_mesh_if_necessary()
            else:

                #####################################################################################################
                # region ===== prepare source point cloud & RGB image ====
                #####################################################################################################
                #  when we track 0-to-t, we force reusing original frame for the source.
                if sf.source_image_mode == sf.SourceImageMode.REUSE_PREVIOUS_FRAME \
                        or sf.tracking_span_mode == sf.TrackingSpanMode.ZERO_TO_T:
                    source_depth = previous_depth_image_np
                    source_color = previous_color_image_np
                else:
                    source_depth, source_color = renderer.render_mesh(warped_mesh, depth_scale=settings_general.depth_scale)
                    source_depth = source_depth.astype(np.uint16)
                    telemetry_generator.process_rendering_result(source_color, source_depth, current_frame.frame_index)

                    # flip channels, i.e. RGB<-->BGR
                    source_color = cv2.cvtColor(source_color, cv2.COLOR_BGR2RGB)
                    if sf.source_image_mode == sf.SourceImageMode.RENDERED_WITH_PREVIOUS_FRAME_OVERLAY:
                        # re-use pixel data from previous frame
                        source_depth[previous_mask_image_np] = previous_depth_image_np[previous_mask_image_np]
                        source_color[previous_mask_image_np] = previous_color_image_np[previous_mask_image_np]

                source_point_image = image_utils.backproject_depth(source_depth, fx, fy, cx, cy,
                                                                   depth_scale=settings_general.depth_scale)  # (h, w, 3)

                source_rgbxyz, _, cropper = DeformDataset.prepare_pytorch_input(
                    source_color, source_point_image, self.intrinsics_dict,
                    settings_general.alignment_image_height, settings_general.alignment_image_width
                )
                # endregion
                #####################################################################################################
                # region === prepare target point cloud, RGB image, normal map, pixel anchors, and pixel weights ====
                #####################################################################################################
                # TODO: replace options.depth_scale by a calibration/intrinsic property read from disk for each dataset, like InfiniTAM
                target_point_image = image_utils.backproject_depth(depth_image_np, fx, fy, cx, cy,
                                                                   depth_scale=settings_general.depth_scale)  # (h, w, 3)
                target_rgbxyz, _, _ = DeformDataset.prepare_pytorch_input(
                    color_image_np, target_point_image, self.intrinsics_dict,
                    settings_general.alignment_image_height, settings_general.alignment_image_width,
                    cropper=cropper
                )
                self.telemetry_generator.process_source_and_target_point_clouds(source_rgbxyz, target_rgbxyz)
                if device.get_type() == o3c.Device.CUDA:
                    target_normal_map = cuda_compute_normal(target_point_image)
                else:
                    target_normal_map = cpu_compute_normal(target_point_image)
                target_normal_map_o3d = o3c.Tensor(target_normal_map, dtype=o3c.Dtype.Float32, device=device)

                if sf.pixel_anchor_computation_mode == sf.AnchorComputationMode.EUCLIDEAN:
                    pixel_anchors, pixel_weights = nnrt.compute_pixel_anchors_euclidean(
                        self.graph.nodes, source_point_image, settings_general.node_coverage
                    )
                elif sf.pixel_anchor_computation_mode == sf.AnchorComputationMode.SHORTEST_PATH:
                    pixel_anchors, pixel_weights = nnrt.compute_pixel_anchors_shortest_path(
                        source_point_image, self.graph.nodes, self.graph.edges, sf.anchor_node_count, settings_general.node_coverage
                    )
                elif sf.pixel_anchor_computation_mode == sf.AnchorComputationMode.PRECOMPUTED:
                    if sf.tracking_span_mode is not sf.TrackingSpanMode.ZERO_TO_T:
                        raise ValueError(f"Illegal value: {sf.AnchorComputationMode.__name__:s} "
                                         f"{sf.AnchorComputationMode.PRECOMPUTED} for pixel anchors is only allowed when "
                                         f"{sf.TrackingSpanMode.__name__} is set to {sf.TrackingSpanMode.ZERO_TO_T}")
                    pixel_anchors = precomputed_anchors
                    pixel_weights = precomputed_weights
                else:
                    raise NotImplementedError(
                        f"{sf.AnchorComputationMode.__name__:s} '{sf.pixel_anchor_computation_mode.name:s}' not "
                        f"implemented for pixel anchors."
                    )

                pixel_anchors = cropper(pixel_anchors)
                pixel_weights = cropper(pixel_weights)
                # endregion
                #####################################################################################################
                # region === adjust intrinsic / projection parameters due to cropping ====
                #####################################################################################################
                fx_cropped, fy_cropped, cx_cropped, cy_cropped = image_processing.image_processing2.modify_intrinsics_due_to_cropping(
                    fx, fy, cx, cy, settings_general.alignment_image_height, settings_general.alignment_image_width, original_h=cropper.h,
                    original_w=cropper.w
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
                if sf.tracking_span_mode is sf.TrackingSpanMode.ZERO_TO_T:
                    self.graph.rotations_mat = rotations_pred
                    self.graph.translations_vec = translations_pred

                for rotation, translation, i_node in zip(rotations_pred, translations_pred, np.arange(0, node_count)):
                    if sf.tracking_span_mode is sf.TrackingSpanMode.ZERO_TO_T:
                        self.graph.transformations_dq[i_node] = dualquat(quat(rotation), translation)
                    elif sf.tracking_span_mode is sf.TrackingSpanMode.T_MINUS_ONE_TO_T:
                        node_position = self.graph.nodes[i_node]
                        current_rotation = self.graph.rotations_mat[i_node] = \
                            rotation.dot(self.graph.rotations_mat[i_node])
                        current_translation = self.graph.translations_vec[i_node] = \
                            translation + self.graph.translations_vec[i_node]
                        translation_global = node_position + current_translation - current_rotation.dot(node_position)
                        self.graph.transformations_dq[i_node] = dualquat(quat(current_rotation), translation_global)
                    else:
                        raise NotImplementedError(f"{sf.TrackingSpanMode.__class__:s} {sf.tracking_span_mode.name:s} "
                                                  f" post-processing not implemented.")

                # prepare data for Open3D integration
                nodes_o3d = o3c.Tensor(self.graph.nodes, dtype=o3c.Dtype.Float32, device=device)
                edges_o3d = o3c.Tensor(self.graph.edges, dtype=o3c.Dtype.Int32, device=device)

                # TODO: to eliminate some of the indirection here, see TODO above DeformationGraphOpen3D. Might be able to
                #  condense the four variants here into a single function with two flag arguments controlling behavior.
                if sf.transformation_mode == sf.TransformationMode.QUATERNIONS:
                    # prepare quaternion data for Open3D integration
                    node_dual_quaternions = np.array([np.concatenate((dq.real.data, dq.dual.data)) for dq in self.graph.transformations_dq])
                    node_dual_quaternions_o3d = o3c.Tensor(node_dual_quaternions, dtype=o3c.Dtype.Float32, device=device)
                    if sf.voxel_anchor_computation_mode == sf.AnchorComputationMode.EUCLIDEAN:
                        cos_voxel_ray_to_normal = volume.integrate_warped_euclidean_dq(
                            depth_image_open3d, color_image_open3d, target_normal_map_o3d,
                            self.intrinsics_open3d_device, self.extrinsics_open3d_device,
                            nodes_o3d, node_dual_quaternions_o3d, settings_general.node_coverage,
                            anchor_count=sf.anchor_node_count, minimum_valid_anchor_count=sf.fusion_minimum_valid_anchor_count,
                            depth_scale=settings_general.depth_scale, depth_max=3.0)
                    elif sf.voxel_anchor_computation_mode == sf.AnchorComputationMode.SHORTEST_PATH:
                        cos_voxel_ray_to_normal = volume.integrate_warped_shortest_path_dq(
                            depth_image_open3d, color_image_open3d, target_normal_map_o3d,
                            self.intrinsics_open3d_device, self.extrinsics_open3d_device,
                            nodes_o3d, edges_o3d, node_dual_quaternions_o3d, settings_general.node_coverage,
                            anchor_count=sf.anchor_node_count, minimum_valid_anchor_count=sf.fusion_minimum_valid_anchor_count,
                            depth_scale=settings_general.depth_scale, depth_max=3.0)
                    else:
                        raise NotImplementedError(
                            f"{sf.AnchorComputationMode.__name__} '{sf.voxel_anchor_computation_mode.name:s}' not "
                            f"implemented for TSDF voxel anchors using {sf.transformation_mode.name:s} "
                            f"'{sf.TransformationMode.__name__}'")
                elif sf.transformation_mode == sf.TransformationMode.MATRICES:
                    node_rotations_o3d = o3c.Tensor(self.graph.rotations_mat, dtype=o3c.Dtype.Float32, device=device)
                    node_translations_o3d = o3c.Tensor(self.graph.translations_vec, dtype=o3c.Dtype.Float32, device=device)
                    if sf.voxel_anchor_computation_mode == sf.AnchorComputationMode.EUCLIDEAN:
                        cos_voxel_ray_to_normal = volume.integrate_warped_euclidean_mat(
                            depth_image_open3d, color_image_open3d, target_normal_map_o3d,
                            self.intrinsics_open3d_device, self.extrinsics_open3d_device,
                            nodes_o3d, node_rotations_o3d, node_translations_o3d, settings_general.node_coverage,
                            anchor_count=sf.anchor_node_count, minimum_valid_anchor_count=sf.fusion_minimum_valid_anchor_count,
                            depth_scale=settings_general.depth_scale, depth_max=3.0)
                    elif sf.voxel_anchor_computation_mode == sf.AnchorComputationMode.SHORTEST_PATH:
                        cos_voxel_ray_to_normal = volume.integrate_warped_shortest_path_mat(
                            depth_image_open3d, color_image_open3d, target_normal_map_o3d,
                            self.intrinsics_open3d_device, self.extrinsics_open3d_device,
                            nodes_o3d, edges_o3d, node_rotations_o3d, node_translations_o3d, settings_general.node_coverage,
                            anchor_count=sf.anchor_node_count, minimum_valid_anchor_count=sf.fusion_minimum_valid_anchor_count,
                            depth_scale=settings_general.depth_scale, depth_max=3.0)
                    else:
                        raise NotImplementedError(
                            f"{sf.AnchorComputationMode.__name__:s} '{sf.voxel_anchor_computation_mode.name:s}' not "
                            f"implemented for TSDF voxel anchors using {sf.transformation_mode.name:s} "
                            f"'{sf.TransformationMode.__name__:s}'")
                else:
                    raise NotImplementedError(
                        f"Unsupported {sf.TransformationMode.__name__:s} '{sf.voxel_anchor_computation_mode.name:s}' "
                        f"for motion blending of voxels during integration.")
                # TODO: not sure how the cos_voxel_ray_to_normal can be useful after the integrate_warped operation.
                #  Check BaldrLector's NeuralTracking fork code.
                # endregion
                #####################################################################################################

                telemetry_generator.process_result_visualization_and_logging(
                    canonical_mesh, warped_mesh,
                    deform_net_data,
                    settings_general.alignment_image_height,
                    settings_general.alignment_image_width,
                    source_rgbxyz, target_rgbxyz,
                    pixel_anchors, pixel_weights,
                    self.graph
                )
                canonical_mesh, warped_mesh = self.extract_and_warp_canonical_mesh_if_necessary()

            if sf.tracking_span_mode is not sf.TrackingSpanMode.ZERO_TO_T \
                    or current_frame.frame_index == sequence.start_frame_index:
                previous_color_image_np = color_image_np
                previous_depth_image_np = depth_image_np
                previous_mask_image_np = mask_image_np

        return PROGRAM_EXIT_SUCCESS
