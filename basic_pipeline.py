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
from pynvml import *
from scipy.spatial.transform.rotation import Rotation

# local
import nnrt

import options
from data import camera
from data import *
from pipeline.numba_cuda.preprocessing import cuda_compute_normal
from pipeline.numpy_cpu.preprocessing import cpu_compute_normal
from pipeline.rendering.pytorch3d_renderer import PyTorch3DRenderer
import utils.image
import utils.voxel_grid
import utils.viz.tracking as tracking_viz
from model.model import DeformNet
from model.default import load_default_nnrt_model
from pipeline.graph import DeformationGraph, build_deformation_graph_from_mesh
import pipeline.pipeline_options as po
from pipeline.pipeline_options import VisualizationMode
from utils.viz.fusion_visualization_recorder import FusionVisualizationRecorder

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


def count_frames(directory: str, file_regex: re.Pattern) -> int:
    """
    Count frames of a sequence in the specified directory.
    Only files whose filenames match the file_regex are assumed to be frame images in a single image stream
    within the sequence.
    :param directory: the directory to check
    :param file_regex: regex for files belonging to a single image stream in the sequence
    :return: count of the specified files
    """
    count = 0
    for filename in os.listdir(directory):
        if file_regex.match(filename):
            count += 1

    return count


def reset(visualizer: o3d.pybind.visualization.VisualizerWithKeyCallback) -> None:
    view_control: o3d.visualization.ViewControl = visualizer.get_view_control()
    view_control.reset_camera_local_rotate()


def print_cuda_memory_info():
    device_handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(device_handle)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')


def main() -> int:
    # set up mesh recorder
    mesh_video_recorder = None
    if po.record_visualization_to_disk:
        # TODO fix recording, it currently doesn't seem to work. Perhaps use something other than Open3D for rendering,
        #  e.g. PyTorch3D, and ffmpeg-python for recording instead of OpenCV
        mesh_video_recorder = FusionVisualizationRecorder(
            output_video_path=os.path.join(options.output_directory, "mesh_visualization.mkv"),
            front=[0, 0, -1], lookat=[0, 0, 1.5],
            up=[0, -1.0, 0], zoom=0.7
        )

    if po.print_gpu_memory_info:
        nvmlInit()

    #####################################################################################################
    # region === load model, configure device ===
    #####################################################################################################
    model: DeformNet = load_default_nnrt_model()
    device = o3d.core.Device('cuda:0')
    # endregion
    #####################################################################################################
    # region === dataset, intrinsics & extrinsics in various shapes, sizes, and colors ===
    #####################################################################################################
    sequence: FrameSequenceDataset = po.sequence
    first_frame = sequence.get_frame_at(0)

    intrinsics_open3d_cpu = camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(sequence.get_intrinsics_path(),
                                                                                         first_frame.get_depth_image_path())
    fx, fy, cx, cy = camera.extract_intrinsic_projection_parameters(intrinsics_open3d_cpu)
    intrinsics_dict = camera.intrinsic_projection_parameters_as_dict(intrinsics_open3d_cpu)
    if po.print_intrinsics:
        camera.print_intrinsic_projection_parameters(intrinsics_open3d_cpu)

    intrinsics_open3d_cuda = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix, o3d.core.Dtype.Float32, device)
    extrinsics_open3d_cuda = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32, device)
    # endregion
    #####################################################################################################
    # region === initialize data structures ===
    #####################################################################################################

    volume = utils.voxel_grid.make_default_tsdf_voxel_grid(device)
    deformation_graph: typing.Union[DeformationGraph, None] = None
    renderer = PyTorch3DRenderer(sequence.resolution, device, intrinsics_open3d_cuda)

    previous_depth_image_np: typing.Union[None, np.ndarray] = None
    previous_color_image_np: typing.Union[None, np.ndarray] = None
    previous_mask_image_np: typing.Union[None, np.ndarray] = None

    while sequence.has_more_frames():
        current_frame = sequence.get_next_frame()
        #####################################################################################################
        # region ===== grab images, mask / clip if necessary, transfer to GPU versions for Open3D ===========
        #####################################################################################################
        if po.print_frame_info:
            print("Processing frame:", current_frame.frame_index)
            print("Color path:", current_frame.color_image_path)
            print("Depth path:", current_frame.depth_image_path)
        if po.print_gpu_memory_info:
            print_cuda_memory_info()

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
            volume.integrate(depth_image_open3d, color_image_open3d, intrinsics_open3d_cuda, extrinsics_open3d_cuda, options.depth_scale, 3.0)
            canonical_mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()

            # === Construct initial deformation graph
            deformation_graph = build_deformation_graph_from_mesh(canonical_mesh, options.node_coverage,
                                                                  erosion_iteration_count=10,
                                                                  neighbor_count=8)
        else:
            # TODO: try to speed up by using the extracted CUDA-based mesh directly (and converting to torch tensors via dlpack for rendering).
            #  Conversion to legacy mesh can be delegated to before visualization, and only we're trying to visualize one of these meshes
            #  The first step is to provide warping for the o3d.t.geometry.TriangleMesh (see graph.py).
            #  This may involve augmenting the Open3D extension in the local C++/CUDA code.
            canonical_mesh = None
            if po.source_image_mode != po.SourceImageMode.REUSE_PREVIOUS_FRAME or \
                    (po.visualization_mode == po.VisualizationMode.CANONICAL_MESH or
                     po.visualization_mode == po.VisualizationMode.WARPED_MESH):
                canonical_mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
                if po.visualization_mode == po.VisualizationMode.CANONICAL_MESH:
                    if po.record_visualization_to_disk:
                        mesh_video_recorder.capture_frame([canonical_mesh])
                    else:
                        o3d.visualization.draw_geometries([canonical_mesh],
                                                          front=[0, 0, -1],
                                                          lookat=[0, 0, 1.5],
                                                          up=[0, -1.0, 0],
                                                          zoom=0.7)
            warped_mesh = None
            # TODO: perform topological graph update
            if po.source_image_mode != po.SourceImageMode.REUSE_PREVIOUS_FRAME or \
                    po.visualization_mode == po.VisualizationMode.WARPED_MESH:
                if po.transformation_mode == po.TransformationMode.QUATERNIONS:
                    warped_mesh = deformation_graph.warp_mesh_dq_python(canonical_mesh, options.node_coverage)
                else:
                    warped_mesh = deformation_graph.warp_mesh_mat_python(canonical_mesh, options.node_coverage)
                if po.visualization_mode == po.VisualizationMode.WARPED_MESH:
                    if po.record_visualization_to_disk:
                        mesh_video_recorder.capture_frame([warped_mesh])
                    else:
                        o3d.visualization.draw_geometries([warped_mesh],
                                                          front=[0, 0, -1],
                                                          lookat=[0, 0, 1.5],
                                                          up=[0, -1.0, 0],
                                                          zoom=0.7)

            #####################################################################################################
            # region ===== prepare source point cloud & RGB image ====
            #####################################################################################################
            if po.source_image_mode == po.SourceImageMode.REUSE_PREVIOUS_FRAME:
                source_depth = previous_depth_image_np
                source_color = previous_color_image_np
            else:
                source_depth, source_color = renderer.render_mesh(warped_mesh, depth_scale=options.depth_scale)
                source_depth = source_depth.astype(np.uint16)

                # flip channels, i.e. RGB<-->BGR
                source_color = cv2.cvtColor(source_color, cv2.COLOR_BGR2RGB)
                if po.source_image_mode == po.SourceImageMode.RENDERED_WITH_PREVIOUS_FRAME_OVERLAY:
                    # re-use pixel data from previous frame
                    source_depth[previous_mask_image_np] = previous_depth_image_np[previous_mask_image_np]
                    source_color[previous_mask_image_np] = previous_color_image_np[previous_mask_image_np]

            source_point_image = image_utils.backproject_depth(source_depth, fx, fy, cx, cy, depth_scale=options.depth_scale)  # (h, w, 3)

            source_rgbxyz, _, cropper = DeformDataset.prepare_pytorch_input(
                source_color, source_point_image, intrinsics_dict,
                options.image_height, options.image_width
            )
            # endregion
            #####################################################################################################
            # region === prepare target point cloud, RGB image, normal map, pixel anchors, and pixel weights ====
            #####################################################################################################
            # TODO: replace options.depth_scale by a calibration property read from disk for each dataset
            target_point_image = image_utils.backproject_depth(depth_image_np, fx, fy, cx, cy, depth_scale=options.depth_scale)  # (h, w, 3)
            target_rgbxyz, _, _ = DeformDataset.prepare_pytorch_input(
                color_image_np, target_point_image, intrinsics_dict,
                options.image_height, options.image_width,
                cropper=cropper
            )
            if device.get_type() == o3c.Device.CUDA:
                target_normal_map = cuda_compute_normal(target_point_image)
            else:
                target_normal_map = cpu_compute_normal(target_point_image)
            target_normal_map_o3d = o3c.Tensor(target_normal_map, dtype=o3c.Dtype.Float32, device=device)

            pixel_anchors, pixel_weights = nnrt.compute_pixel_anchors_euclidean(
                deformation_graph.nodes, source_point_image,
                options.node_coverage)

            pixel_anchors = cropper(pixel_anchors)
            pixel_weights = cropper(pixel_weights)
            # endregion
            #####################################################################################################
            # region === adjust intrinsic / projection parameters due to cropping ====
            #####################################################################################################
            fx_cropped, fy_cropped, cx_cropped, cy_cropped = utils.image.modify_intrinsics_due_to_cropping(
                fx, fy, cx, cy, options.image_height, options.image_width, original_h=cropper.h, original_w=cropper.w
            )
            cropped_intrinsics_numpy = np.array([fx_cropped, fy_cropped, cx_cropped, cy_cropped], dtype=np.float32)
            # endregion
            #####################################################################################################
            # region === prepare pytorch inputs for the depth prediction model ====
            #####################################################################################################
            node_count = np.array(deformation_graph.nodes.shape[0], dtype=np.int64)
            source_cuda = torch.from_numpy(source_rgbxyz).cuda().unsqueeze(0)
            target_cuda = torch.from_numpy(target_rgbxyz).cuda().unsqueeze(0)
            warped_nodes = deformation_graph.get_warped_nodes()
            graph_nodes_cuda = torch.from_numpy(warped_nodes).cuda().unsqueeze(0)
            graph_edges_cuda = torch.from_numpy(deformation_graph.edges).cuda().unsqueeze(0)
            graph_edges_weights_cuda = torch.from_numpy(deformation_graph.edge_weights).cuda().unsqueeze(0)
            graph_clusters_cuda = torch.from_numpy(deformation_graph.clusters.reshape(-1, 1)).cuda().unsqueeze(0)
            pixel_anchors_cuda = torch.from_numpy(pixel_anchors).cuda().unsqueeze(0)
            pixel_weights_cuda = torch.from_numpy(pixel_weights).cuda().unsqueeze(0)
            intrinsics_cuda = torch.from_numpy(cropped_intrinsics_numpy).cuda().unsqueeze(0)
            num_nodes_cuda = torch.from_numpy(node_count).cuda().unsqueeze(0)
            # endregion
            #####################################################################################################
            # region === run the motion prediction & optimization ====
            #####################################################################################################
            with torch.no_grad():
                model_data = model(
                    source_cuda, target_cuda,
                    graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
                    pixel_anchors_cuda, pixel_weights_cuda,
                    num_nodes_cuda, intrinsics_cuda,
                    evaluate=True, split="test"
                )

            # Get some of the results
            rotations_pred = model_data["node_rotations"].view(node_count, 3, 3).cpu().numpy()
            translations_pred = model_data["node_translations"].view(node_count, 3).cpu().numpy()

            if po.visualization_mode == VisualizationMode.POINT_CLOUD_TRACKING:
                # TODO: not sure what the mask prediction can be useful for except in visualization so far...
                mask_pred = model_data["mask_pred"]
                assert mask_pred is not None, "Make sure use_mask=True in options.py"
                mask_pred = mask_pred.view(-1, options.image_height, options.image_width).cpu().numpy()
                # Compute mask gt for mask baseline
                _, source_points, valid_source_points, target_matches, valid_target_matches, valid_correspondences, _, _ \
                    = model_data["correspondence_info"]

                target_matches = target_matches.view(-1, options.image_height, options.image_width).cpu().numpy()
                valid_source_points = valid_source_points.view(-1, options.image_height, options.image_width).cpu().numpy()
                valid_correspondences = valid_correspondences.view(-1, options.image_height, options.image_width).cpu().numpy()

                tracking_viz.visualize_tracking(source_rgbxyz, target_rgbxyz, pixel_anchors, pixel_weights,
                                                warped_nodes, deformation_graph.edges,
                                                rotations_pred, translations_pred, mask_pred,
                                                valid_source_points, valid_correspondences, target_matches)




            # endregion
            #####################################################################################################
            # region === fuse model ====
            #####################################################################################################
            # use the resulting frame transformation predictions to update the global, cumulative node transformations

            for rotation, translation, i_node in zip(rotations_pred, translations_pred, np.arange(0, node_count)):
                node_position = deformation_graph.nodes[i_node]
                current_rotation = deformation_graph.rotations_mat[i_node] = \
                    rotation.dot(deformation_graph.rotations_mat[i_node])
                current_translation = deformation_graph.translations_vec[i_node] = \
                    translation + deformation_graph.translations_vec[i_node]

                translation_global = node_position + current_translation - current_rotation.dot(node_position)
                deformation_graph.transformations_dq[i_node] = dualquat(quat(current_rotation), translation_global)

            # prepare data for Open3D integration
            nodes_o3d = o3c.Tensor(deformation_graph.nodes, dtype=o3c.Dtype.Float32, device=device)

            if po.transformation_mode == po.TransformationMode.QUATERNIONS:
                # prepare data for Open3D integration
                node_dual_quaternions = np.array([np.concatenate((dq.real.data, dq.dual.data)) for dq in deformation_graph.transformations_dq])
                node_dual_quaternions_o3d = o3c.Tensor(node_dual_quaternions, dtype=o3c.Dtype.Float32, device=device)
                cos_voxel_ray_to_normal = volume.integrate_warped_dq(
                    depth_image_open3d, color_image_open3d, target_normal_map_o3d,
                    intrinsics_open3d_cuda, extrinsics_open3d_cuda,
                    nodes_o3d, node_dual_quaternions_o3d, options.node_coverage,
                    anchor_count=po.anchor_node_count, depth_scale=options.depth_scale, depth_max=3.0)
            elif po.transformation_mode == po.TransformationMode.MATRICES:
                node_rotations_o3d = o3c.Tensor(deformation_graph.rotations_mat, dtype=o3c.Dtype.Float32, device=device)
                node_translations_o3d = o3c.Tensor(deformation_graph.translations_vec, dtype=o3c.Dtype.Float32, device=device)
                cos_voxel_ray_to_normal = volume.integrate_warped_mat(
                    depth_image_open3d, color_image_open3d, target_normal_map_o3d,
                    intrinsics_open3d_cuda, extrinsics_open3d_cuda,
                    nodes_o3d, node_rotations_o3d, node_translations_o3d, options.node_coverage,
                    anchor_count=po.anchor_node_count, depth_scale=options.depth_scale, depth_max=3.0)
            else:
                raise ValueError("Unsupported motion blending mode")

            # endregion
            #####################################################################################################
            # TODO: not sure how the cos_voxel_ray_to_normal can be useful after the integrate_warped operation.
            #  Check BaldrLector's NeuralTracking fork code.
        previous_color_image_np = color_image_np
        previous_depth_image_np = depth_image_np
        previous_mask_image_np = mask_image_np

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Basic Fusion Pipeline based on Neural Non-Rigid Tracking + Fusion4D + Open3D spatial hashing")
    parser.add_argument("--profile", action='store_true')
    args = parser.parse_args()
    if args.profile:
        cProfile.run('main()')
    else:
        sys.exit(main())
