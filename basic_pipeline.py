#!/usr/bin/python3

# experimental fusion pipeline based on original NNRT code
# Copyright 2021 Gregory Kramida

# stdlib
import sys

# 3rd-party
import open3d as o3d

# local
import options as opt
import nnrt
from data import camera
from data import *
from pipeline.renderer import Renderer
from utils import image, voxel_grid
from model.model import DeformNet
from model.default import load_default_nnrt_model
from pipeline import graph, renderer

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


def main() -> int:
    #####################################################################################################
    # region ==== options ====
    #####################################################################################################

    # We will overwrite the default value in options.py / settings.py
    opt.use_mask = True
    opt.gn_max_nodes = 1500

    # internal verbosity options
    print_frame_info = False
    print_intrinsics = False

    # TODO: should be part of dataset!
    input_image_size = (480, 640)
    # endregion
    #####################################################################################################
    # region === load model, configure device ===
    #####################################################################################################
    model: DeformNet = load_default_nnrt_model()
    device = o3d.core.Device('cuda:0')
    # endregion
    #####################################################################################################
    # region === dataset, intrinsics & extrinsics in various shapes, sizes, and colors ===
    #####################################################################################################

    sequence: FrameSequenceDataset = FrameSequencePreset.BERLIN_50.value
    first_frame = sequence.get_frame_at(0)

    intrinsics_open3d_cpu = camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(sequence.get_intrinsics_path(),
                                                                                         first_frame.get_depth_image_path())
    fx, fy, cx, cy = camera.extract_intrinsic_projection_parameters(intrinsics_open3d_cpu)
    intrinsics_dict = camera.intrinsic_projection_parameters_as_dict(intrinsics_open3d_cpu)
    if print_intrinsics:
        camera.print_intrinsic_projection_parameters(intrinsics_open3d_cpu)

    intrinsics_open3d_gpu = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix, o3d.core.Dtype.Float32, device)
    extrinsics_gpu = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32, device)
    # endregion
    #####################################################################################################
    # region === initialize data structures ===
    #####################################################################################################

    volume = voxel_grid.make_default_tsdf_voxel_grid(device)
    deformation_graph: typing.Union[graph.DeformationGraph, None] = None
    renderer = Renderer(input_image_size, device, intrinsics_open3d_gpu)

    for current_frame in sequence:
        #####################################################################################################
        # grab images, transfer to GPU versions for Open3D
        #####################################################################################################
        if print_frame_info:
            print("Processing frame:", current_frame.frame_index)
            print(current_frame.color_image_path)
            print(current_frame.depth_image_path)

        depth_image_open3d_legacy = o3d.io.read_image(current_frame.depth_image_path)
        depth_image_np = np.array(depth_image_open3d_legacy)
        depth_image_open3d = o3d.t.geometry.Image.from_legacy_image(depth_image_open3d_legacy, device=device)

        color_image_open3d_legacy = o3d.io.read_image(current_frame.color_image_path)
        color_image_np = np.array(color_image_open3d_legacy)
        color_image_open3d = o3d.t.geometry.Image.from_legacy_image(color_image_open3d_legacy, device=device)

        if current_frame.frame_index == 0:
            first_color_image = color_image_np
            first_depth_image = depth_image_np

            volume.integrate(depth_image_open3d, color_image_open3d, intrinsics_open3d_gpu, extrinsics_gpu, 1000.0, 3.0)
            mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
            mesh.compute_vertex_normals()

            # === Construct initial deformation graph
            deformation_graph = graph.build_deformation_graph_from_mesh(mesh, options.node_coverage)
        else:
            mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
            warped_mesh = deformation_graph.warp_mesh(mesh, options.node_coverage)

            rendered_depth, rendered_color = renderer.render_mesh(warped_mesh, depth_scale=options.depth_scale)

            source, _, cropper = DeformDataset.prepare_pytorch_input(
                rendered_color, rendered_depth, intrinsics_dict,
                opt.image_height, opt.image_width
            )
            target, _, _ = DeformDataset.prepare_pytorch_input(
                color_image_np, depth_image_np, intrinsics_dict,
                opt.image_height, opt.image_width,
                cropper=cropper
            )

            mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
            mesh.compute_vertex_normals()

            source_point_image = nnrt.backproject_depth_ushort(rendered_depth, fx, fy, cx, cy, options.depth_scale)
            vertex_positions = np.array(mesh.vertices)

            valid_nodes_mask = np.ones((vertex_positions.shape[0], 1), dtype=bool)
            pixel_anchors, pixel_weights = nnrt.compute_pixel_anchors_euclidean(
                deformation_graph.nodes, source_point_image,
                options.node_coverage)

            pixel_anchors = cropper(pixel_anchors)
            pixel_weights = cropper(pixel_weights)

            fx, fy, cx, cy = image.modify_intrinsics_due_to_cropping(
                intrinsics_dict['fx'], intrinsics_dict['fy'], intrinsics_dict['cx'], intrinsics_dict['cy'],
                opt.image_height, opt.image_width, original_h=cropper.h, original_w=cropper.w
            )
            adjusted_intrinsics_numpy = np.array([fx, fy, cx, cy], dtype=np.float32)

            # TODO: not sure how this is used yet
            num_nodes = np.array(deformation_graph.nodes.shape[0], dtype=np.int64)

            source_cuda = torch.from_numpy(source).cuda().unsqueeze(0)
            target_cuda = torch.from_numpy(target).cuda().unsqueeze(0)
            graph_nodes_cuda = torch.from_numpy(deformation_graph.nodes).cuda().unsqueeze(0)
            graph_edges_cuda = torch.from_numpy(deformation_graph.edges).cuda().unsqueeze(0)
            graph_edges_weights_cuda = torch.from_numpy(deformation_graph.edge_weights).cuda().unsqueeze(0)
            graph_clusters_cuda = torch.from_numpy(deformation_graph.clusters.reshape(-1, 1)).cuda().unsqueeze(0)
            pixel_anchors_cuda = torch.from_numpy(pixel_anchors).cuda().unsqueeze(0)
            pixel_weights_cuda = torch.from_numpy(pixel_weights).cuda().unsqueeze(0)
            intrinsics_cuda = torch.from_numpy(adjusted_intrinsics_numpy).cuda().unsqueeze(0)

            num_nodes_cuda = torch.from_numpy(num_nodes).cuda().unsqueeze(0)

            with torch.no_grad():
                model_data = model(
                    source_cuda, target_cuda,
                    graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
                    pixel_anchors_cuda, pixel_weights_cuda,
                    num_nodes_cuda, intrinsics_cuda,
                    evaluate=True, split="test"
                )

            # Get some of the results
            rotations_pred = model_data["node_rotations"].view(num_nodes, 3, 3).cpu().numpy()
            translations_pred = model_data["node_translations"].view(num_nodes, 3).cpu().numpy()

            # TODO: use rotations_pred and translations_pred to update graph!

            mask_pred = model_data["mask_pred"]
            assert mask_pred is not None, "Make sure use_mask=True in options.py"
            mask_pred = mask_pred.view(-1, opt.image_height, opt.image_width).cpu().numpy()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
