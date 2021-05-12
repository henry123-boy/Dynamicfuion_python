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
from utils import image, voxel_grid
from model.model import DeformNet
from model.default import load_default_nnrt_model
from pipeline import graph

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


def main() -> None:
    #####################################################################################################
    # Options
    #####################################################################################################

    # We will overwrite the default value in options.py / settings.py
    opt.use_mask = True
    opt.gn_max_nodes = 1500

    # internal verbosity options
    print_frame_info = False
    print_intrinsics = False

    #####################################################################################################
    # === load model, configure device ===
    #####################################################################################################

    model: DeformNet = load_default_nnrt_model()
    device = o3d.core.Device('cuda:0')

    #####################################################################################################
    # === dataset, intrinsics & extrinsics in various shapes, sizes, and colors ===
    #####################################################################################################

    frame_sequence: FrameSequenceDataset = FrameSequencePreset.RED_SHORTS_40.value
    first_frame = frame_sequence.get_frame_at(0)

    intrinsics_open3d_cpu = camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(frame_sequence.get_intrinsics_path(),
                                                                                         first_frame.get_depth_image_path())
    fx, fy, cx, cy = camera.extract_intrinsic_projection_parameters(intrinsics_open3d_cpu)
    intrinsics_dict = camera.intrinsic_projection_parameters_as_dict(intrinsics_open3d_cpu)
    if print_intrinsics:
        camera.print_intrinsic_projection_parameters(intrinsics_open3d_cpu)

    intrinsics_open3d_gpu = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix, o3d.core.Dtype.Float32, device)

    extrinsics_gpu = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32, device)

    volume = voxel_grid.make_default_tsdf_voxel_grid(device)

    # __DEBUG
    previous_color_image = None
    previous_depth_image = None
    first_color_image = None
    first_depth_image = None
    target_frame_index = 39

    deformation_graph: typing.Union[graph.DeformationGraph, None] = None

    for current_frame in frame_sequence:
        #####################################################################################################
        # grab images, transfer to GPU versions for Open3D
        #####################################################################################################
        if print_frame_info:
            print("Processing frame:", current_frame.frame_index)
            print(current_frame.color_image_path)
            print(current_frame.depth_image_path)

        depth_image_open3d_legacy = o3d.io.read_image(current_frame.depth_image_path)
        depth_image = np.array(depth_image_open3d_legacy)

        color_image_open3d_legacy = o3d.io.read_image(current_frame.color_image_path)
        color_image = np.array(color_image_open3d_legacy)

        if current_frame.frame_index == 0:
            first_color_image = color_image
            first_depth_image = depth_image
            depth_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(depth_image_open3d_legacy, device=device)
            color_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(color_image_open3d_legacy, device=device)
            volume.integrate(depth_image_gpu, color_image_gpu, intrinsics_open3d_gpu, extrinsics_gpu, 1000.0, 3.0)
            mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
            mesh.compute_vertex_normals()

            # === Construct initial deformation graph
            deformation_graph = graph.build_deformation_graph_from_mesh(mesh, options.node_coverage)

            # depth, point_image = nnrt.render_mesh(vertex_positions, face_indices, opt.image_width, opt.image_height,
            #                                       intrinsics_open3d_cpu.intrinsic_matrix, 1000.0)

            # uncomment to construct from image instead
            # deformation_graph = graph.build_deformation_graph_from_depth_image(depth_image, intrinsics_open3d_cpu, 16)

            # uncomment to visualize KNN graph with background image (no mesh) (deformation graph)
            # graph.draw_deformation_graph(deformation_graph, color_image_open3d_legacy)

            # uncomment to visualize isosurface + KNN graph
            # knn_graph = graph.knn_graph_to_line_set(canonical_node_positions, edges, clusters)
            # o3d.visualization.draw_geometries([mesh, knn_graph],
            #                                   front=[0, 0, -1],
            #                                   lookat=[0, 0, 1.5],
            #                                   up=[0, -1.0, 0],
            #                                   zoom=0.7)
        # else: # __DEBUG
        elif current_frame.frame_index < target_frame_index:
            pass
        elif current_frame.frame_index == target_frame_index:
            # TODO: replace source with deformed isosurface render
            source, _, cropper = DeformDataset.prepare_pytorch_input(
                first_color_image, first_depth_image, intrinsics_dict,
                opt.image_height, opt.image_width
            )
            target, _, _ = DeformDataset.prepare_pytorch_input(
                color_image, depth_image, intrinsics_dict,
                opt.image_height, opt.image_width,
                cropper=cropper
            )

            mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
            mesh.compute_vertex_normals()

            source_point_image = nnrt.backproject_depth_ushort(first_depth_image, fx, fy, cx, cy, 1000.0)
            vertex_positions = np.array(mesh.vertices)

            # TODO: this will all need to be replaced somehow by using the isosurface vertices / render process...
            #  before this, it can be optimized -- target_point_image is already computed inside prepare_pytorch_input,
            #  that can be further split into subroutines that isolate the depth projection
            #  Some toy programs were this were implemented in pipeline/rendering_test.py and pipeline/skinning_3d_test.py
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

            mask_pred = model_data["mask_pred"]
            assert mask_pred is not None, "Make sure use_mask=True in options.py"
            mask_pred = mask_pred.view(-1, opt.image_height, opt.image_width).cpu().numpy()
        else:  # __DEBUG
            break

        # __DEBUG these will be unneeded once the isosurface deformation & rendering are implemented
        previous_depth_image = depth_image
        previous_color_image = color_image

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
