import os
import sys
import math
import re
import numpy as np
import cv2
import open3d as o3d
import open3d.core as o3c
import nnrt
from scipy.spatial.transform.rotation import Rotation

import options
from pipeline.rendering.camera import Camera
from pipeline.graph import DeformationGraphNumpy, DeformationGraphOpen3D
from pipeline.rendering.pytorch3d_renderer import PyTorch3DRenderer, make_ndc_intrinsic_matrix


def main():
    # prepare folders
    root_output_directory = os.path.join(options.output_directory, "twisted_cube")
    if not os.path.exists(root_output_directory):
        os.makedirs(root_output_directory)
    depth_output_directory = os.path.join(root_output_directory, "depth")
    if not os.path.exists(depth_output_directory):
        os.makedirs(depth_output_directory)
    color_output_directory = os.path.join(root_output_directory, "color")
    if not os.path.exists(color_output_directory):
        os.makedirs(color_output_directory)

    # prepare geometry
    deepdeform_origin = np.array([0.0, 0.0, 2.8], dtype=np.float32)

    original_mesh = o3d.geometry.TriangleMesh.create_box()
    original_mesh.translate(deepdeform_origin.T, relative=False)
    original_mesh = original_mesh.subdivide_midpoint(number_of_iterations=3)

    # set vertex colors for texturing
    original_mesh.paint_uniform_color([1, 0.706, 0])

    nodes = np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.],
                      [1., 0., 1.],
                      [0., 1., 0.],
                      [1., 1., 0.],
                      [0., 1., 1.],
                      [1., 1., 1.]], dtype=np.float32)
    nodes_center = nodes.mean(axis=0)
    nodes_offset = nodes - nodes_center
    nodes = nodes_offset + deepdeform_origin
    nodes_center = deepdeform_origin
    # nodes += np.array([0.1, 0.2, 0.3])  # to make distances unique

    edges = np.array([[1, 2, 4],
                      [0, 5, 3],
                      [0, 3, 6],
                      [1, 2, 7],
                      [0, 5, 6],
                      [1, 4, 7],
                      [2, 4, 7],
                      [3, 5, 6]], dtype=np.int32)

    # copy to device
    device = o3c.Device("cuda:0")
    nodes_o3d = o3c.Tensor(nodes, device=device)
    edges_o3d = o3c.Tensor(edges, device=device)
    edge_weights_o3d = o3c.Tensor(np.array([1] * len(edges)), device=device)
    clusters_o3d = o3c.Tensor(np.array([0] * len(nodes)), device=device)

    # setup cameras and renderer
    camera = Camera(800, 800)
    intrinsics_open3d_cuda = o3c.Tensor(camera.get_intrinsic_matrix(), device=device)
    extrinsics_open3d_cuda = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32, device)

    renderer = PyTorch3DRenderer((camera.height, camera.width), device, intrinsics_open3d_cuda)

    intrinsics_file = os.path.join(root_output_directory, "intrinsics.txt")
    with open(intrinsics_file, 'w') as f:
        print(re.sub(r'[\[\]]', '', np.array_str(intrinsics_open3d_cuda.cpu().numpy())), file=f)

    # setup deformation graph and subroutine
    graph_open3d = DeformationGraphOpen3D(nodes_o3d, edges_o3d, edge_weights_o3d, clusters_o3d)

    def rotate_cube(current_mesh: o3d.geometry.TriangleMesh, angle: float) -> o3d.geometry.TriangleMesh:
        global_rotation_matrix_top = np.array(Rotation.from_euler('y', angle).as_matrix(), dtype=np.float32)
        global_rotation_matrix_bottom = np.array(Rotation.from_euler('y', -angle).as_matrix(), dtype=np.float32)

        nodes_rotated = nodes_center + np.concatenate(((nodes[:4] - nodes_center).dot(global_rotation_matrix_bottom),
                                                       (nodes[4:] - nodes_center).dot(global_rotation_matrix_top)),
                                                      axis=0)

        graph_open3d.rotations_mat = o3c.Tensor(np.stack([global_rotation_matrix_top] * 4 +
                                                         [global_rotation_matrix_bottom] * 4), device=device)
        translations = nodes_rotated - nodes
        graph_open3d.translations_vec = o3c.Tensor(translations, device=device)

        mesh = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(current_mesh, device=device)

        warped_mesh = graph_open3d.warp_mesh_mat(mesh, 0.5)
        warped_mesh_legacy: o3d.geometry.TriangleMesh = warped_mesh.to_legacy_triangle_mesh()
        warped_mesh_legacy.compute_vertex_normals()
        return warped_mesh_legacy

    # record animation rendering output
    frame_count = 10
    rotation_increment = 2.0  # degrees
    visualize_results = True
    rotated_mesh = original_mesh
    for i_frame in range(0, frame_count):
        i_angle = math.radians(rotation_increment)
        rotated_mesh = rotate_cube(rotated_mesh, i_angle)
        depth, color = renderer.render_mesh(rotated_mesh, depth_scale=1000.0, extrinsics=None)
        color_path = os.path.join(color_output_directory, f"{i_frame:06d}.jpg")
        depth_path = os.path.join(depth_output_directory, f"{i_frame:06d}.png")
        cv2.imwrite(color_path, color)
        cv2.imwrite(depth_path, depth.astype(np.uint16))
        if visualize_results:
            o3d.visualization.draw_geometries([rotated_mesh],
                                              zoom=0.8,
                                              front=[0, 0, -1],
                                              lookat=[0, 0, 0],
                                              up=[0, 1, 0])


if __name__ == "__main__":
    sys.exit(main())
