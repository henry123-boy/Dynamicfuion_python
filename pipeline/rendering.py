import open3d as o3d
import numpy as np


def render_mesh(mesh: o3d.geometry.TriangleMesh, intrinsics: o3d.camera.PinholeCameraIntrinsic):
    vertex_positions = np.array(mesh.vertices)
    triangle_vertex_indices = np.array(mesh.triangles)




