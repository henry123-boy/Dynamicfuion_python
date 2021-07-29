# A minimal example that generates a TSDF based on a depth image and then extracts the mesh from it
# Copyright 2021 Gregory Kramida
import os
import sys
import open3d as o3d
import numpy as np
import re

from data import camera
from data.presets import StandaloneFramePreset, StandaloneFrameDataset
from settings import settings_general
from fusion2.default_voxel_grid import make_default_tsdf_voxel_grid

PROGRAM_EXIT_SUCCESS = 0


def main():
    # Options

    use_mask = False
    segment = None

    #####################################################################################################
    # === open3d device, image paths, camera intrinsics ===
    #####################################################################################################

    # === device config ===

    device = o3d.core.Device('cuda:0')

    # === dataset ===
    # preset: StandaloneFramePreset = StandaloneFramePreset.BERLIN_0
    preset: StandaloneFramePreset = StandaloneFramePreset.RED_SHORTS_0

    dataset: StandaloneFrameDataset = preset.value
    dataset_name = preset.name
    # remove digit from end & make lowercase
    match = re.search(r'\w+(_\d+)', dataset_name)
    if match is not None:
        dataset_name = dataset_name[:-len(match.group(1))]
    dataset_name = dataset_name.lower()

    depth_image_path = dataset.get_depth_image_path()
    color_image_path = dataset.get_color_image_path()
    mask_image_path = dataset.get_mask_image_path()

    # === handle intrinsics ===

    depth_intrinsics_path = dataset.get_intrinsics_path()
    intrinsics_open3d_cpu, _ = camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(depth_intrinsics_path, depth_image_path)
    intrinsics_open3d_gpu = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix, o3d.core.Dtype.Float32, device)

    extrinsics_numpy = np.eye(4)
    extrinsics_open3d_gpu = o3d.core.Tensor(extrinsics_numpy, o3d.core.Dtype.Float32, device)

    volume = make_default_tsdf_voxel_grid(device)

    #####################################################################################################
    # === load images, fuse into TSDF, extract & visualize mesh ===
    #####################################################################################################

    # === images & TSDF integration/tsdf_management ===

    if use_mask:
        depth_image_numpy = np.array(o3d.io.read_image(depth_image_path))
        color_image_numpy = np.array(o3d.io.read_image(color_image_path))
        mask_image_numpy = np.array(o3d.io.read_image(mask_image_path))
        mask_image_numpy_color = np.dstack((mask_image_numpy, mask_image_numpy, mask_image_numpy)).astype(np.uint8)
        # apply mask
        depth_image_numpy &= mask_image_numpy
        color_image_numpy &= mask_image_numpy_color
        depth_image_open3d_legacy = o3d.geometry.Image(depth_image_numpy)
        color_image_open3d_legacy = o3d.geometry.Image(color_image_numpy)
    else:
        depth_image_open3d_legacy = o3d.io.read_image(depth_image_path)
        color_image_open3d_legacy = o3d.io.read_image(color_image_path)

    depth_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(depth_image_open3d_legacy, device=device)

    color_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(color_image_open3d_legacy, device=device)
    volume.integrate(depth_image_gpu, color_image_gpu, intrinsics_open3d_gpu, extrinsics_open3d_gpu, 1000.0, 3.0)

    # === mesh extraction ===

    mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(-1, 0).to_legacy_triangle_mesh()
    mesh.compute_vertex_normals()

    # === visualization ===

    o3d.visualization.draw_geometries([mesh],
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1.0, 0],
                                      zoom=0.7)
    o3d.io.write_triangle_mesh(os.path.join(settings_general.output_directory, f"mesh_{dataset.frame_index:06d}_{dataset_name}.ply"), mesh)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
