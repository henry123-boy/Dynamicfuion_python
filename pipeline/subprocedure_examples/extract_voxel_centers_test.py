# ==================================================================================================
# A toy code example that tests extracting the TSDF voxel centers from a TSDF
#
# Please run script from repository root, i.e.:
# python3 ./pipeline/extract_voxel_centers_test.py
#
# Copyright 2021 Gregory Kramida
# ==================================================================================================
import os
import sys
import open3d as o3d
import open3d.core as o3c
import numpy as np
import options

import nnrt

from data import camera
from utils.voxel_grid import make_default_tsdf_voxel_grid

PROGRAM_EXIT_SUCCESS = 0


def main():
    # Options

    use_mask = True
    segment = None

    #####################################################################################################
    # === open3d device, image paths, camera intrinsics, volume ===
    #####################################################################################################

    # === device config ===

    device = o3d.core.Device('cuda:0')

    # === compile image paths ===

    # TODO: instead of using real data, generate toy color & image data of a plane at a fixed distance from the camera
    sequence_number = 14
    frame_index = 200
    split = "val"
    frames_directory = os.path.join(options.dataset_base_directory, "{:s}/seq{:03d}/".format(split, sequence_number))
    color_image_filename_mask = frames_directory + "color/{:06d}.jpg"
    color_image_path = color_image_filename_mask.format(frame_index)
    depth_image_filename_mask = frames_directory + "depth/{:06d}.png"
    depth_image_path = depth_image_filename_mask.format(frame_index)

    mask_image_folder = frames_directory + "mask"
    if segment is None:
        segment = os.path.splitext(os.listdir(mask_image_folder)[0])[0].split('_')[1]
    mask_image_path = os.path.join(mask_image_folder, "{:06d}_{:s}.png".format(frame_index, segment))

    # === handle intrinsics ===

    depth_intrinsics_path = os.path.join(options.dataset_base_directory, "val/seq014/intrinsics.txt")
    intrinsics_open3d_cpu = camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(depth_intrinsics_path, depth_image_path)
    intrinsics_open3d_gpu = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix, o3d.core.Dtype.Float32, device)

    extrinsics_numpy = np.eye(4)
    extrinsics_open3d_gpu = o3d.core.Tensor(extrinsics_numpy, o3d.core.Dtype.Float32, device)

    # === open3d volume ===

    volume = make_default_tsdf_voxel_grid(device)

    #####################################################################################################
    # === load images, fuse into TSDF, extract & visualize mesh ===
    #####################################################################################################

    # === images & TSDF integration/fusion ===
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

    voxel_centers: o3c.Tensor = volume.extract_voxel_centers()
    voxel_centers_np = voxel_centers.cpu().numpy()

    print(voxel_centers_np)

    with open(os.path.join(options.output_directory, "voxel_centers_000200_red_shorts.np"), 'wb') as file:
        np.save(file, voxel_centers_np)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
