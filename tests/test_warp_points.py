#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 10/18/21.
#  Copyright (c) 2021 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
import numpy as np
from pathlib import Path
import open3d as o3d
import open3d.core as o3c

import image_processing
from data import DeformDataset, StaticCenterCrop
import nnrt.geometry as ng


def test_warp_points_cpp_vs_legacy():
    device = o3d.core.Device('cuda:0')

    test_data_dir = Path("test_data/point_warping")
    source_rgbxyz = np.load(test_data_dir / "000033_source_rgbxyz.npy")
    nodes = np.load(test_data_dir / "000033_nodes.npy")
    node_translations = np.load(test_data_dir / "000033_node_translations.npy")
    node_rotations = np.load(test_data_dir / "000033_node_rotations.npy")

    cropper = StaticCenterCrop((480, 640), (448, 640))

    pixel_anchors, pixel_weights = \
        DeformDataset.load_anchors_and_weights(str(test_data_dir / "pixel_anchors.bin"), str(test_data_dir / "pixel_weights.bin"), cropper)

    source_rgbxyz_reshaped = source_rgbxyz.T.reshape(-1, 448, 640)

    legacy_warped_points_reshaped = image_processing.warp_deform_3d(
        source_rgbxyz_reshaped, pixel_anchors, pixel_weights, nodes, node_rotations, node_translations
    )
    legacy_warped_points = legacy_warped_points_reshaped.reshape(3, -1).T

    pc_legacy = o3d.geometry.PointCloud()
    pc_legacy.points = o3d.utility.Vector3dVector(source_rgbxyz[:, 3:])
    pc_legacy.colors = o3d.utility.Vector3dVector(source_rgbxyz[:, :3])
    pc = o3d.t.geometry.PointCloud.from_legacy_pointcloud(pc_legacy, device=device)

    nodes_o3d = o3c.Tensor(nodes, dtype=o3c.Dtype.Float32, device=device)
    node_rotations_o3d = o3c.Tensor(node_rotations, dtype=o3c.Dtype.Float32, device=device)
    node_translations_o3d = o3c.Tensor(node_translations, dtype=o3c.Dtype.Float32, device=device)

    point_anchors_o3d = o3c.Tensor(pixel_anchors.reshape(-1, 4), dtype=o3c.Dtype.Int32, device=device)
    point_weights_o3d = o3c.Tensor(pixel_weights.reshape(-1, 4), dtype=o3c.Dtype.Float32, device=device)

    # warped_pc_o3d = ng.warp_point_cloud_mat(pc, nodes_o3d, node_rotations_o3d, node_translations_o3d, 4, 0.05)

    warped_pc_o3d = ng.warp_point_cloud_mat(pc, nodes_o3d, node_rotations_o3d, node_translations_o3d, point_anchors_o3d, point_weights_o3d)
    warped_points = warped_pc_o3d.point["points"].cpu().numpy()

    valid_point_count = len(np.nonzero(source_rgbxyz[:, 3:].sum(axis=1))[0])

    # print(source_rgbxyz[:10, 3:])
    # print(legacy_warped_points[:10])
    # print(warped_points[:10])
    distances = np.linalg.norm(warped_points - legacy_warped_points, axis=1)
    absolute_tolerance = 1e-6
    distances[distances < absolute_tolerance] = 0
    indices_of_unmatched_points = np.nonzero(distances)[0]
    indices_of_unmatched_pixels = np.unravel_index(indices_of_unmatched_points, (448, 640))
    indices_of_unmatched_pixels = np.vstack((indices_of_unmatched_pixels[0], indices_of_unmatched_pixels[1])).T
    print()
    print("indices_of_unmatched_pixels (:30)")
    print(indices_of_unmatched_pixels[:30])
    source_unmatched_points = source_rgbxyz[:, 3:][indices_of_unmatched_points]
    print("Source unmatched points XYZ (:10)")
    print(source_unmatched_points[:10])
    source_unmatched_colors = (source_rgbxyz[:, :3][indices_of_unmatched_points] * 255).astype(np.uint8)
    print("Source unmatched points RGB")
    print(source_unmatched_colors[:10])
    print("Warped unmatched points")
    print(warped_points[indices_of_unmatched_points][:10])
    print("GT warped unmatched points")
    print(legacy_warped_points[indices_of_unmatched_points][:10])

    print(distances[indices_of_unmatched_points][:10])
    print(indices_of_unmatched_points[:10])
    print("Unmatched point ratio:")
    print(len(indices_of_unmatched_points) / valid_point_count)

