#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 5/23/23.
#  Copyright (c) 2023 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

import sys
import nnrt.geometry
import open3d as o3d
import open3d.core as o3c
import image_processing

import camera
from data import StandaloneFrameDataset
from frame import DataSplit
from collections import namedtuple


class SamplingResult:
    def __init__(self):
        self.minimum_distances = []
        self.sample_point_counts = []
        self.runtimes = []


def main():
    DOG_TRAINING_0 = StandaloneFrameDataset(0, 0, DataSplit.TRAIN, has_masks=True)
    DOG_TRAINING_1 = StandaloneFrameDataset(0, 1, DataSplit.TRAIN, has_masks=True)
    DOG_TRAINING_3 = StandaloneFrameDataset(0, 3, DataSplit.TRAIN, has_masks=True)
    DOG_0 = StandaloneFrameDataset(0, 4, DataSplit.TRAIN, has_masks=True)
    DOG_1 = StandaloneFrameDataset(0, 5, DataSplit.TRAIN, has_masks=True)
    DOG_TREAT_0 = StandaloneFrameDataset(0, 6, DataSplit.TRAIN, has_masks=True)
    DOG_2 = StandaloneFrameDataset(0, 7, DataSplit.TRAIN, has_masks=True)
    DOG_TREAT_1 = StandaloneFrameDataset(0, 8, DataSplit.TRAIN, has_masks=True)
    LITTLE_GIRL_0 = StandaloneFrameDataset(0, 10, DataSplit.TRAIN, has_masks=True)
    LITTLE_GIRL_1 = StandaloneFrameDataset(0, 11, DataSplit.TRAIN, has_masks=True)
    LITTLE_GIRL_2 = StandaloneFrameDataset(0, 12, DataSplit.TRAIN, has_masks=True)
    LITTLE_GIRL_3 = StandaloneFrameDataset(0, 13, DataSplit.TRAIN, has_masks=True)
    DOG_TREAT_2 = StandaloneFrameDataset(0, 14, DataSplit.TRAIN, has_masks=True)
    WACK_0 = StandaloneFrameDataset(0, 16, DataSplit.TRAIN, has_masks=True)
    NECK_PILLOW_0 = StandaloneFrameDataset(0, 18, DataSplit.TRAIN, has_masks=True)
    BLUE_SWEATER_GUY_0 = StandaloneFrameDataset(0, 20, DataSplit.TRAIN, has_masks=True)
    BACKPACK_0 = StandaloneFrameDataset(0, 22, DataSplit.TRAIN, has_masks=True)
    BERLIN_0 = StandaloneFrameDataset(0, 70, DataSplit.TRAIN, has_masks=True)
    BLUE_SHIRT_GUY_0 = StandaloneFrameDataset(0, 24, DataSplit.TRAIN, has_masks=True)

    all_frame_sets = [DOG_TRAINING_0, DOG_TRAINING_1, DOG_TRAINING_3, DOG_0, DOG_1, DOG_TREAT_0, DOG_2, DOG_TREAT_1,
                      LITTLE_GIRL_0, LITTLE_GIRL_1, LITTLE_GIRL_2, LITTLE_GIRL_3, DOG_TREAT_2, WACK_0, NECK_PILLOW_0,
                      BLUE_SWEATER_GUY_0, BACKPACK_0, BERLIN_0, BLUE_SHIRT_GUY_0]

    sequential_epsilon_sampling_result = SamplingResult()
    mean_grid_downsampling_result = SamplingResult()
    fast_radius_mean_grid_downsampling_result = SamplingResult()
    median_grid_downsampling_result = SamplingResult()
    fast_radius_median_grid_downsampling_result = SamplingResult()

    for frame_set in all_frame_sets:
        depth_image: o3d.t.geometry.Image = o3d.t.io.read_image(frame_set.get_depth_image_path())
        mask_image: o3d.t.geometry.Image = o3d.t.io.read_image(frame_set.get_mask_image_path())
        fx, fy, cx, cy = camera.load_intrinsic_matrix_entries_from_text_4x4_matrix(frame_set.get_intrinsics_path())

        # sequential epsilon sampling
        depth_image_np = depth_image.as_tensor().numpy()
        mask_image_np = mask_image.as_tensor().numpy()
        point_image = image_processing.backproject_depth(depth_image_np, fx, fy, cx, cy, depth_scale=1000.0)
        max_triangle_distance: float = 0.05

        vertices, vertex_pixels, faces = nnrt.compute_mesh_from_depth(point_image, max_triangle_distance)
        non_eroded_vertices = nnrt.get_vertex_erosion_mask(point_image, faces, 4, 4)
        node_coverage = 0.05

        node_coords, node_point_indices = nnrt.sample_nodes(point_image, non_eroded_vertices, node_coverage, True, False)


    return 0


if __name__ == "__main__":
    sys.exit(main())
