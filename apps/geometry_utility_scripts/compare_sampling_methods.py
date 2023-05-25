#!/usr/bin/python
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
import os
import sys
from pathlib import Path

import nnrt.geometry
import open3d as o3d
import open3d.core as o3c
import ext_argparse

import image_processing
import numpy as np

import data.camera as camera
from data import StandaloneFrameDataset
from data.frame import DataSplit
from collections import namedtuple
from settings import process_arguments, Parameters, PathParameters
from warp_field.distance_matrix import compute_distance_matrix
from timeit import default_timer as timer


class SamplingMethodResults:
    def __init__(self):
        self.minimum_distances = []
        self.maximum_distances = []
        self.mean_distances = []
        self.median_distances = []
        self.distance_standard_deviations = []
        self.sample_point_counts = []
        self.runtimes = []

    def append_data(self, minimum_distance: float, maximum_distance: float, mean_distance: float,
                    median_distance: float, distance_standard_deviation: float, sample_point_count: int, runtime: float):
        self.minimum_distances.append(minimum_distance)
        self.maximum_distances.append(maximum_distance)
        self.mean_distances.append(mean_distance)
        self.median_distances.append(median_distance)
        self.distance_standard_deviations.append(distance_standard_deviation)
        self.sample_point_counts.append(sample_point_count)
        self.runtimes.append(runtime)


    def save_to_file(self, file: Path):
        np.savez(
            file,
            minimum_distances=np.array(self.minimum_distances),
            maximum_distances=np.array(self.maximum_distances),
            mean_distances=np.array(self.mean_distances),
            median_distances=np.array(self.median_distances),
            sample_point_counts=np.array(self.sample_point_counts),
            distance_standard_deviations=np.array(self.distance_standard_deviations),
            runtimes=np.array(self.runtimes)
        )

    def __repr__(self):
        return "Sampling method result: "


def process_sampling_result(node_coords: np.ndarray, runtime: float, result_accumulator: SamplingMethodResults):
    distances = compute_distance_matrix(node_coords, node_coords)
    distances[np.diag_indices_from(distances)] = 1000.
    minimum_distances: np.ndarray = distances.min(axis=0) # distances to nearest nodes
    result_accumulator.append_data(
        minimum_distance=minimum_distances.min(),
        maximum_distance=minimum_distances.max(),
        mean_distance=minimum_distances.mean(),
        median_distance=float(np.median(minimum_distances)),
        distance_standard_deviation = minimum_distances.std(),
        sample_point_count=len(node_coords),
        runtime=runtime
    )


def main():
    configuration_path = os.path.join(Path(__file__).parent.parent.parent.resolve(),
                                      f"configuration_files/nnrt_fusion_parameters.yaml")
    ext_argparse.process_settings_file(Parameters, configuration_path, generate_default_settings_if_missing=True)

    DOG_TRAINING_0 = StandaloneFrameDataset(0, 0, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    DOG_TRAINING_1 = StandaloneFrameDataset(0, 1, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    DOG_TRAINING_3 = StandaloneFrameDataset(0, 3, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    DOG_0 = StandaloneFrameDataset(0, 4, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    DOG_1 = StandaloneFrameDataset(0, 5, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    DOG_TREAT_0 = StandaloneFrameDataset(0, 6, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    DOG_2 = StandaloneFrameDataset(0, 7, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    DOG_TREAT_1 = StandaloneFrameDataset(0, 8, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    LITTLE_GIRL_0 = StandaloneFrameDataset(0, 10, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    LITTLE_GIRL_1 = StandaloneFrameDataset(0, 11, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    LITTLE_GIRL_2 = StandaloneFrameDataset(0, 12, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    LITTLE_GIRL_3 = StandaloneFrameDataset(0, 13, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    DOG_TREAT_2 = StandaloneFrameDataset(0, 14, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    WACK_0 = StandaloneFrameDataset(0, 16, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    NECK_PILLOW_0 = StandaloneFrameDataset(0, 18, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    BLUE_SWEATER_GUY_0 = StandaloneFrameDataset(0, 20, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    BACKPACK_0 = StandaloneFrameDataset(0, 22, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    BERLIN_0 = StandaloneFrameDataset(0, 70, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")
    BLUE_SHIRT_GUY_0 = StandaloneFrameDataset(0, 24, DataSplit.TRAIN, has_masks=False, masks_subfolder="sod")

    # all_frame_sets = [DOG_TRAINING_0, DOG_TRAINING_1, DOG_TRAINING_3, DOG_0, DOG_1, DOG_TREAT_0, DOG_2, DOG_TREAT_1,
    #                   LITTLE_GIRL_0, LITTLE_GIRL_1, LITTLE_GIRL_2, LITTLE_GIRL_3, DOG_TREAT_2, WACK_0, NECK_PILLOW_0,
    #                   BLUE_SWEATER_GUY_0, BACKPACK_0, BERLIN_0, BLUE_SHIRT_GUY_0]
    # all_frame_sets = [DOG_TRAINING_0, DOG_TRAINING_1, DOG_TRAINING_3, DOG_0, DOG_1, DOG_TREAT_0, DOG_2, WACK_0, NECK_PILLOW_0, BLUE_SWEATER_GUY_0, BERLIN_0]
    all_frame_sets = [DOG_TRAINING_0, BLUE_SWEATER_GUY_0, BERLIN_0]
    for set in all_frame_sets:
        set.load()

    vertex_counts = []

    sequential_epsilon_sampling_results = SamplingMethodResults()
    mean_grid_downsampling_results = SamplingMethodResults()
    fast_radius_mean_grid_downsampling_results = SamplingMethodResults()
    closest_to_mean_grid_subsampling_results = SamplingMethodResults()

    path_params = Parameters.path
    output_directory = Path(path_params.output_directory.value) / Path("sampling_method_comparison")
    output_directory.mkdir(parents=True, exist_ok=True)

    use_sequential_epsilon_sampling = False

    for frame_set in all_frame_sets:
        print(f"Processing frame {frame_set.frame_index} of sequence {frame_set.sequence_id}.")
        depth_image: o3d.t.geometry.Image = o3d.t.io.read_image(frame_set.get_depth_image_path())
        # mask_image: o3d.t.geometry.Image = o3d.t.io0.read_image(frame_set.get_mask_image_path())
        fx, fy, cx, cy = camera.load_intrinsic_matrix_entries_from_text_4x4_matrix(frame_set.get_intrinsics_path())

        # sequential epsilon sampling
        depth_image_np: np.ndarray = depth_image.as_tensor().numpy()
        depth_image_np.resize(depth_image_np.shape[0], depth_image_np.shape[1])
        # mask_image_np = mask_image.as_tensor().numpy()
        point_image = image_processing.backproject_depth(depth_image_np, fx, fy, cx, cy, depth_scale=1000.0)
        max_triangle_distance: float = 0.05

        vertices_np, vertex_pixels, faces = nnrt.compute_mesh_from_depth(point_image, max_triangle_distance)
        vertex_counts.append(len(vertices_np))
        print(f"Vertex count: {len(vertices_np)}.")
        non_eroded_vertices = nnrt.get_vertex_erosion_mask(vertices_np, faces, 4, 4)
        node_coverage = 0.05

        if use_sequential_epsilon_sampling:
            print("Sequential epsilon sampling...")
            start = timer()
            node_coords, node_point_indices = \
                nnrt.sample_nodes(vertices_np, non_eroded_vertices, node_coverage, False, False)
            end = timer()
            process_sampling_result(node_coords, end - start, sequential_epsilon_sampling_results)

        vertices = o3c.Tensor(vertices_np, device=o3c.Device("CUDA:0"))

        print("Mean grid downsampling...")
        start = timer()
        sampled_vertices_mean_grid = \
            nnrt.geometry.functional.mean_grid_downsample_3d_points(vertices, node_coverage * 1.5)
        end = timer()
        mean_grid_runtime = end - start
        process_sampling_result(sampled_vertices_mean_grid.cpu().numpy(), mean_grid_runtime,
                                mean_grid_downsampling_results)

        print("Fast mean radius downsampling...")
        start = timer()
        sampled_vertices_fast_mean_radius = \
            nnrt.geometry.functional.fast_mean_radius_downsample_3d_points(vertices, node_coverage * (2 / 3))
        end = timer()
        fast_mean_radius_runtime = end - start
        process_sampling_result(sampled_vertices_fast_mean_radius.cpu().numpy(), fast_mean_radius_runtime,
                                fast_radius_mean_grid_downsampling_results)

        print("Closest-to-mean-grid subsampling...")
        start = timer()
        sampled_indices_closest_to_mean_grid = \
            nnrt.geometry.functional.closest_to_mean_grid_subsample_3d_points(vertices, node_coverage * 1.5)
        end = timer()
        closest_to_mean_grid_runtime = end - start
        process_sampling_result(vertices_np[sampled_indices_closest_to_mean_grid.cpu().numpy()],
                                closest_to_mean_grid_runtime,
                                closest_to_mean_grid_subsampling_results)

    print("Done. Now saving results.")
    np.save(str(output_directory / "vertex_counts.npy"), np.array(vertex_counts))
    if use_sequential_epsilon_sampling:
        sequential_epsilon_sampling_results.save_to_file(output_directory / "sequential_epsilon_sampling_results.npz")

    mean_grid_downsampling_results.save_to_file(
        output_directory / "mean_grid_downsampling_results.npz")
    fast_radius_mean_grid_downsampling_results.save_to_file(
        output_directory / "fast_radius_mean_grid_downsampling_results.npz")
    closest_to_mean_grid_subsampling_results.save_to_file(
        output_directory / "closest_to_mean_grid_subsampling_results.npz")
    print("Saving results completed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
