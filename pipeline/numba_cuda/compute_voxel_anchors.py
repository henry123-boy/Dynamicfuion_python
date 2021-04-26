import numba
import numba.cuda as cuda
import numpy as np
import math
import cmath
import typing
from pipeline.numba_cuda.cuda_device_functions import *

GRAPH_K = 4


# Adapted from https://github.com/BaldrLector/NeuralTracking
@cuda.jit()
def cuda_compute_voxel_anchors_kernel(voxel_anchors, voxel_weights, graph_nodes,
                                      camera_rotation, camera_translation, voxel_size, node_coverage, volume_offset):
    workload_x, workload_y = cuda.grid(2)
    anchors_size_x, anchors_size_y, anchors_size_z = voxel_anchors.shape[:3]

    if workload_x >= anchors_size_x or workload_y >= anchors_size_y:
        return

    for z in range(anchors_size_z):
        voxel_x = (workload_x + 0.5) * voxel_size[0] + volume_offset[0]
        voxel_y = (workload_y + 0.5) * voxel_size[1] + volume_offset[1]
        voxel_z = (z + 0.5) * voxel_size[2] + volume_offset[2]

        voxel_depth_frame_z = camera_rotation[2, 0] * voxel_x + camera_rotation[2, 1] * voxel_y + \
                              camera_rotation[2, 2] * voxel_z + camera_translation[2]
        if voxel_depth_frame_z < 0:
            continue

        num_nodes = graph_nodes.shape[0]
        dist_array = cuda.local.array(shape=GRAPH_K, dtype=numba.types.float32)
        index_array = cuda.local.array(shape=GRAPH_K, dtype=numba.types.int32)
        for i_anchor in range(GRAPH_K):
            dist_array[i_anchor] = math.inf
            index_array[i_anchor] = -1

        # find nearest nodes ( without order )
        max_at_index = 0
        for node_index in range(num_nodes):
            new_distance = euclidean_distance(voxel_x, voxel_y, voxel_z,
                                              graph_nodes[node_index, 0],
                                              graph_nodes[node_index, 1],
                                              graph_nodes[node_index, 2])
            if new_distance < dist_array[max_at_index]:
                dist_array[max_at_index] = new_distance
                index_array[max_at_index] = node_index
                # update the maximum distance
                max_at_index = 0
                maximum_dist = dist_array[0]
                for j in range(1, GRAPH_K):
                    if dist_array[j] > maximum_dist:
                        max_at_index = j
                        maximum_dist = dist_array[j]

        anchor_count = 0
        weight_sum = 0
        for i_anchor in range(GRAPH_K):
            distance = dist_array[i_anchor]
            index = index_array[i_anchor]
            if distance > 2 * node_coverage:
                continue
            weight = math.exp(-math.pow(distance, 2) /
                              (2 * node_coverage * node_coverage))
            weight_sum += weight
            anchor_count += 1

            voxel_anchors[workload_x, workload_y, z, i_anchor] = index
            voxel_weights[workload_x, workload_y, z, i_anchor] = weight

        if weight_sum > 0:
            for i_anchor in range(GRAPH_K):
                voxel_weights[workload_x, workload_y, z, i_anchor] = voxel_weights[workload_x, workload_y, z, i_anchor] / weight_sum
        elif anchor_count > 0:
            for i_anchor in range(GRAPH_K):
                voxel_weights[workload_x, workload_y, z, i_anchor] = 1 / anchor_count


def cuda_compute_voxel_anchors(tsdf, graph_nodes,
                               camera_rotation, camera_translation, voxel_size, node_coverage, volume_offset):
    volume_resolution = tsdf.shape[:3]
    voxel_anchors = -np.ones(shape=list(volume_resolution) + [4], dtype=np.int32)
    voxel_weights = np.zeros(shape=list(volume_resolution) + [4], dtype=np.float32)
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(voxel_anchors.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(voxel_anchors.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    cuda_compute_voxel_anchors_kernel[blocks_per_grid, threads_per_block](voxel_anchors, voxel_weights, graph_nodes,
                                                                          camera_rotation, camera_translation, voxel_size,
                                                                          node_coverage, volume_offset)
    cuda.synchronize()
    return voxel_anchors, voxel_weights


@cuda.jit()
def cuda_compute_voxel_center_anchors_kernel(voxel_centers, voxel_center_anchors, voxel_center_weights, graph_nodes,
                                             camera_rotation, camera_translation, node_coverage):
    workload_index = cuda.grid(1)
    voxel_center_count = voxel_centers.shape[0]

    if workload_index >= voxel_center_count:
        return

    voxel_center = voxel_centers[workload_index]
    voxel_x, voxel_y, voxel_z = voxel_center

    voxel_depth_frame_z = camera_rotation[2, 0] * voxel_x + camera_rotation[2, 1] * voxel_y + \
                          camera_rotation[2, 2] * voxel_z + camera_translation[2]
    if voxel_depth_frame_z < 0:
        return

    node_coverage_squared = node_coverage ** 2

    num_nodes = graph_nodes.shape[0]
    distance_array = cuda.local.array(shape=GRAPH_K, dtype=numba.types.float32)
    index_array = cuda.local.array(shape=GRAPH_K, dtype=numba.types.int32)
    for i_anchor in range(GRAPH_K):
        distance_array[i_anchor] = math.inf
        index_array[i_anchor] = -1

    # find nearest nodes ( without order )
    max_at_index = 0
    for node_index in range(num_nodes):
        new_distance = euclidean_distance(voxel_x, voxel_y, voxel_z,
                                                 graph_nodes[node_index, 0],
                                                 graph_nodes[node_index, 1],
                                                 graph_nodes[node_index, 2])
        if new_distance < distance_array[max_at_index]:
            distance_array[max_at_index] = new_distance
            index_array[max_at_index] = node_index
            # update the maximum distance
            max_at_index = 0
            max_distance = distance_array[0]
            for distance_index in range(1, GRAPH_K):
                if distance_array[distance_index] > max_distance:
                    max_at_index = distance_index
                    max_distance = distance_array[distance_index]

    anchor_count = 0
    weight_sum = 0
    for i_anchor in range(GRAPH_K):
        distance = distance_array[i_anchor]
        index = index_array[i_anchor]
        if distance > 2 * node_coverage:
            continue
        weight = math.exp(-distance**2 / (2 * node_coverage * node_coverage))
        weight_sum += weight
        anchor_count += 1

        voxel_center_anchors[workload_index, i_anchor] = index
        voxel_center_weights[workload_index, i_anchor] = weight
        # voxel_center_weights[workload_index, i_anchor] = distance

    if weight_sum > 0:
        for i_anchor in range(GRAPH_K):
            voxel_center_weights[workload_index, i_anchor] /= weight_sum
            # voxel_center_weights[workload_index, i_anchor] = weight_sum
    elif anchor_count > 0:
        for i_anchor in range(GRAPH_K):
            voxel_center_weights[workload_index, i_anchor] = 1 / anchor_count


def cuda_compute_voxel_center_anchors(voxel_centers: np.ndarray, graph_nodes: np.ndarray,
                                      camera_rotation: np.ndarray, camera_translation: np.ndarray,
                                      node_coverage: float) -> typing.Tuple[np.ndarray, np.ndarray]:
    voxel_center_count = voxel_centers.shape[0]
    voxel_center_anchors = -np.ones(shape=(voxel_center_count, 4), dtype=np.int32)
    voxel_center_weights = np.zeros_like(voxel_center_anchors, dtype=np.float32)
    cuda_block_size = (256,)
    cuda_grid_size = (math.ceil(voxel_center_count / cuda_block_size[0]),)
    cuda_compute_voxel_center_anchors_kernel[cuda_grid_size, cuda_block_size](voxel_centers, voxel_center_anchors, voxel_center_weights,
                                                                              graph_nodes, camera_rotation, camera_translation, node_coverage)
    cuda.synchronize()
    return voxel_center_anchors, voxel_center_weights
