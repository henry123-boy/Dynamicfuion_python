import numba
import numba.cuda as cuda
import numpy as np
import math
import cmath
import typing
from tsdf.numba_cuda.device_functions import *

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
    weight_sum = 0.0
    for i_anchor in range(GRAPH_K):
        distance = distance_array[i_anchor]
        index = index_array[i_anchor]
        if distance > 2 * node_coverage:
            continue
        weight = math.exp(-distance ** 2 / (2 * node_coverage * node_coverage))
        weight_sum += weight
        anchor_count += 1.0

        voxel_center_anchors[workload_index, i_anchor] = index
        voxel_center_weights[workload_index, i_anchor] = weight

    if weight_sum > 0:
        for i_anchor in range(GRAPH_K):
            voxel_center_weights[workload_index, i_anchor] /= weight_sum

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


@cuda.jit()
def cuda_depth_warp_integrate_kernel(depth_image, depth_intrinsics, camera_rotation, camera_translation, cell_size, tsdf_volume,
                                     trunc_dist, voxel_anchors, voxel_weights, node_positions, node_rotations, node_translations,
                                     offset, normal_map, mask):
    workload_x, workload_y = cuda.grid(2)

    fx, fy, cx, cy = depth_intrinsics[0, 0], depth_intrinsics[1, 1], depth_intrinsics[0, 2], depth_intrinsics[1, 2]

    depth_image_height, depth_image_width = depth_image.shape[:2]
    volume_size_x, volume_size_y, volume_size_z = tsdf_volume.shape[:3]
    if workload_x >= volume_size_x or workload_y >= volume_size_y:
        return

    for z in range(volume_size_z):
        voxel_x = (workload_x + 0.5) * cell_size[0] + offset[0]
        voxel_y = (workload_y + 0.5) * cell_size[1] + offset[1]
        voxel_z = (z + 0.5) * cell_size[2] + offset[2]

        voxel_depth_frame_x = camera_rotation[0, 0] * voxel_x + \
                              camera_rotation[0, 1] * voxel_y + camera_rotation[0, 2] * voxel_z + camera_translation[0]
        voxel_depth_frame_y = camera_rotation[1, 0] * voxel_x + \
                              camera_rotation[1, 1] * voxel_y + camera_rotation[1, 2] * voxel_z + camera_translation[1]
        voxel_depth_frame_z = camera_rotation[2, 0] * voxel_x + \
                              camera_rotation[2, 1] * voxel_y + camera_rotation[2, 2] * voxel_z + camera_translation[2]

        point_is_valid = True
        invalid_count = 0
        for i in range(GRAPH_K):
            if voxel_anchors[workload_x, workload_y, z, i] == -1:
                invalid_count += 1
            if invalid_count > 1:
                point_is_valid = False
                break

        if not point_is_valid:
            continue
        else:
            deformed_pos_x = 0.0
            deformed_pos_y = 0.0
            deformed_pos_z = 0.0
            for i in range(GRAPH_K):
                if voxel_anchors[workload_x, workload_y, z, i] != -1:
                    new_x, new_y, new_z = warp_point_with_nodes(node_positions[voxel_anchors[workload_x, workload_y, z, i]],
                                                                node_rotations[voxel_anchors[workload_x, workload_y, z, i]],
                                                                node_translations[voxel_anchors[workload_x, workload_y, z, i]],
                                                                voxel_depth_frame_x, voxel_depth_frame_y, voxel_depth_frame_z)
                    deformed_pos_x += voxel_weights[workload_x, workload_y, z, i] * new_x
                    deformed_pos_y += voxel_weights[workload_x, workload_y, z, i] * new_y
                    deformed_pos_z += voxel_weights[workload_x, workload_y, z, i] * new_z

        if deformed_pos_z <= 0:
            continue

        du = int(round(fx * (deformed_pos_x / deformed_pos_z) + cx))
        dv = int(round(fy * (deformed_pos_y / deformed_pos_z) + cy))
        if 0 < du < depth_image_width and 0 < dv < depth_image_height:
            depth = depth_image[dv, du] / 1000.
            psdf = depth - deformed_pos_z

            view_direction_x, view_direction_y, view_direction_z = normalize(deformed_pos_x, deformed_pos_y, deformed_pos_z)
            view_direction_x, view_direction_y, view_direction_z = -view_direction_x, -view_direction_y, -view_direction_z
            dn_x, dn_y, dn_z = normal_map[dv, du, 0], normal_map[dv, du, 1], normal_map[dv, du, 2]
            cosine = dot(dn_x, dn_y, dn_z, view_direction_x, view_direction_y, view_direction_z)

            if depth > 0:
                mask[dv, du] = cosine
            if depth > 0 and psdf > -trunc_dist and cosine > 0.5:
                tsdf = min(1., psdf / trunc_dist)
                tsdf_prev, weight_prev = tsdf_volume[workload_x,
                                                     workload_y, z][0], tsdf_volume[workload_x, workload_y, z][1]
                weight_new = 1
                tsdf_new = (tsdf_prev * weight_prev +
                            weight_new * tsdf) / (weight_prev + weight_new)
                weight_new = min(weight_prev + weight_new, 255)
                # cut off weight at 255 -- if weight can be mate a field bigger than byte, the upper cap should be a parameter
                tsdf_volume[workload_x, workload_y, z][0], tsdf_volume[workload_x, workload_y, z][1] = tsdf_new, weight_new


def cuda_depth_warp_integrate(depth_frame, intrinsics, camera_rotation, camera_translation, cell_size, tsdf_volume, trunc_dist,
                              voxel_anchors, voxel_weights, node_positions, node_rotations, node_translations, volume_offset, norma_map,
                              mask=None):
    cuda_block_size = (16, 16)
    cuda_grid_size_x = math.ceil(voxel_anchors.shape[0] / cuda_block_size[0])
    cuda_grid_size_y = math.ceil(voxel_anchors.shape[1] / cuda_block_size[1])
    cuda_grid_size = (cuda_grid_size_x, cuda_grid_size_y)

    # in depth_integration v2.py (BaldrLector), mask is currently set to None
    if mask is None:
        mask = np.zeros_like(depth_frame, dtype=np.float32)
    else:
        mask = np.copy(mask)
    tsdf_volume = np.copy(tsdf_volume)

    cuda_depth_warp_integrate_kernel[cuda_grid_size, cuda_block_size](depth_frame, intrinsics, camera_rotation, camera_translation, cell_size,
                                                                      tsdf_volume, trunc_dist,
                                                                      voxel_anchors, voxel_weights, node_positions, node_rotations, node_translations,
                                                                      volume_offset, norma_map, mask)
    cuda.synchronize()
    return tsdf_volume, mask


@cuda.jit(device=True)
def cuda_warp_and_project_voxel_centers_mat(intrinsic_matrix, camera_rotation, camera_translation,
                                            voxel_centers, voxel_center_anchors, voxel_center_weights,
                                            nodes, node_translations, node_rotations, workload_index):
    voxel_center = voxel_centers[workload_index]
    voxel_x, voxel_y, voxel_z = voxel_center  # in meters

    # transform voxel from world to camera coordinates
    voxel_camera_z = camera_rotation[2, 0] * voxel_x + camera_rotation[2, 1] * voxel_y + camera_rotation[2, 2] * voxel_z + camera_translation[2]
    voxel_camera_x = camera_rotation[0, 0] * voxel_x + camera_rotation[0, 1] * voxel_y + camera_rotation[0, 2] * voxel_z + camera_translation[0]
    voxel_camera_y = camera_rotation[1, 0] * voxel_x + camera_rotation[1, 1] * voxel_y + camera_rotation[1, 2] * voxel_z + camera_translation[1]

    invalid_count = 0

    for anchor_index in range(GRAPH_K):
        if voxel_center_anchors[workload_index, anchor_index] == -1:
            invalid_count += 1
        if invalid_count > 1:
            return -1, -1, math.nan, math.nan, math.nan

    source_point = cuda.local.array(3, dtype=numba.types.float32)
    source_point[0], source_point[1], source_point[2] = voxel_camera_x, voxel_camera_y, voxel_camera_z
    warped_point = cuda.local.array(3, dtype=numba.types.float32)
    temp1 = cuda.local.array(3, dtype=numba.types.float32)
    temp2 = cuda.local.array(3, dtype=numba.types.float32)

    linearly_blend_matrices(warped_point, temp1, temp2, source_point, nodes, node_translations, node_rotations, voxel_center_anchors,
                            voxel_center_weights, workload_index)

    warped_voxel_x, warped_voxel_y, warped_voxel_z = warped_point

    if warped_voxel_z <= 0:
        return -1, -1, math.nan, math.nan, math.nan

    # project
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    du = int32(round(fx * (warped_voxel_x / warped_voxel_z) + cx))
    dv = int32(round(fy * (warped_voxel_y / warped_voxel_z) + cy))

    return du, dv, warped_voxel_x, warped_voxel_y, warped_voxel_z


@cuda.jit()
def cuda_update_warped_voxel_center_tsdf_and_weights_mat_kernel(weights_and_tsdf, cos_voxel_ray_to_normal, truncation_distance,
                                                                depth_image, normals, intrinsic_matrix, camera_rotation,
                                                                camera_translation, voxel_centers, voxel_center_anchors,
                                                                voxel_center_weights, nodes, node_translations, node_rotations):
    workload_index = cuda.grid(1)

    voxel_center_count = voxel_centers.shape[0]

    if workload_index >= voxel_center_count:
        return

    du, dv, warped_voxel_x, warped_voxel_y, warped_voxel_z = \
        cuda_warp_and_project_voxel_centers_mat(intrinsic_matrix, camera_rotation, camera_translation,
                                                voxel_centers, voxel_center_anchors, voxel_center_weights,
                                                nodes, node_translations, node_rotations, workload_index)
    if du < 0:
        return

    depth_image_height, depth_image_width = depth_image.shape[:2]

    if 0 < du < depth_image_width and 0 < dv < depth_image_height:
        depth = depth_image[dv, du] / 1000.
        psdf = depth - warped_voxel_z

        view_direction_x, view_direction_y, view_direction_z = normalize(warped_voxel_x, warped_voxel_y, warped_voxel_z)
        view_direction_x, view_direction_y, view_direction_z = -view_direction_x, -view_direction_y, -view_direction_z
        dn_x, dn_y, dn_z = normals[dv, du, 0], normals[dv, du, 1], normals[dv, du, 2]

        cosine = dot(dn_x, dn_y, dn_z, view_direction_x, view_direction_y, view_direction_z)

        cos_voxel_ray_to_normal[dv, du] = depth
        if depth > 0:
            cos_voxel_ray_to_normal[dv, du] = cosine
        if depth > 0 and psdf > -truncation_distance and cosine > 0.5:
            tsdf = min(1., psdf / truncation_distance)
            tsdf_prev, weight_prev = weights_and_tsdf[workload_index]
            weight_new = 1
            tsdf_new = (tsdf_prev * weight_prev + weight_new * tsdf) / (weight_prev + weight_new)
            weight_new = min(weight_prev + weight_new, 255)
            # cut off weight at 255 -- if weight can be mate a field bigger than byte, the upper cap should be a parameter
            weights_and_tsdf[workload_index] = tsdf_new, weight_new


def cuda_update_warped_voxel_center_tsdf_and_weights(
        weights_and_tsdf, truncation_distance, depth_image, normals,
        intrinsic_matrix, camera_rotation, camera_translation,
        voxel_centers, voxel_center_anchors, voxel_center_weights,
        nodes, node_translations, node_rotations
):
    voxel_center_count = voxel_centers.shape[0]
    cos_voxel_ray_to_normal = -np.ones(shape=depth_image.shape, dtype=np.float32)
    assert weights_and_tsdf.shape[0] == voxel_center_count

    cuda_block_size = (256,)
    cuda_grid_size = (math.ceil(voxel_center_count / cuda_block_size[0]),)

    cuda_update_warped_voxel_center_tsdf_and_weights_mat_kernel[cuda_grid_size, cuda_block_size](
        weights_and_tsdf, cos_voxel_ray_to_normal, truncation_distance,
        depth_image, normals, intrinsic_matrix, camera_rotation,
        camera_translation, voxel_centers, voxel_center_anchors,
        voxel_center_weights, nodes, node_translations, node_rotations)
    cuda.synchronize()

    return cos_voxel_ray_to_normal
