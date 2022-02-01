import numba
from numba import cuda, float32, int32
import numpy as np
import math
import cmath


@cuda.jit(device=True)
def euclidean_distance(x1, y1, z1, x2, y2, z2):
    square_distance = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    distance = math.sqrt(square_distance)
    return distance


@cuda.jit(device=True)
def square_euclidean_distance(x1, y1, z1, x2, y2, z2):
    square_distance = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    return square_distance


@cuda.jit(device=True)
def warp_point_with_nodes(node_positions, nodes_rotation, nodes_translation, pos_x, pos_y, pos_z):
    now_x = pos_x - node_positions[0]
    now_y = pos_y - node_positions[1]
    now_z = pos_z - node_positions[2]

    now_x = nodes_rotation[0, 0] * now_x + \
            nodes_rotation[0, 1] * now_y + \
            nodes_rotation[0, 2] * now_z

    now_y = nodes_rotation[1, 0] * now_x + \
            nodes_rotation[1, 1] * now_y + \
            nodes_rotation[1, 2] * now_z

    now_z = nodes_rotation[2, 0] * now_x + \
            nodes_rotation[2, 1] * now_y + \
            nodes_rotation[2, 2] * now_z

    now_x = now_x + node_positions[0] + nodes_translation[0]
    now_y = now_y + node_positions[1] + nodes_translation[1]
    now_z = now_z + node_positions[2] + nodes_translation[2]

    return now_x, now_y, now_z


@cuda.jit(device=True)
def warp_normal_with_nodes(nodes_rotation, normal_x, normal_y, normal_z):
    now_x = nodes_rotation[0, 0] * normal_x + \
            nodes_rotation[0, 1] * normal_y + \
            nodes_rotation[0, 2] * normal_z

    now_y = nodes_rotation[1, 0] * normal_x + \
            nodes_rotation[1, 1] * normal_y + \
            nodes_rotation[1, 2] * normal_z

    now_z = nodes_rotation[2, 0] * normal_x + \
            nodes_rotation[2, 1] * normal_y + \
            nodes_rotation[2, 2] * normal_z
    return now_x, now_y, now_z


@cuda.jit(device=True)
def tsdf_bilinear_sample(data_volume, pos_x, pos_y, pos_z):
    x_up = int(math.ceil(pos_x))
    x_low = int(math.floor(pos_x))
    y_up = int(math.ceil(pos_y))
    y_low = int(math.floor(pos_y))
    z_up = int(math.ceil(pos_z))
    z_low = int(math.floor(pos_z))

    a_x = pos_x - x_low
    a_y = pos_y - y_low
    a_z = pos_z - z_low

    bilinear_sampled_tsdf = 0.0
    bilinear_sampled_weigth = 0.0

    weight_sum = 0.0
    valid_count = 0

    if data_volume[x_low, y_low, z_low, 1] > 0:
        weight_sum += (a_x) * (a_y) * (a_z)
        valid_count += 1

    if data_volume[x_up, y_low, z_low, 1] > 0:
        weight_sum += (1 - a_x) * (a_y) * (a_z)
        valid_count += 1

    if data_volume[x_low, y_up, z_low, 1] > 0:
        weight_sum += (a_x) * (1 - a_y) * (a_z)
        valid_count += 1

    if data_volume[x_low, y_low, z_up, 1] > 0:
        weight_sum += (a_x) * (a_y) * (1 - a_z)
        valid_count += 1

    if data_volume[x_up, y_up, z_low, 1] > 0:
        weight_sum += (1 - a_x) * (1 - a_y) * (a_z)
        valid_count += 1

    if data_volume[x_low, y_up, z_up, 1] > 0:
        weight_sum += (a_x) * (1 - a_y) * (1 - a_z)
        valid_count += 1

    if data_volume[x_up, y_low, z_up, 1] > 0:
        weight_sum += (1 - a_x) * (a_y) * (1 - a_z)
        valid_count += 1

    if data_volume[x_up, y_up, z_up, 1] > 0:
        weight_sum += (1 - a_x) * (1 - a_y) * (1 - a_z)
        valid_count += 1

    if weight_sum > 0 and valid_count > 4:
        if data_volume[x_low, y_low, z_low, 1] > 0:
            bilinear_sampled_tsdf += data_volume[x_low, y_low,
                                                 z_low, 0] * (a_x) * (a_y) * (a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if data_volume[x_up, y_low, z_low, 1] > 0:
            bilinear_sampled_tsdf += data_volume[x_up, y_low,
                                                 z_low, 0] * (1 - a_x) * (a_y) * (a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if data_volume[x_low, y_up, z_low, 1] > 0:
            bilinear_sampled_tsdf += data_volume[x_low, y_up,
                                                 z_low, 0] * (a_x) * (1 - a_y) * (a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if data_volume[x_low, y_low, z_up, 1] > 0:
            bilinear_sampled_tsdf += data_volume[x_low, y_low,
                                                 z_up, 0] * (a_x) * (a_y) * (1 - a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if data_volume[x_up, y_up, z_low, 1] > 0:
            bilinear_sampled_tsdf += data_volume[x_up, y_up,
                                                 z_low, 0] * (1 - a_x) * (1 - a_y) * (a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if data_volume[x_low, y_up, z_up, 1] > 0:
            bilinear_sampled_tsdf += data_volume[x_low, y_up,
                                                 z_up, 0] * (a_x) * (1 - a_y) * (1 - a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if data_volume[x_up, y_low, z_up, 1] > 0:
            bilinear_sampled_tsdf += data_volume[x_up, y_low,
                                                 z_up, 0] * (1 - a_x) * (a_y) * (1 - a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if data_volume[x_up, y_up, z_up, 1] > 0:
            bilinear_sampled_tsdf += data_volume[x_up, y_up,
                                                 z_up, 0] * (1 - a_x) * (1 - a_y) * (1 - a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        return bilinear_sampled_tsdf, bilinear_sampled_weigth
    else:
        return 32767, 0


@cuda.jit(device=True)
def tsdf_bounded_sample(data_volume, pos_x, pos_y, pos_z, min_tsdf):
    yta = 3
    x_up = int(math.ceil(pos_x))
    x_low = int(math.floor(pos_x))
    y_up = int(math.ceil(pos_y))
    y_low = int(math.floor(pos_y))
    z_up = int(math.ceil(pos_z))
    z_low = int(math.floor(pos_z))

    a_x = pos_x - x_low
    a_y = pos_y - y_low
    a_z = pos_z - z_low

    bilinear_sampled_tsdf = 0.0
    bilinear_sampled_weigth = 0.0

    weight_sum = 0.0
    valid_count = 0

    if abs(data_volume[x_low, y_low, z_low, 0] - min_tsdf) < yta:
        weight_sum += (a_x) * (a_y) * (a_z)
        valid_count += 1

    if abs(data_volume[x_up, y_low, z_low, 0] - min_tsdf) < yta:
        weight_sum += (1 - a_x) * (a_y) * (a_z)
        valid_count += 1

    if abs(data_volume[x_low, y_up, z_low, 0] - min_tsdf) < yta:
        weight_sum += (a_x) * (1 - a_y) * (a_z)
        valid_count += 1

    if abs(data_volume[x_low, y_low, z_up, 0] - min_tsdf) < yta:
        weight_sum += (a_x) * (a_y) * (1 - a_z)
        valid_count += 1

    if abs(data_volume[x_up, y_up, z_low, 0] - min_tsdf) < yta:
        weight_sum += (1 - a_x) * (1 - a_y) * (a_z)
        valid_count += 1

    if abs(data_volume[x_low, y_up, z_up, 0] - min_tsdf) < yta:
        weight_sum += (a_x) * (1 - a_y) * (1 - a_z)
        valid_count += 1

    if abs(data_volume[x_up, y_low, z_up, 0] - min_tsdf) < yta:
        weight_sum += (1 - a_x) * (a_y) * (1 - a_z)
        valid_count += 1

    if abs(data_volume[x_up, y_up, z_up, 0] - min_tsdf) < yta:
        weight_sum += (1 - a_x) * (1 - a_y) * (1 - a_z)
        valid_count += 1

    if valid_count > 0 and weight_sum > 0:
        if abs(data_volume[x_low, y_low, z_low, 0] - min_tsdf) < yta:
            bilinear_sampled_tsdf += data_volume[x_low, y_low,
                                                 z_low, 0] * (a_x) * (a_y) * (a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if abs(data_volume[x_up, y_low, z_low, 0] - min_tsdf) < yta:
            bilinear_sampled_tsdf += data_volume[x_up, y_low,
                                                 z_low, 0] * (1 - a_x) * (a_y) * (a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if abs(data_volume[x_low, y_up, z_low, 0] - min_tsdf) < yta:
            bilinear_sampled_tsdf += data_volume[x_low, y_up,
                                                 z_low, 0] * (a_x) * (1 - a_y) * (a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if abs(data_volume[x_low, y_low, z_up, 0] - min_tsdf) < yta:
            bilinear_sampled_tsdf += data_volume[x_low, y_low,
                                                 z_up, 0] * (a_x) * (a_y) * (1 - a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if abs(data_volume[x_up, y_up, z_low, 0] - min_tsdf) < yta:
            bilinear_sampled_tsdf += data_volume[x_up, y_up,
                                                 z_low, 0] * (1 - a_x) * (1 - a_y) * (a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if abs(data_volume[x_low, y_up, z_up, 0] - min_tsdf) < yta:
            bilinear_sampled_tsdf += data_volume[x_low, y_up,
                                                 z_up, 0] * (a_x) * (1 - a_y) * (1 - a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if abs(data_volume[x_up, y_low, z_up, 0] - min_tsdf) < yta:
            bilinear_sampled_tsdf += data_volume[x_up, y_low,
                                                 z_up, 0] * (1 - a_x) * (a_y) * (1 - a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        if abs(data_volume[x_up, y_up, z_up, 0] - min_tsdf) < yta:
            bilinear_sampled_tsdf += data_volume[x_up, y_up,
                                                 z_up, 0] * (1 - a_x) * (1 - a_y) * (1 - a_z) / weight_sum
            bilinear_sampled_weigth += data_volume[x_low, y_low,
                                                   z_low, 1] * (a_x) * (a_y) * (a_z) / weight_sum

        return bilinear_sampled_tsdf, bilinear_sampled_weigth
    else:
        return 32767, 0


@cuda.jit(device=True)
def tsdf_nearest_sample(data_volume, pos_x, pos_y, pos_z):
    x = int(round(pos_x))
    y = int(round(pos_y))
    z = int(round(pos_z))
    X_SIZE, Y_SIZE, Z_SIZE = data_volume.shape[:3]
    if x >= X_SIZE - 1 or y >= Y_SIZE - 1 or z >= Z_SIZE - 1:
        return 32767, 0
    else:
        return data_volume[x, y, z, 0], data_volume[x, y, z, 1]


@cuda.jit(device=True)
def tsdf_smallest_tsdf(data_volume, pos_x, pos_y, pos_z):
    min_tsdf = math.inf
    x_up = int(math.ceil(pos_x))
    x_low = int(math.floor(pos_x))
    y_up = int(math.ceil(pos_y))
    y_low = int(math.floor(pos_y))
    z_up = int(math.ceil(pos_z))
    z_low = int(math.floor(pos_z))
    min_tsdf = min(min_tsdf, data_volume[x_low, y_low, z_low, 0])
    min_tsdf = min(min_tsdf, data_volume[x_up, y_low, z_low, 0])
    min_tsdf = min(min_tsdf, data_volume[x_low, y_up, z_low, 0])
    min_tsdf = min(min_tsdf, data_volume[x_low, y_low, z_up, 0])
    min_tsdf = min(min_tsdf, data_volume[x_up, y_up, z_low, 0])
    min_tsdf = min(min_tsdf, data_volume[x_up, y_low, z_up, 0])
    min_tsdf = min(min_tsdf, data_volume[x_low, y_up, z_up, 0])
    min_tsdf = min(min_tsdf, data_volume[x_up, y_up, z_up, 0])
    return min_tsdf


@cuda.jit(device=True)
def tsdf_gradient_corrected_smaple(ref_volume, data_volume, volume_gradient, x, y, z,
                                   deoformed_vol_x, deoformed_vol_y, deoformed_vol_z):
    grad_x = volume_gradient[x, y, z, 0]
    grad_y = volume_gradient[x, y, z, 1]
    grad_z = volume_gradient[x, y, z, 2]

    ref_tsdf = ref_volume[x, y, z, 0]
    ref_weight = ref_volume[x, y, z, 1]


@cuda.jit(device=True)
def cross(x, y, z, x_, y_, z_):
    new_x = y * z_ - z * y_
    new_y = z * x_ - x * z_
    new_z = x * y_ - y * x_
    return new_x, new_y, new_z


@cuda.jit(device=True)
def dot(x, y, z, x_, y_, z_):
    s = x * x_ + y * y_ + z * z_
    return s


@cuda.jit(device=True)
def norm(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@cuda.jit(device=True)
def normalize(x, y, z):
    s = math.sqrt(x * x + y * y + z * z)
    return x / s, y / s, z / s


# region ================= vec3 =================================


@cuda.jit(device=True)
def vec3_cross(vec3_out, vec3_1, vec3_2):
    vec3_out[0] = vec3_1[1] * vec3_2[2] - vec3_1[2] * vec3_2[1]
    vec3_out[1] = vec3_1[2] * vec3_2[0] - vec3_1[0] * vec3_2[2]
    vec3_out[2] = vec3_1[0] * vec3_2[1] - vec3_1[1] * vec3_2[0]


@cuda.jit(device=True)
def vec3_dot(vec3_1, vec3_2):
    return vec3_1[0] * vec3_2[0] + vec3_1[1] * vec3_2[1] + vec3_1[2] * vec3_2[2]


@cuda.jit(device=True)
def vec3_elementwise_add(vec3_out, vec3_in):
    vec3_out[0] = vec3_out[0] + vec3_in[0]
    vec3_out[1] = vec3_out[1] + vec3_in[1]
    vec3_out[2] = vec3_out[2] + vec3_in[2]


@cuda.jit(device=True)
def vec3_elementwise_subtract(vec3_out, vec3_in):
    vec3_out[0] = vec3_out[0] - vec3_in[0]
    vec3_out[1] = vec3_out[1] - vec3_in[1]
    vec3_out[2] = vec3_out[2] - vec3_in[2]


@cuda.jit(device=True)
def vec3_elementwise_add_factor(vec3_out, vec3_in, factor):
    vec3_out[0] = vec3_out[0] + vec3_in[0] * factor
    vec3_out[1] = vec3_out[1] + vec3_in[1] * factor
    vec3_out[2] = vec3_out[2] + vec3_in[2] * factor


@cuda.jit(device=True)
def mat3_vec3_inner_product(vec3_out, mat3_in, vec3_in):
    vec3_out[0] = mat3_in[0, 0] * vec3_in[0] + mat3_in[0, 1] * vec3_in[1] + mat3_in[0, 2] * vec3_in[2]
    vec3_out[1] = mat3_in[1, 0] * vec3_in[0] + mat3_in[1, 1] * vec3_in[1] + mat3_in[1, 2] * vec3_in[2]
    vec3_out[2] = mat3_in[2, 0] * vec3_in[0] + mat3_in[2, 1] * vec3_in[1] + mat3_in[2, 2] * vec3_in[2]


# endregion
# region ================= vec4 =================================


@cuda.jit(device=True)
def vec4_elementwise_sub_factor(vec4_out, vec4_in, factor):
    vec4_out[0] = vec4_out[0] - vec4_in[0] * factor
    vec4_out[1] = vec4_out[1] - vec4_in[1] * factor
    vec4_out[2] = vec4_out[2] - vec4_in[2] * factor
    vec4_out[3] = vec4_out[3] - vec4_in[3] * factor


@cuda.jit(device=True)
def vec4_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


@cuda.jit(device=True)
def vec4_elementwise_mul(vec4_out, vec4_1, vect4_2):
    vec4_out[0] = vec4_1[0] * vect4_2[0]
    vec4_out[1] = vec4_1[1] * vect4_2[1]
    vec4_out[2] = vec4_1[2] * vect4_2[2]
    vec4_out[3] = vec4_1[3] * vect4_2[3]


@cuda.jit(device=True)
def vec4_elementwise_mul_factor(vec4_out, vec4_1, vec4_2, factor):
    vec4_out[0] = vec4_1[0] * vec4_2[0] * factor
    vec4_out[1] = vec4_1[1] * vec4_2[1] * factor
    vec4_out[2] = vec4_1[2] * vec4_2[2] * factor
    vec4_out[3] = vec4_1[3] * vec4_2[3] * factor


@cuda.jit(device=True)
def vec4_elementwise_add(vec4_out, vec4_1, vec4_2):
    vec4_out[0] = vec4_1[0] + vec4_2[0]
    vec4_out[1] = vec4_1[1] + vec4_2[1]
    vec4_out[2] = vec4_1[2] + vec4_2[2]
    vec4_out[3] = vec4_1[3] + vec4_2[3]


# endregion
# region ================= vec (arbitrary length) =================================
@cuda.jit(device=True)
def vec_elementwise_add_factor(vec_out, vec_in, factor):
    for i_element in range(vec_out.shape[0]):
        vec_out[i_element] = vec_out[i_element] + vec_in[i_element] * factor


@cuda.jit(device=True)
def vec_elementwise_add(result, a, b):
    for i_element in range(result.shape[0]):
        result[i_element] = a[i_element] + b[i_element]


@cuda.jit(device=True)
def vec_mul_factor(vec_out, factor):
    for i_element in range(vec_out.shape[0]):
        vec_out[i_element] = vec_out[i_element] * factor


# endregion
# region ==================== matrix blending ==========================================================================
@cuda.jit(device=True)
def linearly_blend_matrices(warped_point, temp1, temp2, source_point, nodes, node_translations, node_rotations, anchors,
                            weights, workload_index):
    # initialize
    for i_element in range(3):
        warped_point[i_element] = 0.0

    # add up node influences
    for i_anchor in range(anchors.shape[1]):
        anchor = anchors[workload_index, i_anchor]
        if anchor != -1:
            weight = weights[workload_index, i_anchor]
            node = nodes[anchor]  # vector3
            node_rotation = node_rotations[anchor]  # matrix 3x3
            node_translation = node_translations[anchor]  # vector3
            temp1[0] = source_point[0]
            temp1[1] = source_point[1]
            temp1[2] = source_point[2]
            vec3_elementwise_subtract(temp1, node)
            mat3_vec3_inner_product(temp2, node_rotation, temp1)
            vec3_elementwise_add(temp2, node)
            vec3_elementwise_add(temp2, node_translation)
            vec3_elementwise_add_factor(warped_point, temp2, weight)

    return warped_point
# endregion
