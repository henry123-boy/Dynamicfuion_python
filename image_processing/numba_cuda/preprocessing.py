import math
from numba import cuda
import numpy as np

from tsdf.numba_cuda.device_functions import cross, normalize


@cuda.jit()
def cuda_compute_normal_kernel(vertex_map, normal_map):
    y, x = cuda.grid(2)
    height, width = vertex_map.shape[:2]
    if y >= height or x >= width:
        return

    # @formatter:off
    left_x, left_y, left_z =    vertex_map[y, x - 1, 0], vertex_map[y, x - 1, 1], vertex_map[y, x - 1, 2]
    right_x, right_y, right_z = vertex_map[y, x + 1, 0], vertex_map[y, x + 1, 1], vertex_map[y, x + 1, 2]
    up_x, up_y, up_z =          vertex_map[y - 1, x, 0], vertex_map[y - 1, x, 1], vertex_map[y - 1, x, 2]
    down_x, down_y, down_z =    vertex_map[y + 1, x, 0], vertex_map[y + 1, x, 1], vertex_map[y + 1, x, 2]
    # @formatter:on

    if left_z == 0 or right_z == 0 or up_z == 0 or down_z == 0:
        normal_map[y, x, 0] = 0.0
        normal_map[y, x, 1] = 0.0
        normal_map[y, x, 2] = 0.0
    else:
        hor_x, hor_y, hor_z = right_x - left_x, right_y - left_y, right_z - left_z
        ver_x, ver_y, ver_z = up_x - down_x, up_y - down_y, up_z - down_z
        cx, cy, cz = cross(hor_x, hor_y, hor_z, ver_x, ver_y, ver_z)
        ncx, ncy, ncz = normalize(cx, cy, cz)
        if ncz > 0:
            normal_map[y, x, 0] = -ncx
            normal_map[y, x, 1] = -ncy
            normal_map[y, x, 2] = -ncz
        else:
            normal_map[y, x, 0] = ncx
            normal_map[y, x, 1] = ncy
            normal_map[y, x, 2] = ncz


def cuda_compute_normal(vertex_map):
    cuda_block_size = (16, 16)
    height, width = vertex_map.shape[:2]
    cuda_grid_size_y = math.ceil(height / cuda_block_size[0])
    cuda_grid_size_x = math.ceil(width / cuda_block_size[1])
    cuda_grid_size = (cuda_grid_size_y, cuda_grid_size_x)
    normal_map = np.zeros(shape=[height, width, 3], dtype=np.float32)
    cuda_compute_normal_kernel[cuda_grid_size, cuda_block_size](vertex_map, normal_map)
    cuda.synchronize()
    return normal_map
