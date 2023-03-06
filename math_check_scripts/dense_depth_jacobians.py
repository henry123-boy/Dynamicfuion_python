#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 3/1/23.
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
import numpy as np
from collections import namedtuple

Intrinsics = namedtuple("Intrinsics", "fx fy cx cy")

PROGRAM_EXIT_SUCCESS = 0

face_vertex0 = np.array([0.0625, -0.0625, 1.2])
face_vertex1 = np.array([-0.0625, -0.0625, 1.2])
face_vertex2 = np.array([-0.0625, 0.0625, 1.2])

residual = 0.19999985

tested_ray_point = np.array([-0.0299999714, -0.0299999714])

barycentrics_perspective_distorted = np.array([0.356, 2.14576641e-08, 0.643999577])

intrinsics_ndc = Intrinsics(2.0, -2.0, 0.0, 0.0)

rendered_point_position = np.array([-0.02799999, -0.02799999, 1.3999996])
rendered_point_normal = np.array([0., 0., -0.99999964])
reference_point_position = np.array([-0.02399999, -0.02399999, 1.1999997])


def project_point(point: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    return np.array([point[0] * intrinsics.fx / point[2], point[1] * intrinsics.fy / point[2]])


def wedge(vector0: np.ndarray, vector1: np.ndarray) -> float:
    return np.cross(vector0, vector1) - np.cross(vector1, vector0)


def wedge2(vector0: np.ndarray, vector1: np.ndarray) -> float:
    return vector0[0] * vector1[1] - vector1[0] * vector0[1]


def compute_sp_area(vector0: np.ndarray, vector1: np.ndarray, vector2: np.ndarray) -> float:
    return wedge2(vector0 - vector1, vector1 - vector2)


def compute_d_total_fp_area_d_vertices(ndc_vertex0, ndc_vertex1, ndc_vertex2) -> np.ndarray:
    return np.array([
        [ndc_vertex1[1] - ndc_vertex2[1], ndc_vertex2[0] - ndc_vertex1[0]],
        [ndc_vertex2[1] - ndc_vertex0[1], ndc_vertex0[0] - ndc_vertex2[0]],
        [ndc_vertex0[1] - ndc_vertex1[1], ndc_vertex1[0] - ndc_vertex0[0]]
    ])


def compute_d_fp_area_d_vertices(ray_point: np.ndarray, ndc_vertex0: np.ndarray, ndc_vertex1: np.ndarray) -> np.ndarray:
    return np.array([
        [ndc_vertex1[1] - ray_point[1], ray_point[0] - ndc_vertex1[0]],
        [ray_point[1] - ndc_vertex0[1], ndc_vertex0[0] - ray_point[0]],
    ])


def jacobian_percpsective_projections_wrt_vertex(vertex: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    return np.array([
        [intrinsics.fx / vertex[2], 0., -intrinsics.fx * vertex[0] / vertex[2] ** 2],
        [0., intrinsics.fy / vertex[2], -intrinsics.fy * vertex[1] / vertex[2] ** 2],
    ])


def jacobian_barycentrics_perspective_distorted_wrt_vertices(vertex0, vertex1, vertex2, intrinsics, ray_point):
    ndc_vertex0 = project_point(vertex0, intrinsics)
    ndc_vertex1 = project_point(vertex1, intrinsics)
    ndc_vertex2 = project_point(vertex2, intrinsics)

    print("ndc vertices: ", ndc_vertex0, ndc_vertex1, ndc_vertex2)

    total_fp_area = compute_sp_area(ndc_vertex0, ndc_vertex1, ndc_vertex2)

    fp_area0 = compute_sp_area(ray_point, ndc_vertex1, ndc_vertex2)
    fp_area1 = compute_sp_area(ray_point, ndc_vertex2, ndc_vertex0)
    fp_area2 = compute_sp_area(ray_point, ndc_vertex0, ndc_vertex1)

    bary_distorted0 = fp_area0 / total_fp_area
    bary_distorted1 = fp_area1 / total_fp_area
    bary_distorted2 = fp_area2 / total_fp_area

    print("perspective-distorted barycentrics: ", bary_distorted0, bary_distorted1, bary_distorted2)

    d_total_fp_area_d_ndc_v0v1v2 = compute_d_total_fp_area_d_vertices(ndc_vertex0, ndc_vertex1, ndc_vertex2)
    d_fp_area0_d_ndc_v1v2 = compute_d_fp_area_d_vertices(ray_point, ndc_vertex1, ndc_vertex2)
    d_fp_area1_d_ndc_v2v0 = compute_d_fp_area_d_vertices(ray_point, ndc_vertex2, ndc_vertex0)
    d_fp_area2_d_ndc_v0v1 = compute_d_fp_area_d_vertices(ray_point, ndc_vertex0, ndc_vertex1)

    d_bary_distorted_d_ndc_v0 = np.zeros([3, 2])
    d_bary_distorted_d_ndc_v1 = np.zeros([3, 2])
    d_bary_distorted_d_ndc_v2 = np.zeros([3, 2])

    # d_bary_distorted0_d_v0v1v2 --> [1 x 6]
    d_bary_distorted_d_ndc_v0[0] = \
        (-fp_area0 * d_total_fp_area_d_ndc_v0v1v2[0]) / total_fp_area ** 2
    d_bary_distorted_d_ndc_v1[0] = \
        (total_fp_area * d_fp_area0_d_ndc_v1v2[0] - fp_area0 * d_total_fp_area_d_ndc_v0v1v2[1]) / total_fp_area ** 2
    d_bary_distorted_d_ndc_v2[0] = \
        (total_fp_area * d_fp_area0_d_ndc_v1v2[1] - fp_area0 * d_total_fp_area_d_ndc_v0v1v2[2]) / total_fp_area ** 2

    # d_bary_distorted1_d_v0v1v2 --> [1 x 6]
    d_bary_distorted_d_ndc_v0[1] = \
        (total_fp_area * d_fp_area1_d_ndc_v2v0[1] - fp_area1 * d_total_fp_area_d_ndc_v0v1v2[0]) / total_fp_area ** 2
    d_bary_distorted_d_ndc_v1[1] = \
        (-fp_area1 * d_total_fp_area_d_ndc_v0v1v2[1]) / total_fp_area ** 2
    d_bary_distorted_d_ndc_v2[1] = \
        (total_fp_area * d_fp_area1_d_ndc_v2v0[0] - fp_area1 * d_total_fp_area_d_ndc_v0v1v2[2]) / total_fp_area ** 2

    # d_bary_distorted2_d_v0v1v2 --> [1 x 6]
    d_bary_distorted_d_ndc_v0[2] = \
        (total_fp_area * d_fp_area2_d_ndc_v0v1[0] - fp_area2 * d_total_fp_area_d_ndc_v0v1v2[0]) / total_fp_area ** 2
    d_bary_distorted_d_ndc_v1[2] = \
        (total_fp_area * d_fp_area2_d_ndc_v0v1[1] - fp_area2 * d_total_fp_area_d_ndc_v0v1v2[1]) / total_fp_area ** 2
    d_bary_distorted_d_ndc_v2[2] = \
        (-fp_area2 * d_total_fp_area_d_ndc_v0v1v2[2]) / total_fp_area ** 2

    print(d_bary_distorted_d_ndc_v0, d_bary_distorted_d_ndc_v1, d_bary_distorted_d_ndc_v2, sep="\n")

    return [d_bary_distorted_d_ndc_v0, d_bary_distorted_d_ndc_v1, d_bary_distorted_d_ndc_v2]


def main():
    np.set_printoptions(suppress=True, linewidth=120)
    d_bary_distorted_d_ndc_v0, d_bary_distorted_d_ndc_v1, d_bary_distorted_d_ndc_v2 = \
        jacobian_barycentrics_perspective_distorted_wrt_vertices(face_vertex0, face_vertex1, face_vertex2,
                                                                 intrinsics_ndc, tested_ray_point)
    d_ndc_v0_d_v0 = jacobian_percpsective_projections_wrt_vertex(face_vertex0, intrinsics_ndc)
    d_ndc_v1_d_v1 = jacobian_percpsective_projections_wrt_vertex(face_vertex1, intrinsics_ndc)
    d_ndc_v2_d_v2 = jacobian_percpsective_projections_wrt_vertex(face_vertex2, intrinsics_ndc)
    print("Jacobians dndc dv:", d_ndc_v0_d_v0, d_ndc_v1_d_v1, d_ndc_v2_d_v2, sep="\n")

    d_bary_distorted_d_v0 = d_bary_distorted_d_ndc_v0.dot(d_ndc_v0_d_v0)
    d_bary_distorted_d_v1 = d_bary_distorted_d_ndc_v1.dot(d_ndc_v1_d_v1)
    d_bary_distorted_d_v2 = d_bary_distorted_d_ndc_v2.dot(d_ndc_v2_d_v2)

    d_bary_distorted_d_v0v1v2 = np.hstack((d_bary_distorted_d_v0, d_bary_distorted_d_v1, d_bary_distorted_d_v2))

    print()
    print("Jacobian dbary_dist dv:", d_bary_distorted_d_v0v1v2)

    d_residual_d_rendered_normal = rendered_point_position - reference_point_position  # 3x1
    d_residual_d_rendered_vector = rendered_point_normal  # 3x1


    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
