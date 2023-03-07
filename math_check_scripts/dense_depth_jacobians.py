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
from typing import Tuple

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


def jacobian_barycentrics_perspective_distorted_wrt_vertices(vertex0, vertex1, vertex2, intrinsics, ray_point) -> \
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
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

    return (d_bary_distorted_d_ndc_v0, d_bary_distorted_d_ndc_v1, d_bary_distorted_d_ndc_v2), np.array(
        [bary_distorted0, bary_distorted1, bary_distorted2])


def perspective_correct_barycentrics(barycentrics_distorted: np.ndarray, vertex0: np.ndarray, vertex1: np.ndarray,
                                     vertex2: np.ndarray) -> np.ndarray:
    rho0 = barycentrics_distorted[0]
    rho1 = barycentrics_distorted[1]
    rho2 = barycentrics_distorted[2]
    z0 = vertex0[2]
    z1 = vertex1[2]
    z2 = vertex2[2]

    bary0 = rho0 * z1 * z2
    bary1 = rho1 * z0 * z2
    bary2 = rho2 * z0 * z1

    return np.array([bary0, bary1, bary2]) / (bary0 + bary1 + bary2)


def compute_jacobian_perspective_correction(barycentrics_distorted: np.ndarray, vertex0: np.ndarray,
                                            vertex1: np.ndarray,
                                            vertex2: np.ndarray) -> np.ndarray:
    rho0 = barycentrics_distorted[0]
    rho1 = barycentrics_distorted[1]
    rho2 = barycentrics_distorted[2]
    z0 = vertex0[2]
    z1 = vertex1[2]
    z2 = vertex2[2]

    coord_0_numerator = rho0 * z1 * z2
    coord_1_numerator = rho1 * z0 * z2
    coord_2_numerator = rho2 * z0 * z1

    g = coord_0_numerator + coord_1_numerator + coord_2_numerator
    g_squared = g * g

    d_g_d_z0 = (rho1 * z2 + rho2 * z1)
    d_g_d_z1 = (rho0 * z2 + rho2 * z0)
    d_g_d_z2 = (rho0 * z1 + rho1 * z0)

    d_g_d_rho0 = (z1 * z2)
    d_g_d_rho1 = (z0 * z2)
    d_g_d_rho2 = (z0 * z1)

    # row 0
    d_bary0_d_rho0 = (g - coord_0_numerator) * d_g_d_rho0 / g_squared
    d_bary0_d_rho1 = (-coord_0_numerator) * d_g_d_rho1 / g_squared
    d_bary0_d_rho2 = (-coord_0_numerator) * d_g_d_rho2 / g_squared

    d_bary0_d_z0 = (- coord_0_numerator) * d_g_d_z0 / g_squared
    d_bary0_d_z1 = (g * rho0 * z2 - coord_0_numerator * d_g_d_z1) / g_squared
    d_bary0_d_z2 = (g * rho0 * z1 - coord_0_numerator * d_g_d_z2) / g_squared

    # row 1
    d_bary1_d_rho0 = (-coord_1_numerator) * d_g_d_rho0 / g_squared
    d_bary1_d_rho1 = (g - coord_1_numerator) * d_g_d_rho1 / g_squared
    d_bary1_d_rho2 = (-coord_1_numerator) * d_g_d_rho2 / g_squared

    d_bary1_d_z0 = (g * rho1 * z2 - coord_1_numerator * d_g_d_z0) / g_squared
    d_bary1_d_z1 = (- coord_1_numerator * d_g_d_z1) / g_squared
    d_bary1_d_z2 = (g * rho1 * z0 - coord_1_numerator * d_g_d_z2) / g_squared

    # row 2
    d_bary2_d_rho0 = (-coord_2_numerator) * d_g_d_rho0 / g_squared
    d_bary2_d_rho1 = (-coord_2_numerator) * d_g_d_rho1 / g_squared
    d_bary2_d_rho2 = (g - coord_2_numerator) * d_g_d_rho2 / g_squared

    d_bary2_d_z0 = (g * rho2 * z1 - coord_2_numerator * d_g_d_z0) / g_squared
    d_bary2_d_z1 = (g * rho2 * z0 - coord_2_numerator * d_g_d_z1) / g_squared
    d_bary2_d_z2 = (- coord_2_numerator * d_g_d_z2) / g_squared

    return np.array([[d_bary0_d_rho0, d_bary0_d_rho1, d_bary0_d_rho2, d_bary0_d_z0, d_bary0_d_z1, d_bary0_d_z2],
                     [d_bary1_d_rho0, d_bary1_d_rho1, d_bary1_d_rho2, d_bary1_d_z0, d_bary1_d_z1, d_bary1_d_z2],
                     [d_bary2_d_rho0, d_bary2_d_rho1, d_bary2_d_rho2, d_bary2_d_z0, d_bary2_d_z1, d_bary2_d_z2]])


def jacobian_barycentrics_corrected_wrt_vertices(d_bary_distorted_d_v0v1v2: np.ndarray, d_bary_corrected_d_bary_distorted_and_z):
    d_z_d_v0v1v2 = np.kron(np.eye(3), np.array([[0, 0, 1]]))
    d_bary_distorted_and_z_d_v0v1v2 = np.vstack((d_bary_distorted_d_v0v1v2, d_z_d_v0v1v2))
    return d_bary_corrected_d_bary_distorted_and_z.dot(d_bary_distorted_and_z_d_v0v1v2)


def main():
    np.set_printoptions(suppress=True, linewidth=120)
    (d_bary_distorted_d_ndc_v0, d_bary_distorted_d_ndc_v1, d_bary_distorted_d_ndc_v2), bary_distorted = \
        jacobian_barycentrics_perspective_distorted_wrt_vertices(face_vertex0, face_vertex1, face_vertex2,
                                                                 intrinsics_ndc, tested_ray_point)

    print("Distorted barycentrics:", bary_distorted, sep="\n")
    print()
    print("d_bary d ndc 0, 1, and 2:", d_bary_distorted_d_ndc_v0, d_bary_distorted_d_ndc_v1, d_bary_distorted_d_ndc_v2,
          sep="\n")
    print()

    d_ndc_v0_d_v0 = jacobian_percpsective_projections_wrt_vertex(face_vertex0, intrinsics_ndc)
    d_ndc_v1_d_v1 = jacobian_percpsective_projections_wrt_vertex(face_vertex1, intrinsics_ndc)
    d_ndc_v2_d_v2 = jacobian_percpsective_projections_wrt_vertex(face_vertex2, intrinsics_ndc)
    print("Jacobians dndc dV:", d_ndc_v0_d_v0, d_ndc_v1_d_v1, d_ndc_v2_d_v2, sep="\n")
    print()
    d_bary_distorted_d_v0 = d_bary_distorted_d_ndc_v0.dot(d_ndc_v0_d_v0)
    d_bary_distorted_d_v1 = d_bary_distorted_d_ndc_v1.dot(d_ndc_v1_d_v1)
    d_bary_distorted_d_v2 = d_bary_distorted_d_ndc_v2.dot(d_ndc_v2_d_v2)

    d_bary_distorted_d_v0v1v2 = np.hstack((d_bary_distorted_d_v0, d_bary_distorted_d_v1, d_bary_distorted_d_v2))

    print("Jacobian dbary_dist dV:", d_bary_distorted_d_v0v1v2, sep="\n")
    print()

    d_bary_corrected = perspective_correct_barycentrics(bary_distorted, face_vertex0, face_vertex1, face_vertex2)
    print("Barycentrics perspective-corrected:", d_bary_corrected, sep="\n")
    print()

    d_bary_corrected_d_bary_distorted_and_z = \
        compute_jacobian_perspective_correction(bary_distorted, face_vertex0, face_vertex1, face_vertex2)
    print("Barycentrics jacobian wr.t. distorted barycentrics & depths:", d_bary_corrected_d_bary_distorted_and_z, sep="\n")
    print()

    d_bary_corrected_wrt_vertices = \
        jacobian_barycentrics_corrected_wrt_vertices(d_bary_distorted_d_v0v1v2, d_bary_corrected_d_bary_distorted_and_z)
    print("Jacobian dbary_corrected dV:", d_bary_corrected_wrt_vertices, sep="\n")
    print()



    d_residual_d_rendered_normal = rendered_point_position - reference_point_position  # 3x1
    d_residual_d_rendered_vector = rendered_point_normal  # 3x1

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
