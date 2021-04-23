#!/usr/bin/python3

# ==================================================================================================
# A toy code example that tests forward-warping voxel centers using dual quaternion blending & an existing
# sparse motion graph
#
# To make "output" folder path coincide with repository root, please run script from repository root, i.e.:
# python3 ./pipeline/extract_voxel_centers_test.py
#
# Copyright 2021 Gregory Kramida
# ==================================================================================================

import sys
import numpy as np
from dq3d import quat, dualquat
from dq3d import op

PROGRAM_EXIT_SUCCESS = 0


def main():
    with open("output/voxel_centers_000200_red_shorts.np", 'rb') as file:
        voxel_centers = np.load(file)

    # Load graph transformation
    with open("output/red_shorts_shorts_000200_000400_rotations.np", 'rb') as file:
        rotations = np.load(file)

    with open("output/red_shorts_shorts_000200_000400_translations.np", 'rb') as file:
        translations = np.load(file)

    node_transformations_dual_quaternions = np.array([dualquat(quat(rotation), translation) for rotation, translation in
                                                      zip(rotations, translations)])


    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
