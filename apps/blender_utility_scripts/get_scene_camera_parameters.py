#!/usr/bin/python
import bpy
import numpy as np
import sys
import argparse

PROGRAM_EXIT_SUCCESS = 0


def get_camera_internal_matrix_from_current_scene(mode='simple'):
    scene = bpy.context.scene
    bpy.utils

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    camdata = scene.camera.data

    if mode == 'simple':
        aspect_ratio = width / height
        internal_matrix = np.zeros((3, 3), dtype=np.float32)
        internal_matrix[0][0] = width / 2 / np.tan(camdata.angle / 2)
        internal_matrix[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        internal_matrix[0][2] = width / 2.
        internal_matrix[1][2] = height / 2.
        internal_matrix[2][2] = 1.
        internal_matrix.transpose()
    elif mode == 'complete':

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        print(sensor_height, sensor_width)

        if camdata.sensor_fit == 'VERTICAL':
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width * pixel_aspect_ratio / sensor_width
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels

        internal_matrix = np.array([
            [alpha_u, skew, u_0],
            [0, alpha_v, v_0],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        raise ValueError(f"The parameter `mode` is set to {mode}. Expected one of: 'simple', 'complete'")

    return internal_matrix


def main():
    parser = argparse.ArgumentParser(
        "Program for extracting camera parameters from a blender scene file. Assumes there is a camera in the scene.")
    parser.add_argument('scene_file', metavar='S', type=str,
                        help='The .bpy blender scene file.')
    parser.add_argument('--mode', '-m', default='complete',
                        help="Acquisition mode (may be one of {'simple', 'complete'}).")
    args = parser.parse_args()

    bpy.ops.wm.open_mainfile(filepath=args.scene_file)

    print(get_camera_internal_matrix_from_current_scene(args.mode))

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
