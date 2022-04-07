#!/usr/bin/python3
from pathlib import Path

import sys
from apps.frameviewer.app import FrameViewerApp, CameraProjection
from data.camera import load_intrinsic_matrix_entries_from_text_4x4_matrix
from apps.frameviewer.parameters import FrameviewerParameters
from ext_argparse import process_arguments
import os.path

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAILURE = -1


def main():
    default_configuration_path = os.path.join(Path(__file__).parent.resolve(), "configuration_files/frameviewer_parameters.yaml")
    process_arguments(FrameviewerParameters, "An app to view a masked RGB-D sequence frame-by-frame and analyze target"
                                             "hash blocks for the surface (in a spatially-hashed voxel volume)."
                                             "Also allows to determine the optimal threshold for masking. ",
                      default_settings_file=default_configuration_path,
                      generate_default_settings_if_missing=True)
    print("Reading data from ", FrameviewerParameters.input.value)

    FrameViewerApp.VOXEL_BLOCK_SIZE_METERS = \
        FrameviewerParameters.tsdf.voxel_size.value * FrameviewerParameters.tsdf.block_resolution.value
    FrameViewerApp.VOXEL_BLOCK_SIZE_VOXELS = FrameviewerParameters.tsdf.block_resolution.value
    FrameViewerApp.VOXEL_SIZE = FrameviewerParameters.tsdf.voxel_size.value

    fx, fy, cx, cy = load_intrinsic_matrix_entries_from_text_4x4_matrix(os.path.join(FrameviewerParameters.input.value, "intrinsics.txt"))
    FrameViewerApp.PROJECTION = CameraProjection(fx, fy, cx, cy)

    app = FrameViewerApp(FrameviewerParameters.input.value, FrameviewerParameters.output.value,
                         FrameviewerParameters.start_frame_index.value)
    app.launch()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
