#!/usr/bin/python3
from pathlib import Path
import sys
import os
import re
from multiprocessing import Process, Queue
import ctypes

from typing import List

import vtk

from apps.visualizer.parameters import VisualizerParameters
from apps.visualizer.app import VisualizerApp
from ext_argparse import process_arguments

from apps.frameviewer.app import FrameViewerApp, CameraProjection
from data.camera import load_intrinsic_matrix_entries_from_text_4x4_matrix
from apps.frameviewer.parameters import FrameviewerParameters

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAILURE = -1


def start_synchronized_frameviewer(frameviewer_settings_path: Path, queue: Queue):
    process_arguments(FrameviewerParameters, "An app to view a masked RGB-D sequence frame-by-frame and analyze target"
                                             "hash blocks for the surface (in a spatially-hashed voxel volume)."
                                             "Also allows to determine the optimal threshold for masking. ",
                      default_settings_file=frameviewer_settings_path,
                      generate_default_settings_if_missing=True)
    print("Reading data from ", FrameviewerParameters.input.value)

    FrameViewerApp.VOXEL_BLOCK_SIZE_METERS = \
        FrameviewerParameters.tsdf.voxel_size.value * FrameviewerParameters.tsdf.block_resolution.value
    FrameViewerApp.VOXEL_BLOCK_SIZE_VOXELS = FrameviewerParameters.tsdf.block_resolution.value
    FrameViewerApp.VOXEL_SIZE = FrameviewerParameters.tsdf.voxel_size.value

    fx, fy, cx, cy = load_intrinsic_matrix_entries_from_text_4x4_matrix(
        os.path.join(FrameviewerParameters.input.value, "intrinsics.txt"))
    FrameViewerApp.PROJECTION = CameraProjection(fx, fy, cx, cy)

    frameviewer = FrameViewerApp(FrameviewerParameters.input.value, FrameviewerParameters.output.value,
                                 FrameviewerParameters.start_frame_index.value, queue)
    frameviewer.launch()


class TimerCallback:
    def __init__(self, queue: Queue):
        self.queue = queue

    def execute(self, object, event):
        if not self.queue.empty():
            print("External process key:", self.queue.get())


def main():
    settings_path = os.path.join(Path(__file__).parent.resolve(), "configuration_files/visualizer_parameters.yaml")
    process_arguments(VisualizerParameters,
                      "App for visualizing block allocation and generated mesh alignment.",
                      default_settings_file=settings_path,
                      generate_default_settings_if_missing=True)
    experiment_folder = VisualizerParameters.experiment_folder.value
    base_output_folder = VisualizerParameters.base_output_path.value
    sortable_timestamp_pattern = re.compile(r"^\d\d-\d\d-\d\d-\d\d-\d\d-\d\d")

    if experiment_folder == "!LATEST!":
        folders = [item for item in os.listdir(base_output_folder) if
                   os.path.isdir(os.path.join(base_output_folder, item))]
        if folders is None:
            raise ValueError(
                f"No experiment-run output folders detected in the base output directory {base_output_folder}")
        sortable_timestamp_named_folders = []
        for folder in folders:
            match_result = sortable_timestamp_pattern.match(folder)
            if match_result is not None:
                sortable_timestamp_named_folders.append(folder)
        if len(sortable_timestamp_named_folders) > 0:
            sortable_timestamp_named_folders.sort(reverse=True)
            experiment_folder = sortable_timestamp_named_folders[0]
        else:
            base_output_directory = Path(base_output_folder)
            paths: List[Path] = \
                [path for path in sorted(base_output_directory.iterdir(), key=os.path.getmtime, reverse=True) if
                 path.is_dir()]
            experiment_folder = paths[0].name

    print(type(experiment_folder), experiment_folder)

    frame_output_folder = Path(base_output_folder, experiment_folder, "frame_output")
    print("Reading data from ", frame_output_folder)
    visualizer = VisualizerApp(frame_output_folder, VisualizerParameters.start_frame.value)

    # launch frameviewer if we have info for it -- TODO also make a setting/parameter for this.
    #  TODO: launching multiple VTK windows via multiprocessing fails on VTK 9.1 (at least w/ Xorg server).
    #   Use VTK 9.0.1 for now if you're facing this issue...
    frameviewer_info_path = Path(base_output_folder, experiment_folder, "frameviewer_info.yaml")
    frameviewer_running = False
    frameviewer_process = None
    if frameviewer_info_path.exists():
        queue = Queue()
        frameviewer_process = Process(target=start_synchronized_frameviewer, args=(frameviewer_info_path, queue))
        frameviewer_process.start()
        frameviewer_running = True
        timer_callback = TimerCallback(queue)
        visualizer.interactor.AddObserver('TimerEvent', timer_callback.execute)
        timer_id = visualizer.interactor.CreateRepeatingTimer(1000)

    visualizer.launch()

    if frameviewer_running:
        frameviewer_process.join()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
