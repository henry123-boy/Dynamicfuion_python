#!/usr/bin/python3
from pathlib import Path

import sys
import os
import re

from typing import List

from apps.visualizer.parameters import VisualizerParameters
from apps.visualizer.app import VisualizerApp
from ext_argparse import process_arguments

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAILURE = -1


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
        folders = [item for item in os.listdir(base_output_folder) if os.path.isdir(os.path.join(base_output_folder, item))]
        if folders is None:
            raise ValueError(f"No experiment-run output folders detected in the base output directory {base_output_folder}")
        sortable_timestamp_named_folders = []
        for folder in folders:
            match_result = sortable_timestamp_pattern.match(folder)
            if match_result is not None:
                sortable_timestamp_named_folders.append(match_result[0])
        if len(sortable_timestamp_named_folders) > 0:
            sortable_timestamp_named_folders.sort(reverse=True)
            experiment_folder = sortable_timestamp_named_folders[0]
        else:
            base_output_directory = Path(base_output_folder)
            paths: List[Path] = \
                [path for path in sorted(base_output_directory.iterdir(), key=os.path.getmtime, reverse=True) if path.is_dir()]
            experiment_folder = paths[0].name

    print(type(experiment_folder), experiment_folder)
    output_folder = os.path.join(base_output_folder, experiment_folder, "frame_output")
    print("Reading data from ", output_folder)
    visualizer = VisualizerApp(output_folder, VisualizerParameters.start_frame.value)
    visualizer.launch()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
