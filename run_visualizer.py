#!/usr/bin/python3
from pathlib import Path

import sys
import os
import argparse

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
    output_folder = os.path.join(VisualizerParameters.base_output_path.value, VisualizerParameters.experiment_folder.value)
    print("Reading data from ", output_folder)
    visualizer = VisualizerApp(output_folder, VisualizerParameters.start_frame.value)
    visualizer.launch()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
