#!/usr/bin/python3
import sys
import os
import argparse

from settings import PathParameters, process_arguments
from apps.visualizer.visualizerapp import VisualizerApp

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAILURE = -1


def main():
    process_arguments()
    parser = argparse.ArgumentParser("App for visualizing block allocation and generated mesh alignment.")
    run_output_folder = "21-08-09-18-22-01_BERLIN_advanced_pc"
    parser.add_argument("--output", "-o", type=str, help="Path to output folder",
                        default=os.path.join(PathParameters.output_directory.value, run_output_folder, "frame_output"))
    parser.add_argument("--initial_frame", "-i", type=int, help="Index of the first frame to process",
                        default=-1)
    args = parser.parse_args()
    print("Reading data from ", args.output)
    visualizer = VisualizerApp(args.output, args.initial_frame)
    visualizer.launch()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
