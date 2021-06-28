#!/usr/bin/python3
import sys
import os
import argparse

import options
from apps.visualizer.visualizerapp import VisualizerApp

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAILURE = -1


def main():
    parser = argparse.ArgumentParser("App for visualizing block allocation and generated mesh alignment.")
    sequence_id = "21-06-23-19-03-59_shorts_activate_no_rot"
    parser.add_argument("--output", "-o", type=str, help="Path to output folder",
                        default=os.path.join(options.output_directory, sequence_id, "frame_output"))
    parser.add_argument("--initial_frame", "-i", type=int, help="Index of the first frame to process",
                        default=1)
    args = parser.parse_args()
    print("Reading data from ", args.output)
    visualizer = VisualizerApp(args.output, args.initial_frame)
    visualizer.launch()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
