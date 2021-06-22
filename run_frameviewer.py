#!/usr/bin/python3
import sys
import argparse
from apps.frameviewer.frameviewer import FrameViewerApp
from apps.shared import screen_management
import options
import os.path

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAILURE = -1


def main():
    parser = argparse.ArgumentParser("App for visualizing RGB-D frame data.")
    parser.add_argument("--output", "-o", type=str, help="Path to folder with frame data",
                        default=os.path.join(options.dataset_base_directory, "train/seq070"))
    args = parser.parse_args()
    print("Reading data from ", args.output)

    app = FrameViewerApp(args.output, 16)
    app.launch()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
