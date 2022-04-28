#!/usr/bin/python3
import os
import sys

import argparse
import subprocess

# TODO read dataset directory from main settings file -- requires upgrading ext_argparse to ensure support of
#  nested parsers/options
# from settings import Parameters, process_arguments

PROGRAM_EXIT_SUCCESS = 0


def main():
    possible_splits = ["train", "test", "val"]
    parser = argparse.ArgumentParser("Run salient object detection to generate greyscale masks for an RGB image "
                                     "sequence.")
    parser.add_argument("-d", "--dataset", type=str, default="/mnt/Data/Reconstruction/real_data/deepdeform",
                        help=f"The root directory of your DeepDeform dataset (or another dataset following the same "
                             f"directory structure as DeepDeform).")
    parser.add_argument("-sp", "--split", type=str, default="train",
                        help=f"Data split, should be one of {str(possible_splits)})")
    parser.add_argument("-si", "--sequence_index", type=int, default=70, help="The sequence index.")
    parser.add_argument("-i", "--input_folder", type=str, default=None,
                        help="Custom sequence input folder (overrides the --split and --sequence arguments).")
    parser.add_argument("-o", "--output_folder", type=str, default="sod",
                        help="output folder. If an absolute path is given, uses that directly, otherwise assumed to be "
                             "relative to the input sequence. For custom --input_folder, assumed to be one level above "
                             "the frame input.")
    args = parser.parse_args()
    if args.input_folder is not None:
        absolute_input_folder = args.input_folder
        if os.path.isabs(args.output_foler):
            absolute_output_folder = args.output_folder
        else:
            absolute_output_folder = os.path.join(os.path.dirname(absolute_input_folder), args.output_folder)
    else:
        if args.split not in possible_splits:
            raise ValueError(f"--split should be one of one of {str(possible_splits)}, got {args.split}")
        sequence_folder = os.path.join(args.dataset, args.split, f"seq{args.sequence_index:03d}")
        absolute_input_folder = os.path.join(sequence_folder, "color")
        absolute_output_folder = os.path.join(sequence_folder, args.output_folder)

    os.makedirs(absolute_output_folder)

    u2net_script_path = os.path.join(os.path.dirname(os.path.relpath(__file__)), "3rd-party/U-2-Net/u2net_test.py")

    command_line = [sys.executable, u2net_script_path, "-i", absolute_input_folder, "-o", absolute_output_folder]

    subprocess.run(command_line)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
