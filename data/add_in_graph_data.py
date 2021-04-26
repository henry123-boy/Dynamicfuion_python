#!/usr/bin/python3

# ================================================================================================================
# A script to add in graph data from deep_deform_graph into the main deep_deform dataset.
# ================================================================================================================

import sys
import shutil
import os
import click
import argparse

PROGRAM_EXIT_SUCCESS = 0


def main():
    parser = argparse.ArgumentParser("Add in graph data from deep_deform_graph into the main deep_deform dataset.")
    parser.add_argument("-gp", "--graph_data_path", type=str, help="path to the (extracted) deep_deform_graph data root",
                        default="/mnt/Data/Reconstruction/real_data/deepdeform/deepdeform_graph_v1")
    parser.add_argument("-ddp", "--deep_deform_data_path", type=str, help="path to the (extracted) deep_deform data root",
                        default="/mnt/Data/Reconstruction/real_data/deepdeform")
    args = parser.parse_args()
    graph_data_path = args.graph_data_path
    deep_deform_data_path = args.deep_deform_data_path
    splits = ["test", "train", "val"]
    # copy the root-level .json files
    for split in splits:
        shutil.copy(os.path.join(graph_data_path, f"{split:s}_graphs.json"), os.path.join(deep_deform_data_path, "{split:s}_graphs.json"))
    # copy each sequence
    for split in splits:
        deep_deform_split_path = os.path.join(deep_deform_data_path, split)
        graph_split_path = os.path.join(graph_data_path, split)
        deep_deform_split_sequence_folders = os.listdir(deep_deform_split_path)
        deep_deform_split_sequence_folders.sort()
        graph_split_sequence_folders = os.listdir(graph_split_path)
        graph_split_sequence_folders.sort()
        assert (deep_deform_split_sequence_folders == graph_split_sequence_folders)
        for sequence_folder in graph_split_sequence_folders:
            deep_deform_sequence_path = os.path.join(deep_deform_split_path, sequence_folder)
            graph_sequence_path = os.path.join(graph_split_path, sequence_folder)
            for subfolder in os.listdir(graph_sequence_path):
                source_path = os.path.join(graph_sequence_path, subfolder)
                target_path = os.path.join(deep_deform_sequence_path, subfolder)
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                    print("Overwriting", target_path, "from", source_path)
                    shutil.copytree(source_path, target_path)
                else:
                    print("Copying", source_path, "to", target_path)
                    shutil.copytree(source_path, target_path)

    return PROGRAM_EXIT_SUCCESS


if __name__ == '__main__':
    sys.exit(main())
