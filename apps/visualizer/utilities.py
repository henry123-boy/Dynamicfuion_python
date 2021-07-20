import os
import re


def get_start_and_end_frame(output_path):
    filenames = os.listdir(output_path)
    filenames.sort()
    start_frame_ix = int(filenames[0][:6])
    end_frame_ix = int(filenames[-1][:6])
    return start_frame_ix, end_frame_ix


def get_output_frame_count(output_path):
    contents = os.listdir(output_path)
    count = 0
    canonical_mesh_filename_pattern = re.compile(r'\d{6}_canonical_mesh\.ply')
    for item in contents:
        if canonical_mesh_filename_pattern.match(item) is not None:
            count += 1
    return count
