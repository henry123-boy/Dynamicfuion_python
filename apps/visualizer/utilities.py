import os
import re


def get_frame_output_path(output_path, i_frame):
    return os.path.join(output_path, "Frame_{:03}".format(i_frame))


def get_output_frame_count(output_path):
    contents = os.listdir(output_path)
    count = 0
    canonical_mesh_filename_pattern = re.compile(r'\d{6}_canonical_mesh\.ply')
    for item in contents:
        if canonical_mesh_filename_pattern.match(item) is not None:
            count += 1
    return count
