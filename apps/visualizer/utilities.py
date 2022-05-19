import os
import re
from pathlib import Path


def get_start_and_end_frame(output_path: Path):
    filenames = list(output_path.iterdir())
    filenames.sort()
    start_frame_ix = int(filenames[0].stem[:6])
    end_frame_ix = int(filenames[-1].stem[:6])
    return start_frame_ix, end_frame_ix


def get_output_frame_count(output_path):
    contents = os.listdir(output_path)
    count = 0
    canonical_mesh_filename_pattern = re.compile(r'\d{6}_canonical_mesh\.ply')
    for item in contents:
        if canonical_mesh_filename_pattern.match(item) is not None:
            count += 1
    return count


def get_gn_iteration_count(start_frame_ix, output_path):
    # point cloud setup
    start_frame_ix_string = f"{start_frame_ix:06d}"
    first_frame_point_file_pattern = re.compile(start_frame_ix_string + r"_deformed_points_iter_(\d{3})[.]npy")
    first_frame_point_files_in_output_path = [file for file in os.listdir(output_path) if
                                              first_frame_point_file_pattern.match(file) is not None]
    return len(first_frame_point_files_in_output_path)


def source_and_target_point_clouds_are_present(start_frame_ix, output_path):
    start_frame_source_pc_filename = f"{start_frame_ix:06d}_source_rgbxyz.npy"
    start_frame_target_pc_filename = f"{start_frame_ix:06d}_target_rgbxyz.npy"
    all_filenames = os.listdir(output_path)
    return start_frame_source_pc_filename in all_filenames and start_frame_target_pc_filename in all_filenames


def correspondence_info_is_present(start_frame_ix, output_path):
    start_frame_vcm_filename = f"{start_frame_ix:06d}_valid_correspondence_mask.npy"
    start_frame_pm_filename = f"{start_frame_ix:06d}_prediction_mask.npy"
    start_frame_tm_filename = f"{start_frame_ix:06d}_target_matches.npy"
    all_filenames = os.listdir(output_path)
    return start_frame_vcm_filename in all_filenames \
           and start_frame_pm_filename in all_filenames \
           and start_frame_tm_filename in all_filenames


def graph_info_is_present(start_frame_ix: int, output_path: Path) -> bool:
    start_frame_nodes_filename = f"{start_frame_ix:06d}_nodes.npy"
    start_frame_edges_filename = f"{start_frame_ix:06d}_edges.npy"
    start_frame_rotations_filename = f"{start_frame_ix:06d}_rotations.npy"
    start_frame_translations_filename = f"{start_frame_ix:06d}_translations.npy"
    all_filenames = os.listdir(output_path)
    return start_frame_nodes_filename in all_filenames \
        and start_frame_edges_filename in all_filenames \
        and start_frame_rotations_filename in all_filenames \
        and start_frame_translations_filename in all_filenames
