import os
import gzip
import numpy as np


def get_start_and_end_frame(output_path):
    filenames = os.listdir(output_path)
    filenames.sort()
    start_frame_ix = int(filenames[0][:6])
    end_frame_ix = int(filenames[-1][:6])
    return start_frame_ix, end_frame_ix


def load_inverse_matrices(output_folder, input_folder):
    path = os.path.join(output_folder, "camera_matrices.dat")
    if os.path.isfile(path):
        file = gzip.open(path, "rb")
        inverse_camera_matrices = []
        while file.readable():
            buffer = file.read(size=64)
            if not buffer:
                break
            inverse_camera_matrix = np.linalg.inv(np.resize(np.frombuffer(buffer, dtype=np.float32), (4, 4)).T)
            inverse_camera_matrices.append(inverse_camera_matrix)
        print("Read inverse camera matrices for", len(inverse_camera_matrices), "frames.")
        return inverse_camera_matrices
    else:
        color_frames_path = os.path.join(input_folder, "color")
        start_frame_ix, end_frame_ix = get_start_and_end_frame(color_frames_path)
        frame_count = end_frame_ix - start_frame_ix
        return [np.identity(4)] * frame_count


def load_matrices(output_folder):
    path = os.path.join(output_folder, "camera_matrices.dat")
    file = gzip.open(path, "rb")
    camera_matrices = []
    while file.readable():
        buffer = file.read(size=64)
        if not buffer:
            break
        camera_matrix = np.resize(np.frombuffer(buffer, dtype=np.float32), (4, 4)).T
        camera_matrices.append(camera_matrix)
    print("Read camera matrices for", len(camera_matrices), "frames.")
    return camera_matrices
