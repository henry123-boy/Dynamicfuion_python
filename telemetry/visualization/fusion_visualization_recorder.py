import os
from typing import List

import numpy as np
from open3d.visualization import Visualizer
import open3d as o3d
import cv2
from multiprocessing import cpu_count


class FusionVisualizationRecorder():

    def __init__(self, output_video_path,
                 front=[0, 0, -1],
                 lookat=[0, 0, 1.5],
                 up=[0, -1.0, 0],
                 zoom=0.7):
        self.visualizer = Visualizer()
        self.writer = None
        self.front = front
        self.lookat = lookat
        self.up = up
        self.zoom = zoom

        self.output_video_path = output_video_path
        # cv2.VideoWriter_fourcc('X', '2', '6', '4')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fourcc = cv2.VideoWriter_fourcc(*"x264")
        self.writer = cv2.VideoWriter(self.output_video_path,
                                      fourcc,
                                      30, (1920, 1080), True)
        self.writer.set(cv2.VIDEOWRITER_PROP_NSTRIPES, cpu_count())
        self.visualizer.create_window("Fusion output frame capture")

    def __del__(self):
        self.visualizer.destroy_window()
        if self.writer is not None:
            self.writer.release()

    def capture_frame(self, geometry: List[o3d.geometry.Geometry3D]) -> None:

        for item in geometry:
            self.visualizer.add_geometry(item)
            self.visualizer.update_geometry(item)

        view_controller: o3d.visualization.ViewControl = self.visualizer.get_view_control()
        view_controller.set_front(self.front)
        view_controller.set_lookat(self.lookat)
        view_controller.set_up(self.up)
        view_controller.set_zoom(self.zoom)

        self.visualizer.poll_events()
        self.visualizer.update_renderer()

        frame_image = (np.array(self.visualizer.capture_screen_float_buffer()) * 255).astype(np.uint8)
        self.writer.write(frame_image)
        self.visualizer.clear_geometries()
