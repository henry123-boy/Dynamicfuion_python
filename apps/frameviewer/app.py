#!/usr/bin/python3

# standard library
import os
import sys
from enum import Enum
from typing import Union
from multiprocessing import Queue

# third-party
import cv2
import vtk
import numpy as np

# local (frameviewer app, apps shared)
from apps.frameviewer import image_conversion, frameloading
from apps.shared import trajectory_loading
from apps.shared.app import App
from apps.frameviewer.pixel_highlighter import PixelHighlighter
from apps.shared.screen_management import set_up_render_window_bounds


class ViewingMode(Enum):
    DEPTH = 0
    COLOR = 1


class CameraProjection:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self._fx_inv = 1.0 / fx
        self._fy_inv = 1.0 / fy

    def project_to_camera_space(self, u, v, depth):
        x = depth * (u - self.cx) * self._fx_inv
        y = depth * (v - self.cy) * self._fy_inv
        return x, y, depth


class FrameViewerApp(App):
    def __init__(self, input_folder, output_folder, frame_index_to_start_with, initial_mask_threshold,
                 voxel_size: float = 0.004, voxel_block_resolution: int = 8,
                 save_sequence_parameter_state=True,
                 outgoing_queue: Union[None, Queue] = None):
        intrinsic_matrix = np.loadtxt(os.path.join(input_folder, "intrinsics.txt"))

        self.voxel_size = voxel_size
        self.voxel_block_resolution = voxel_block_resolution
        self.voxel_block_size = voxel_size * voxel_block_resolution

        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        self.camera_projection = CameraProjection(fx, fy, cx, cy)

        self.start_frame_index = frame_index_to_start_with
        self.initial_mask_threshold = initial_mask_threshold
        self.input_folder = input_folder
        self.output_folder = output_folder

        state_path = os.path.join(self.output_folder, "frameviewer_state.txt")
        state = (20.0, 20.0, 2.0, frame_index_to_start_with, initial_mask_threshold)
        if os.path.isfile(state_path):
            loaded_state = np.loadtxt(state_path)
            if len(loaded_state) < len(state):
                os.unlink(state_path)
            else:
                state = loaded_state
        self.save_sequence_parameter_state = save_sequence_parameter_state
        self.outgoing_queue = outgoing_queue
        self.inverse_camera_matrices = trajectory_loading.load_inverse_matrices(output_folder, input_folder)

        self.current_camera_matrix = None

        self.image_masks_enabled = True
        self.image_mask_threshold = int(state[4])

        self.color_numpy_image = None
        self.depth_numpy_image = None
        self.mask_image = None
        self.depth_numpy_image_uint8 = None

        self.scaled_color = None
        self.scaled_depth = None
        self.color_vtk_image = None
        self.depth_vtk_image = None
        self.image_size = None

        self.image_mapper = vtk.vtkImageMapper()
        self.image_mapper.SetColorWindow(255)
        self.image_mapper.SetColorLevel(127.5)

        self.image_actor = vtk.vtkActor2D()
        self.image_actor.SetMapper(self.image_mapper)
        self.image_actor.SetPosition(int(state[0]), int(state[1]))

        colors = vtk.vtkNamedColors()

        self.renderer_image = vtk.vtkRenderer()
        self.renderer_image.SetBackground(0.1, 0.1, 0.1)
        self.renderer_image.SetLayer(0)
        self.renderer_highlights = vtk.vtkRenderer()
        self.renderer_highlights.SetLayer(1)
        self.render_window = vtk.vtkRenderWindow()
        set_up_render_window_bounds(self.render_window, None, 2)
        self.render_window.SetNumberOfLayers(2)
        self.render_window.AddRenderer(self.renderer_image)
        self.render_window.AddRenderer(self.renderer_highlights)
        self.render_window.SetWindowName("Frameviewer: " + self.input_folder)

        self.renderer_image.AddActor2D(self.image_actor)

        self.frame_index = -1
        self.viewing_mode = ViewingMode.COLOR
        self.scale = state[2]
        self._scaled_resolution = None
        self.panning = False
        self.zooming = False

        self.text_mapper = vtk.vtkTextMapper()
        self.text_mapper.SetInput("Frame: {:d} | Scale: {:f}\nPixel: 0, 0\nDepth: 0 m\nColor: 0 0 0\n"
                                  "Camera-space: 0 0 0\nWorld-space: 0 0 0\nBlock-space: 0 0 0"
                                  .format(frame_index_to_start_with, self.scale))
        self.number_of_lines = len(self.text_mapper.GetInput().splitlines())
        text_property = self.text_mapper.GetTextProperty()
        self.font_size = 20
        text_property.SetFontSize(self.font_size)
        text_property.SetColor(colors.GetColor3d('Mint'))

        self.text_actor = vtk.vtkActor2D()
        self.text_actor.SetMapper(self.text_mapper)
        self.last_window_width, self.last_window_height = 0, 0
        self.update_window(None, None)
        self.renderer_image.AddActor(self.text_actor)

        self.pixel_highlighter = PixelHighlighter(self.renderer_highlights)
        self.pixel_highlighter.set_scale(self.scale)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(None)
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.AddObserver("ModifiedEvent", self.update_window)
        self.interactor.AddObserver("KeyPressEvent", self.keypress)
        self.interactor.AddObserver("LeftButtonPressEvent", self.button_event)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self.button_event)
        self.interactor.AddObserver("MouseWheelForwardEvent", self.button_event)
        self.interactor.AddObserver("MouseWheelBackwardEvent", self.button_event)
        self.interactor.AddObserver("MouseMoveEvent", self.mouse_move)

        self.frame_index = None
        self.set_frame(int(state[3]))

    def launch(self):
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()

    def update_window(self, obj, event):
        (window_width, window_height) = self.render_window.GetSize()
        if window_width != self.last_window_width or window_height != self.last_window_height:
            self.text_actor.SetDisplayPosition(window_width - 400,
                                               window_height - (self.number_of_lines + 2) * self.font_size)
            self.last_window_width = window_width
            self.last_window_height = window_height

    def update_active_vtk_image(self, force_reset=False):
        data_by_mode = {
            ViewingMode.COLOR: (self.scaled_color, self.color_vtk_image),
            ViewingMode.DEPTH: (self.scaled_depth, self.depth_vtk_image)
        }
        numpy_image_source, vtk_image_target = data_by_mode[self.viewing_mode]

        if vtk_image_target is None or force_reset is True:
            vtk_image_target = image_conversion.numpy_image_as_vtk_image_data(numpy_image_source)
        else:
            image_conversion.update_vtk_image(vtk_image_target, numpy_image_source)

        self.image_mapper.SetInputData(vtk_image_target)
        if self.viewing_mode == ViewingMode.DEPTH:
            self.depth_vtk_image = vtk_image_target
        elif self.viewing_mode == ViewingMode.COLOR:
            self.color_vtk_image = vtk_image_target

        self.render_window.Render()

    def update_scaled_images(self):
        self._scaled_resolution = (self.image_size * self.scale).astype(np.int32)
        if self.scale % 1.0 == 0:
            interpolation_mode = cv2.INTER_NEAREST
        else:
            interpolation_mode = cv2.INTER_LINEAR

        old_scaled_depth_shape = None if self.scaled_depth is None else self.scaled_depth.shape
        self.scaled_depth = cv2.resize(self.depth_numpy_image_uint8, tuple(self._scaled_resolution),
                                       interpolation=interpolation_mode)
        old_scaled_color_shape = None if self.scaled_color is None else self.scaled_color.shape
        self.scaled_color = cv2.resize(self.color_numpy_image, tuple(self._scaled_resolution),
                                       interpolation=interpolation_mode)
        if old_scaled_depth_shape is not None:
            old_image_shape = old_scaled_color_shape if self.viewing_mode == ViewingMode.COLOR else old_scaled_depth_shape
            new_image_shape = self.scaled_color.shape if self.viewing_mode == ViewingMode.COLOR else self.scaled_depth.shape
            image_x, image_y = self.image_actor.GetPosition()
            x, y = self.interactor.GetEventPosition()
            new_image_x = x - ((x - image_x) / old_image_shape[1]) * new_image_shape[1]
            new_image_y = y - ((y - image_y) / old_image_shape[0]) * new_image_shape[0]
            self.image_actor.SetPosition(new_image_x, new_image_y)
        self.update_active_vtk_image(force_reset=True)

    def update_viewing_mode(self, viewing_mode):
        """
        :type viewing_mode ViewingMode
        """
        if self.viewing_mode != viewing_mode:
            print("Viewing mode:", viewing_mode.name)
            self.viewing_mode = viewing_mode
            self.update_scaled_images()

    def update_masking(self):
        if self.image_masks_enabled:
            self.color_numpy_image = frameloading.load_color_numpy_image(self.frame_index, self.input_folder)
            self.depth_numpy_image = frameloading.load_depth_numpy_image(self.frame_index, self.input_folder)
            self.color_numpy_image[self.mask_image < self.image_mask_threshold] = 0
            self.depth_numpy_image[self.mask_image < self.image_mask_threshold] = 0
            self.update_scaled_images()

    def set_frame(self, frame_index):
        print("Frame:", frame_index)
        self.current_camera_matrix = None if len(self.inverse_camera_matrices) <= frame_index \
            else self.inverse_camera_matrices[frame_index - self.start_frame_index]

        print_inverted_camera_matrix = False

        if print_inverted_camera_matrix and self.current_camera_matrix is not None:
            print("Inverted camera pose:")
            print(self.current_camera_matrix)

        self.frame_index = frame_index
        self.color_numpy_image = frameloading.load_color_numpy_image(frame_index, self.input_folder)
        self.depth_numpy_image = frameloading.load_depth_numpy_image(frame_index, self.input_folder)
        self.image_size = np.array((self.color_numpy_image.shape[1], self.color_numpy_image.shape[0]))

        if self.image_masks_enabled:
            self.mask_image = frameloading.load_mask_numpy_image(self.frame_index, self.input_folder)
            self.color_numpy_image[self.mask_image < self.image_mask_threshold] = 0
            self.depth_numpy_image[self.mask_image < self.image_mask_threshold] = 0

        self.depth_numpy_image_uint8 = image_conversion.convert_to_viewable_depth(self.depth_numpy_image)
        self.update_scaled_images()
        x, y = self.interactor.GetEventPosition()
        self.report_on_mouse_location(x, y)

    def zoom_scale(self, step, increment_step=False, increment=0.5):
        x, y = self.interactor.GetEventPosition()
        if increment_step:
            self.scale -= (self.scale % increment)
            self.scale += (step * increment)
        else:
            self.scale *= pow(1.02, (1.0 * step))
        self.pixel_highlighter.set_scale(self.scale)
        self.update_scaled_images()
        self.report_on_mouse_location(x, y)

    # Handle the mouse button events.
    def button_event(self, obj, event):
        if event == "LeftButtonPressEvent":
            self.panning = True
        elif event == "LeftButtonReleaseEvent":
            self.panning = False
        elif event == "RightButtonPressEvent":
            self.zooming = True
        elif event == "RightButtonReleaseEvent":
            self.zooming = False
        elif event == "MouseWheelForwardEvent":
            self.zoom_scale(1, self.interactor.GetControlKey())
        elif event == "MouseWheelBackwardEvent":
            self.zoom_scale(-1, self.interactor.GetControlKey())

    KEYS_TO_SEND = {"q", "Escape", "bracketright", "bracketleft"}

    def keypress(self, obj, event):
        key = obj.GetKeySym()
        if key in FrameViewerApp.KEYS_TO_SEND and self.outgoing_queue is not None:
            self.outgoing_queue.put(key)
        self.handle_key(key)

    def handle_key(self, key):
        print("Key:", key)
        if key == "q" or key == "Escape":
            image_x, image_y = self.image_actor.GetPosition()
            path = os.path.join(self.output_folder, "frameviewer_state.txt")
            frame_index_to_save = self.start_frame_index
            masking_threshold_to_save = self.initial_mask_threshold
            if self.save_sequence_parameter_state:
                frame_index_to_save = self.frame_index
                masking_threshold_to_save = self.image_mask_threshold
            np.savetxt(path, (image_x, image_y, self.scale, frame_index_to_save, masking_threshold_to_save))
            self.interactor.InvokeEvent("DeleteAllObjects")
            sys.exit()
        elif key == "bracketright":
            self.set_frame(self.frame_index + 1)
        elif key == "bracketleft":
            self.set_frame(self.frame_index - 1)
        elif key == "c":
            self.update_viewing_mode(ViewingMode.COLOR)
        elif key == "d":
            self.update_viewing_mode(ViewingMode.DEPTH)
        elif key == "Right":
            pass
        elif key == "Left":
            pass
        elif key == "Up":
            if self.image_mask_threshold > 0:
                self.image_mask_threshold -= 1
                print("Image mask threshold:", self.image_mask_threshold)
                self.update_masking()
        elif key == "Down":
            if self.image_mask_threshold < 255:
                self.image_mask_threshold += 1
                print("Image mask threshold:", self.image_mask_threshold)
                self.update_masking()

    def mouse_move(self, obj, event):
        last_x_y_pos = self.interactor.GetLastEventPosition()
        last_x = last_x_y_pos[0]
        last_y = last_x_y_pos[1]

        x_y_pos = self.interactor.GetEventPosition()
        x = x_y_pos[0]
        y = x_y_pos[1]

        if self.panning:
            self.pan(x, y, last_x, last_y)
        elif self.zooming:
            self.zoom(x, y, last_x, last_y)
        else:
            self.report_on_mouse_location(x, y)

    def zoom(self, x, y, last_x, last_y):
        self.scale *= pow(1.02, (0.5 * (y - last_y)))
        self.update_scaled_images()

    def pan(self, x, y, last_x, last_y):
        point_x = x - last_x
        point_y = y - last_y
        image_position = self.image_actor.GetPosition()
        self.image_actor.SetPosition(image_position[0] + point_x, image_position[1] + point_y)
        self.render_window.Render()

    def is_pixel_within_image(self, x, y):
        image_start_x, image_start_y = self.image_actor.GetPosition()
        image_end_x = image_start_x + self._scaled_resolution[0] - 1
        image_end_y = image_start_y + self._scaled_resolution[1] - 1
        return image_start_x < x < image_end_x and image_start_y < y < image_end_y

    def get_frame_pixel(self, x, y):
        image_start_x, image_start_y = self.image_actor.GetPosition()
        frame_x = ((x - image_start_x) / self.scale)
        frame_y = self.color_numpy_image.shape[0] - ((y - image_start_y + 1) / self.scale)
        return frame_x, frame_y

    def get_block_coordinate(self, point_world):
        return (point_world / self.voxel_block_size) + 1 / (
                2 * self.voxel_block_resolution)

    def update_location_text(self, u, v, depth, color):
        camera_coords = self.camera_projection.project_to_camera_space(u, v, depth)
        camera_coords_homogenized = np.array((camera_coords[0], camera_coords[1], camera_coords[2], 1.0)).T
        world_coords = self.current_camera_matrix.dot(camera_coords_homogenized)
        block_coords = self.get_block_coordinate(world_coords)
        self.text_mapper.SetInput(
            "Frame: {:d} | Scale: {:f}\nPixel: {:d}, {:d}\nDepth: {:f} m\nColor: {:d}, {:d}, {:d}\n"
            "Camera-space: {:02.4f}, {:02.4f}, {:02.4f}\nWorld-space: {:02.4f}, {:02.4f}, {:02.4f}\n"
            "Block-space: {:02.4f}, {:02.4f}, {:02.4f}"
                .format(self.frame_index, self.scale, u, v, depth, color[0], color[1], color[2],
                        camera_coords[0], camera_coords[1], camera_coords[2],
                        world_coords[0], world_coords[1], world_coords[2],
                        block_coords[0], block_coords[1], block_coords[2]))
        self.text_mapper.Modified()
        self.render_window.Render()

    def report_on_mouse_location(self, x, y):
        if self.is_pixel_within_image(x, y):
            frame_x, frame_y = self.get_frame_pixel(x, y)
            frame_x_int = int(frame_x)
            frame_y_int = int(frame_y)
            depth = self.depth_numpy_image[frame_y_int, frame_x_int]
            color = self.color_numpy_image[frame_y_int, frame_x_int]
            image_x, image_y = self.image_actor.GetPosition()
            self.pixel_highlighter.set_position(image_x + frame_x_int * self.scale, image_y + (
                    self.color_numpy_image.shape[0] - frame_y_int - 1) * self.scale)
            self.pixel_highlighter.show()
            self.update_location_text(frame_x_int, frame_y_int, depth, color)
        else:
            self.pixel_highlighter.hide()
