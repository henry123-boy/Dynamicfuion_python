import os
import re
from enum import Enum

import vtk
import sys

from apps.visualizer import utilities
from apps.visualizer.mesh import Mesh
from apps.visualizer.point_cloud import PointCloud


class VisualizerApp:

    def __init__(self, output_path, start_frame_ix=-1):
        self.__alt_pressed = False

        minimum_start_frame, self.frame_index_upper_bound = utilities.get_start_and_end_frame(output_path)

        if start_frame_ix == -1:
            start_frame_ix = minimum_start_frame
        elif start_frame_ix < minimum_start_frame:
            raise ValueError(f"Smallest start frame for given sequence is {minimum_start_frame:d}, "
                             f"cannot start from frame {start_frame_ix:d}.")

        # self.inverse_camera_matrices = trajectory_loading.load_inverse_matrices(output_path)
        self.start_frame_ix = start_frame_ix
        self.output_path = output_path
        self.offset_cam = (0.2562770766576, 0.13962609403401335, -0.2113334598208764)
        colors = vtk.vtkNamedColors()
        self.current_frame = start_frame_ix

        # renderer & render window initialization
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        # mesh setup
        self.canonical_mesh = Mesh(self.renderer, self.render_window, colors.GetColor3d("Peacock"))
        self.warped_live_mesh = Mesh(self.renderer, self.render_window, colors.GetColor3d("Green"))
        self.shown_mesh_index = 0

        self.meshes = [self.canonical_mesh, self.warped_live_mesh]
        self.mesh_names = ["canonical_mesh", "warped_live_mesh"]

        def get_gn_iteration_count():
            start_frame_ix_string = f"{start_frame_ix:06d}"
            first_frame_point_file_pattern = re.compile(start_frame_ix_string + r"_deformed_points_iter_(\d{3})[.]npy")
            first_frame_point_files_in_output_path = [file for file in os.listdir(self.output_path) if
                                                      first_frame_point_file_pattern.match(file) is not None]
            return len(first_frame_point_files_in_output_path)

        self.shown_point_cloud_index = 0

        gn_iteration_count = get_gn_iteration_count()

        self.point_clouds = []
        self.point_cloud_names = []
        for i_point_cloud in range(0, gn_iteration_count):
            self.point_clouds.append(PointCloud(self.renderer, self.render_window, colors.GetColor3d("White")))
            self.point_cloud_names.append(f"gn_point_cloud_iter_{i_point_cloud:03d}")

        self.renderer.SetBackground(colors.GetColor3d("Black"))
        self.render_window.SetSize(1400, 900)
        self.render_window.SetWindowName(output_path)

        # Interactor setup
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(None)
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.Initialize()

        self.interactor.AddObserver("KeyPressEvent", self.keypress)
        self.interactor.AddObserver("KeyReleaseEvent", self.key_release)
        self.interactor.AddObserver("LeftButtonPressEvent", self.button_event)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self.button_event)
        self.interactor.AddObserver("RightButtonPressEvent", self.button_event)
        self.interactor.AddObserver("RightButtonReleaseEvent", self.button_event)
        self.interactor.AddObserver("MiddleButtonPressEvent", self.button_event)
        self.interactor.AddObserver("MiddleButtonReleaseEvent", self.button_event)
        self.interactor.AddObserver("MiddleButtonReleaseEvent", self.button_event)
        self.interactor.AddObserver("MouseWheelForwardEvent", self.button_event)
        self.interactor.AddObserver("MouseWheelBackwardEvent", self.button_event)
        self.interactor.AddObserver("MouseMoveEvent", self.mouse_move)

        self.rotating = False
        self.panning = False
        self.zooming = False

        # axes actor
        self.axes = vtk.vtkAxesActor()
        self.axes_widget = widget = vtk.vtkOrientationMarkerWidget()
        rgba = colors.GetColor4ub("Carrot")
        widget.SetOutlineColor(rgba[0], rgba[1], rgba[2])
        widget.SetOrientationMarker(self.axes)
        widget.SetInteractor(self.interactor)
        widget.SetViewport(0.0, 0.2, 0.2, 0.4)
        widget.SetEnabled(1)
        widget.InteractiveOn()

        self.camera = camera = self.renderer.GetActiveCamera()

        # front=[0, 0, -1],
        # lookat=[0, 0, 1.5],
        # up=[0, -1.0, 0],
        # zoom=0.7

        camera.SetPosition(self.offset_cam[0], self.offset_cam[1], self.offset_cam[2])
        camera.SetPosition(0, 0, -1)
        camera.SetViewUp(0, 1.0, 0)
        camera.SetFocalPoint(0, 0, 1.5)
        camera.SetClippingRange(0.01, 10.0)

        self.render_window.Render()
        # 2nd monitor from left
        self.render_window.SetPosition(1281, 187)
        # 4th monitor
        # self.render_window.SetPosition(5121, 75)
        # fullscreen

        self.set_frame(self.current_frame)
        self.show_mesh_at_index(self.shown_mesh_index)
        self.show_point_cloud_at_index(self.shown_point_cloud_index)

        # uncomment to start out with meshes invisible
        # self.meshes[self.shown_mesh_index].toggle_visibility()
        self.render_window.Render()
        self._pixel_labels_visible = False

    def load_frame_meshes(self, i_frame):
        canonical_path = os.path.join(self.output_path, f"{i_frame:06d}_canonical_mesh.ply")
        warped_live_path = os.path.join(self.output_path, f"{i_frame:06d}_warped_mesh.ply")
        self.canonical_mesh.update(canonical_path)
        self.warped_live_mesh.update(warped_live_path)

    def load_frame_point_clouds(self, i_frame):
        i_gn_iteration = 0
        for point_cloud in self.point_clouds:
            point_cloud.update(os.path.join(self.output_path, f"{i_frame:06d}_deformed_points_iter_{i_gn_iteration:03d}.npy"))
            i_gn_iteration += 1

    def launch(self):
        # Start the event loop.
        self.interactor.Start()

    def show_mesh_at_index(self, i_mesh_to_show):
        print("Mesh:", self.mesh_names[i_mesh_to_show])
        self.shown_mesh_index = i_mesh_to_show
        i_mesh = 0
        for mesh in self.meshes:
            if i_mesh_to_show == i_mesh:
                mesh.show()
            else:
                mesh.hide()
            i_mesh += 1
        self.render_window.Render()

    def show_point_cloud_at_index(self, i_point_cloud_to_show):
        if i_point_cloud_to_show < len(self.point_clouds):
            print("Point cloud:", self.point_cloud_names[i_point_cloud_to_show])
            self.shown_point_cloud_index = i_point_cloud_to_show
            i_point_cloud = 0
            for point_cloud in self.point_clouds:
                if i_point_cloud == i_point_cloud_to_show:
                    point_cloud.show()
                else:
                    point_cloud.hide()
                i_point_cloud += 1
            self.render_window.Render()

    def set_frame(self, i_frame):
        print("Frame:", i_frame)

        self.load_frame_meshes(i_frame)
        self.load_frame_point_clouds(i_frame)
        self.current_frame = i_frame

        self.render_window.Render()

    def advance_mesh(self):
        if self.shown_mesh_index == len(self.meshes) - 1:
            if self.current_frame < self.frame_index_upper_bound - 1:
                self.set_frame(self.current_frame + 1)
                self.show_mesh_at_index(0)
        else:
            self.show_mesh_at_index(self.shown_mesh_index + 1)

    def retreat_mesh(self):
        if self.shown_mesh_index == 0:
            if self.current_frame > self.start_frame_ix:
                self.set_frame(self.current_frame - 1)
                self.show_mesh_at_index(len(self.meshes) - 1)
        else:
            self.show_mesh_at_index(self.shown_mesh_index - 1)

    def advance_point_cloud(self):
        if len(self.point_clouds) > 0:
            if self.shown_point_cloud_index == len(self.point_clouds) - 1:
                if self.current_frame < self.frame_index_upper_bound - 1:
                    self.set_frame(self.current_frame + 1)
                    self.show_point_cloud_at_index(0)
            else:
                self.show_point_cloud_at_index(self.shown_point_cloud_index + 1)

    def retreat_point_cloud(self):
        if len(self.point_clouds) > 0:
            if self.shown_point_cloud_index == 0:
                if self.current_frame > self.start_frame_ix:
                    self.set_frame(self.current_frame - 1)
                    self.show_point_cloud_at_index(len(self.point_clouds) - 1)
            else:
                self.show_point_cloud_at_index(self.shown_point_cloud_index - 1)

    # Handle the mouse button events.
    def button_event(self, obj, event):
        if event == "LeftButtonPressEvent":
            self.rotating = True
        elif event == "LeftButtonReleaseEvent":
            self.rotating = False
        elif event == "RightButtonPressEvent":
            self.panning = True
        elif event == "RightButtonReleaseEvent":
            self.panning = False
        elif event == "MiddleButtonPressEvent":
            self.zooming = True
        elif event == "MiddleButtonReleaseEvent":
            self.zooming = False
        elif event == "MouseWheelForwardEvent":
            self.dolly_step(1)
        elif event == "MouseWheelBackwardEvent":
            self.dolly_step(-1)

    # General high-level logic
    def mouse_move(self, obj, event):
        last_x_y_pos = self.interactor.GetLastEventPosition()
        last_x = last_x_y_pos[0]
        last_y = last_x_y_pos[1]

        x_y_pos = self.interactor.GetEventPosition()
        x = x_y_pos[0]
        y = x_y_pos[1]

        center = self.render_window.GetSize()
        center_x = center[0] / 2.0
        center_y = center[1] / 2.0

        if self.rotating:
            self.rotate(x, y, last_x, last_y)
        elif self.panning:
            self.pan(x, y, last_x, last_y, center_x, center_y)
        elif self.zooming:
            self.dolly(x, y, last_x, last_y)

    def rotate(self, x, y, last_x, last_y):
        speed = 0.5
        self.camera.Azimuth(speed * (last_x - x))
        self.camera.Elevation(speed * (last_y - y))
        self.camera.SetViewUp(0, 0, 0)
        self.render_window.Render()

    def pan(self, x, y, last_x, last_y, center_x, center_y):
        renderer = self.renderer
        camera = self.camera
        f_point = camera.GetFocalPoint()
        f_point0 = f_point[0]
        f_point1 = f_point[1]
        f_point2 = f_point[2]

        p_point = camera.GetPosition()
        p_point0 = p_point[0]
        p_point1 = p_point[1]
        p_point2 = p_point[2]

        renderer.SetWorldPoint(f_point0, f_point1, f_point2, 1.0)
        renderer.WorldToDisplay()
        d_point = renderer.GetDisplayPoint()
        focal_depth = d_point[2]

        a_point0 = center_x + (x - last_x)
        a_point1 = center_y + (y - last_y)

        renderer.SetDisplayPoint(a_point0, a_point1, focal_depth)
        renderer.DisplayToWorld()
        r_point = renderer.GetWorldPoint()
        r_point0 = r_point[0]
        r_point1 = r_point[1]
        r_point2 = r_point[2]
        r_point3 = r_point[3]

        if r_point3 != 0.0:
            r_point0 = r_point0 / r_point3
            r_point1 = r_point1 / r_point3
            r_point2 = r_point2 / r_point3

        camera.SetFocalPoint((f_point0 - r_point0) / 2.0 + f_point0,
                             (f_point1 - r_point1) / 2.0 + f_point1,
                             (f_point2 - r_point2) / 2.0 + f_point2)
        camera.SetPosition((f_point0 - r_point0) / 2.0 + p_point0,
                           (f_point1 - r_point1) / 2.0 + p_point1,
                           (f_point2 - r_point2) / 2.0 + p_point2)
        self.render_window.Render()

    def dolly(self, x, y, last_x, last_y):
        dolly_factor = pow(1.02, (0.5 * (y - last_y)))
        camera = self.camera
        if camera.GetParallelProjection():
            parallel_scale = camera.GetParallelScale() * dolly_factor
            camera.SetParallelScale(parallel_scale)
        else:
            camera.Dolly(dolly_factor)
            self.renderer.ResetCameraClippingRange()

        self.render_window.Render()

    def dolly_step(self, step):
        dolly_factor = pow(1.02, (10.0 * step))
        camera = self.camera
        if camera.GetParallelProjection():
            parallel_scale = camera.GetParallelScale() * dolly_factor
            camera.SetParallelScale(parallel_scale)
        else:
            camera.Dolly(dolly_factor)
            self.renderer.ResetCameraClippingRange()

        self.render_window.Render()

    def keypress(self, obj, event):
        key = obj.GetKeySym()
        print("Key:", key)
        if key == "q" or key == "Escape":
            obj.InvokeEvent("DeleteAllObjects")
            sys.exit()
        elif key == "bracketright":
            if self.current_frame < self.frame_index_upper_bound - 1:
                self.set_frame(self.current_frame + 1)
        elif key == "bracketleft":
            if self.current_frame > 0:
                self.set_frame(self.current_frame - 1)
        elif key == "Right":
            self.advance_mesh()
        elif key == "Left":
            self.retreat_mesh()
        elif key == "Up":
            self.retreat_point_cloud()
        elif key == "Down":
            self.advance_point_cloud()
        elif key == "p":
            if self.__alt_pressed:
                print(self.render_window.GetPosition())
            else:
                if len(self.point_clouds) > 0:
                    self.point_clouds[self.shown_point_cloud_index].toggle_visibility()
                    self.render_window.Render()
        elif key == "m":
            self.meshes[self.shown_mesh_index].toggle_visibility()
            self.render_window.Render()
        elif key == "Alt_L" or key == "Alt_R":
            self.__alt_pressed = True

    def key_release(self, obj, event):
        key = obj.GetKeySym()
        if key == "Alt_L" or key == "Alt_R":
            self.__alt_pressed = False
