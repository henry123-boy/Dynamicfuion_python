# standard library
from multiprocessing import Queue
from pathlib import Path
from typing import Union
import sys

# third-party
import vtkmodules.all as vtk

# local (visualizer app & apps.shared)
from apps.shared.screen_management import set_up_render_window_bounds
from apps.shared.app import App
from apps.visualizer import utilities
from apps.visualizer.mesh import Mesh, MeshColorMode
from apps.visualizer.point_cloud import PointCloud, PointColorMode
from apps.visualizer.correspondence_line_set import CorrespondenceColorMode, CorrespondenceLineSet
from apps.visualizer.graph import Graph


class VisualizerApp(App):

    def __init__(self, output_path: Path, start_frame_ix: int = -1, outgoing_queue: Union[None, Queue] = None):
        self.__alt_pressed = False
        self.outgoing_queue = outgoing_queue

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
        self.render_window.SetWindowName("Visualizer: " + str(output_path))
        set_up_render_window_bounds(self.render_window, None, 1)
        self.render_window.AddRenderer(self.renderer)

        # mesh setup
        self.canonical_mesh = Mesh(self.renderer, self.render_window, colors.GetColor3d("Peacock"))
        self.warped_live_mesh = Mesh(self.renderer, self.render_window, colors.GetColor3d("Green"))
        self.shown_mesh_index = 0

        self.meshes = [self.canonical_mesh, self.warped_live_mesh]
        self.mesh_names = ["canonical", "warped live"]
        self.mesh_color_mode: MeshColorMode = MeshColorMode.COLORED_AMBIENT

        self.shown_point_cloud_index = 0

        # point cloud setup
        self.have_source_and_target_point_clouds = utilities.source_and_target_point_clouds_are_present(
            self.start_frame_ix, output_path)
        gn_iteration_count = utilities.get_gn_iteration_count(self.start_frame_ix, output_path)
        self.point_color_mode = PointColorMode.UNIFORM
        self.point_clouds = []
        self.point_cloud_names = []
        # source
        if self.have_source_and_target_point_clouds:
            self.point_clouds.append(PointCloud(self.renderer, self.render_window, colors.GetColor3d("Blue")))
            self.point_cloud_names.append("source")
        # GN iterations
        for i_point_cloud in range(0, gn_iteration_count):
            self.point_clouds.append(PointCloud(self.renderer, self.render_window, colors.GetColor3d("White")))
            self.point_cloud_names.append(f"GN iteration {i_point_cloud:03d}")
        # target
        if self.have_source_and_target_point_clouds:
            self.point_clouds.append(PointCloud(self.renderer, self.render_window, colors.GetColor3d("Red")))
            self.point_cloud_names.append("target")

        # correspondence line set setup
        self.have_correspondence_info = utilities.correspondence_info_is_present(self.start_frame_ix, output_path)
        self.correspondence_line_sets = []
        self.correspondence_color_mode = CorrespondenceColorMode.PREDICTION_WEIGHTED
        self.correspondence_set = None
        if self.have_correspondence_info:
            self.correspondence_set = CorrespondenceLineSet(self.renderer, self.render_window,
                                                            colors.GetColor3d("Grey"))

        # graph setup
        self.have_graph_info = utilities.graph_info_is_present(self.start_frame_ix, output_path)
        self.graph = None
        if self.have_graph_info:
            self.graph = Graph(self.renderer, self.render_window, colors.GetColor3d("Green"))

        # text setup
        self.text_mapper = vtk.vtkTextMapper()
        shown_point_cloud = "none" if len(self.point_cloud_names) == 0 \
            else self.point_cloud_names[self.shown_point_cloud_index]
        self.text_mapper.SetInput(f"Frame: {start_frame_ix:d}\n"
                                  f"Showing mesh: {self.mesh_names[self.shown_mesh_index]:s}\n"
                                  f"Mesh color mode: f{self.mesh_color_mode.name:s}\n"
                                  f"Showing point cloud: {shown_point_cloud:s}\n"
                                  f"Point color mode: f{self.point_color_mode.name:s}"
                                  f"Visible correspondences: 0%\n"
                                  f"Corresp. color mode: {self.correspondence_color_mode.name:s}\n")

        text_lines = self.text_mapper.GetInput().splitlines()
        self.number_of_text_lines = len(text_lines)
        text_property = self.text_mapper.GetTextProperty()
        self.font_size = 20
        text_property.SetFontSize(self.font_size)
        text_property.SetColor(colors.GetColor3d('Mint'))

        self.text_actor = vtk.vtkActor2D()
        self.text_actor.SetMapper(self.text_mapper)
        self.last_window_width, self.last_window_height = 0, 0
        self.renderer.AddActor(self.text_actor)

        # background
        self.renderer.SetBackground(colors.GetColor3d("Black"))

        # Interactor setup
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(None)
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.Initialize()

        self.interactor.AddObserver("ModifiedEvent", self.update_window)
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

        # set up camera
        camera.SetPosition(self.offset_cam[0], self.offset_cam[1], self.offset_cam[2])
        camera.SetPosition(0, 0, -1)
        camera.SetViewUp(0, 1.0, 0)
        camera.SetFocalPoint(0, 0, 1.5)
        camera.SetClippingRange(0.01, 10.0)

        self.render_window.Render()

        self.update_window(None, None)
        self.set_frame(self.current_frame)
        self.show_mesh_at_index(self.shown_mesh_index)
        self.show_point_cloud_at_index(self.shown_point_cloud_index)  # hide all but one GN point cloud

        # start with point cloud hidden
        if len(self.point_clouds) > self.shown_point_cloud_index:
            self.point_clouds[self.shown_point_cloud_index].hide()
            self.update_text()

        # start with correspondences hidden
        if self.correspondence_set is not None:
            self.correspondence_set.hide()

        self.update_text()
        self.render_window.Render()

        self._pixel_labels_visible = False

    def load_frame_meshes(self, i_frame):
        canonical_path = self.output_path / f"{i_frame:06d}_canonical_mesh.ply"
        self.canonical_mesh.update(canonical_path)
        warped_live_path = self.output_path / f"{i_frame:06d}_warped_mesh.ply"
        self.warped_live_mesh.update(warped_live_path)

    def load_frame_point_clouds(self, i_frame):
        i_gn_iteration = 0
        i_point_cloud_index = 0
        for point_cloud in self.point_clouds:
            if self.have_source_and_target_point_clouds and i_point_cloud_index == 0:
                point_cloud.update(self.output_path / f"{i_frame:06d}_source_rgbxyz.npy")
            elif self.have_source_and_target_point_clouds and i_point_cloud_index == len(self.point_clouds) - 1:
                point_cloud.update(self.output_path / f"{i_frame:06d}_target_rgbxyz.npy")
            else:
                point_cloud.update(self.output_path / f"{i_frame:06d}_deformed_points_iter_{i_gn_iteration:03d}.npy")
                i_gn_iteration += 1
            i_point_cloud_index += 1

    def load_frame_correspondences(self, i_frame):
        if self.correspondence_set is not None:
            self.correspondence_set.update(self.output_path / f"{i_frame:06d}_source_rgbxyz.npy",
                                           self.output_path / f"{i_frame:06d}_target_matches.npy",
                                           self.output_path / f"{i_frame:06d}_valid_correspondence_mask.npy",
                                           self.output_path / f"{i_frame:06d}_prediction_mask.npy")

    def load_frame_graph(self, i_frame):
        if self.graph is not None:
            self.graph.update(self.output_path / f"{i_frame:06d}_nodes.npy",
                              self.output_path / f"{i_frame:06d}_edges.npy",
                              self.output_path / f"{i_frame:06d}_translations.npy")

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
        self.update_text()
        self.render_window.Render()

    def cycle_mesh_color_mode(self):
        self.mesh_color_mode = self.mesh_color_mode.next()
        for mesh in self.meshes:
            mesh.set_color_mode(self.mesh_color_mode)
        self.update_text()

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
            self.update_text()
            self.render_window.Render()

    def cycle_point_color_mode(self):
        self.point_color_mode = self.point_color_mode.next()
        for point_cloud in self.point_clouds:
            point_cloud.set_color_mode(self.point_color_mode)
        self.update_text()

    def cycle_correspondence_color_mode(self):
        self.correspondence_color_mode = self.correspondence_color_mode.next()
        self.correspondence_set.set_color_mode(self.correspondence_color_mode)
        self.update_text()

    def set_frame(self, i_frame):
        print("Frame:", i_frame)

        self.load_frame_meshes(i_frame)
        self.load_frame_point_clouds(i_frame)
        self.load_frame_correspondences(i_frame)
        self.load_frame_graph(i_frame)
        self.current_frame = i_frame

        self.update_text()
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

    KEYS_TO_SEND = {"q", "Escape", "bracketright", "bracketleft"}

    def keypress(self, obj, event):
        key = obj.GetKeySym()
        # send key to linked process, if any
        if key in VisualizerApp.KEYS_TO_SEND and self.outgoing_queue is not None:
            self.outgoing_queue.put(key)
        self.handle_key(key)

    def handle_key(self, key):
        print("Key:", key)
        # ==== window controls ===========
        if key == "q" or key == "Escape":
            self.interactor.InvokeEvent("DeleteAllObjects")
            sys.exit()
        # ==== frame controls ===========
        elif key == "bracketright":
            if self.current_frame < self.frame_index_upper_bound - 1:
                self.set_frame(self.current_frame + 1)
        elif key == "bracketleft":
            if self.current_frame > self.start_frame_ix:
                self.set_frame(self.current_frame - 1)
        # ==== mesh controls ===========
        elif key == "Right":
            self.advance_mesh()
        elif key == "Left":
            self.retreat_mesh()
        elif key == "m":
            if self.__alt_pressed:
                self.cycle_mesh_color_mode()
                self.render_window.Render()
            else:
                self.meshes[self.shown_mesh_index].toggle_visibility()
                self.update_text()
                self.render_window.Render()
        # ==== correspondence controls ===========
        elif key == "c":
            if self.correspondence_set is not None:
                if self.__alt_pressed:
                    self.cycle_correspondence_color_mode()
                    self.render_window.Render()
                else:
                    self.correspondence_set.toggle_visibility()
                    self.update_text()
                    self.render_window.Render()
        elif key == "period":
            if self.correspondence_set is not None:
                self.correspondence_set.increase_visible_match_ratio()
                self.update_text()
                self.render_window.Render()
        elif key == "comma":
            if self.correspondence_set is not None:
                self.correspondence_set.decrease_visible_match_ratio()
                self.update_text()
                self.render_window.Render()
        # ==== point cloud controls ===========
        elif key == "p":
            if self.__alt_pressed:
                self.cycle_point_color_mode()
                self.render_window.Render()
            else:
                if len(self.point_clouds) > 0:
                    self.point_clouds[self.shown_point_cloud_index].toggle_visibility()
                    self.update_text()
                    self.render_window.Render()
        elif key == "Up":
            self.retreat_point_cloud()
        elif key == "Down":
            self.advance_point_cloud()
        # ==== graph controls =================
        elif key == "g":
            if self.graph is not None:
                self.graph.toggle_visibility()
                self.render_window.Render()
        # ==== modifier keys ==================
        elif key == "Alt_L" or key == "Alt_R":
            self.__alt_pressed = True

    def key_release(self, obj, event):
        key = obj.GetKeySym()
        if key == "Alt_L" or key == "Alt_R":
            self.__alt_pressed = False

    def update_window(self, obj, event):
        (window_width, window_height) = self.render_window.GetSize()
        if window_width != self.last_window_width or window_height != self.last_window_height:
            self.text_actor.SetDisplayPosition(window_width - 500,
                                               window_height - (self.number_of_text_lines + 2) * self.font_size)
            self.last_window_width = window_width
            self.last_window_height = window_height

    def update_text(self):
        if self.canonical_mesh.is_visible():
            mesh_mode = "canonical"
        elif self.warped_live_mesh.is_visible():
            mesh_mode = "warped live"
        else:
            mesh_mode = "none"

        if len(self.point_clouds) > 0 and self.point_clouds[self.shown_point_cloud_index].is_visible():
            point_cloud_iteration_text = self.point_cloud_names[self.shown_point_cloud_index]
        else:
            point_cloud_iteration_text = "none"
        showing_correspondences = self.correspondence_set is not None and self.correspondence_set.is_visible()
        visible_correspondence_percentage = 0 if not showing_correspondences else \
            self.correspondence_set.visible_match_percentage
        self.text_mapper.SetInput(f"Frame: {self.current_frame:d}\n"
                                  f"Showing mesh: {mesh_mode:s}\n"
                                  f"Mesh color mode: {self.mesh_color_mode.name:s}\n"
                                  f"Showing point cloud: {point_cloud_iteration_text:s}\n"
                                  f"Point color mode: {self.point_color_mode.name:s}\n"
                                  f"Visible correspondences: {visible_correspondence_percentage}%\n"
                                  f"Corresp. color mode: {self.correspondence_color_mode.name:s}\n")


