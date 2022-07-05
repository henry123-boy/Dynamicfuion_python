# standard library
from multiprocessing import Queue
from pathlib import Path
from typing import Union
import sys

# third-party
import vtkmodules.all as vtk

# local (visualizer app & apps.shared)
from apps.shared.generic_3d_viewer_app import Generic3DViewerApp
from apps.visualizer import utilities
from apps.visualizer.mesh import Mesh, MeshColorMode
from apps.visualizer.point_cloud import PointCloud, PointColorMode
from apps.visualizer.correspondence_line_set import CorrespondenceColorMode, CorrespondenceLineSet
from apps.visualizer.graph import Graph


class VisualizerApp(Generic3DViewerApp):

    def __init__(self, output_path: Path, start_frame_ix: int = -1, outgoing_queue: Union[None, Queue] = None):
        super().__init__()
        self.__alt_pressed = False
        self.outgoing_queue = outgoing_queue

        minimum_start_frame, self.frame_index_upper_bound = utilities.get_start_and_end_frame(output_path)

        if start_frame_ix == -1:
            start_frame_ix = minimum_start_frame
        elif start_frame_ix < minimum_start_frame:
            raise ValueError(f"Smallest start frame for given sequence is {minimum_start_frame:d}, "
                             f"cannot start from frame {start_frame_ix:d}.")

        self.start_frame_ix = start_frame_ix
        self.output_path = output_path
        self.current_frame = start_frame_ix

        colors = vtk.vtkNamedColors()

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

        # === text setup ===
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
        # text actor
        self.text_actor = vtk.vtkActor2D()
        self.text_actor.SetMapper(self.text_mapper)
        self.last_window_width, self.last_window_height = 0, 0
        self.renderer.AddActor(self.text_actor)

        # set up camera
        self.offset_cam = (0.2562770766576, 0.13962609403401335, -0.2113334598208764)
        self.camera.SetPosition(self.offset_cam[0], self.offset_cam[1], self.offset_cam[2])
        self.camera.SetPosition(0, 0, -1)
        self.camera.SetViewUp(0, 1.0, 0)
        self.camera.SetFocalPoint(0, 0, 1.5)
        self.camera.SetClippingRange(0.01, 10.0)

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
        self._pixel_labels_visible = False
        self.render_window.Render()

    def launch(self):
        # Start the event loop.
        self.interactor.Start()

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

    KEYS_TO_SEND = {"q", "Escape", "bracketright", "bracketleft"}

    def key_press(self, obj, event):
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
