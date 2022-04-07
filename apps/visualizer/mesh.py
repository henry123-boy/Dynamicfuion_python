from pathlib import Path

import vtk
from enum import Enum


class MeshColorMode(Enum):
    COLORED_AMBIENT = 0
    UNIFORM_SHADED = 1

    def next(self):
        return MeshColorMode((self.value + 1) % len(self._member_map_))

    def previous(self):
        return MeshColorMode((self.value - 1) % len(self._member_map_))


class Mesh:
    FACTOR = 1.0

    def __init__(self, renderer, render_window, color):
        self.renderer = renderer
        self.render_window = render_window

        # Create a polydata object
        self.triangle_poly_data = vtk.vtkPolyData()

        # Create mapper and actor
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.triangle_poly_data)
        self.actor = vtk.vtkActor()
        self.actor.GetProperty().SetColor(color)
        self.actor.GetProperty().SetBackfaceCulling(True)
        self.reader = vtk.vtkPLYReader()
        self.actor.SetMapper(self.mapper)
        self.actor.SetOrientation(0, 0.0, 180)
        self.renderer.AddActor(self.actor)
        self.color_mode = MeshColorMode.COLORED_AMBIENT
        self.set_color_mode(self.color_mode)

    def set_color_mode(self, mode: MeshColorMode) -> None:
        if mode is MeshColorMode.COLORED_AMBIENT:
            self.mapper.ScalarVisibilityOn()
            self.mapper.Update()
            actor_properties = self.actor.GetProperty()
            actor_properties.SetAmbient(1)
            actor_properties.SetDiffuse(0)
            self.actor.Modified()
        elif mode is MeshColorMode.UNIFORM_SHADED:
            self.mapper.ScalarVisibilityOff()
            self.mapper.Update()
            actor_properties = self.actor.GetProperty()
            actor_properties.SetAmbient(0)
            actor_properties.SetDiffuse(1)
            self.actor.Modified()
        else:
            raise ValueError("Unsupported color mode:" + mode.name)
        self.color_mode = mode

    def update(self, path: Path, render: bool = False) -> None:
        self.reader.SetFileName(str(path))
        self.reader.Update()
        self.mapper.SetInputConnection(self.reader.GetOutputPort())
        self.mapper.Modified()
        self.set_color_mode(self.color_mode)
        if render:
            self.render_window.Render()

    def toggle_visibility(self) -> None:
        self.actor.SetVisibility(not self.actor.GetVisibility())

    def hide(self) -> None:
        self.actor.SetVisibility(False)

    def show(self) -> None:
        self.actor.SetVisibility(True)

    def is_visible(self) -> bool:
        return self.actor.GetVisibility()
