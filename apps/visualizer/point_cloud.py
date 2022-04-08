from pathlib import Path

import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as vtk_np
import numpy as np
from enum import Enum


class PointColorMode(Enum):
    UNIFORM = 0
    SOURCE_COLORED = 1

    def next(self):
        return PointColorMode((self.value + 1) % len(self._member_map_))

    def previous(self):
        return PointColorMode((self.value - 1) % len(self._member_map_))


class PointCloud:
    FACTOR = 1.0

    def __init__(self, renderer, render_window, color):
        self.renderer = renderer
        self.render_window = render_window

        self.points = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        self.scalars = None

        # Create a polydata object
        self.point_poly_data = vtk.vtkPolyData()

        # Create mapper and actor
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.point_poly_data)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.SetOrientation(0, 0.0, 180)
        self.actor.GetProperty().SetColor(color)
        self.actor.GetProperty().SetPointSize(2.0)
        self.renderer.AddActor(self.actor)
        self.color_mode = PointColorMode.UNIFORM
        self.set_color_mode(self.color_mode)

    def set_color_mode(self, mode: PointColorMode) -> None:
        if mode is PointColorMode.UNIFORM:
            self.mapper.ScalarVisibilityOff()
        elif mode is PointColorMode.SOURCE_COLORED:
            if self.scalars is None:
                print("Can't change point cloud to source-colored, source point colors missing")
                return
            self.mapper.ScalarVisibilityOn()
        else:
            raise ValueError("Unsupported color mode: " + mode.name)
        self.color_mode = mode

    def update(self, path_to_npy: Path, render: bool = False) -> None:
        points_np = np.load(str(path_to_npy))
        point_count = points_np.shape[0]

        if points_np.shape[1] > 3:
            self.points.SetData(vtk_np.numpy_to_vtk(points_np[:, 3:]))
            self.scalars = vtk_np.numpy_to_vtk((points_np[:, :3] * 255).astype(np.uint8))
            self.scalars.SetName("Colors")
        else:
            self.points.SetData(vtk_np.numpy_to_vtk(points_np[:, :3]))

        cells_numpy = np.vstack([np.ones(point_count, dtype=np.int64),
                                 np.arange(point_count, dtype=np.int64)]).T.flatten()
        self.cells.SetCells(point_count, vtk_np.numpy_to_vtkIdTypeArray(cells_numpy))
        self.points.Modified()
        self.cells.Modified()
        self.point_poly_data.SetPoints(self.points)
        self.point_poly_data.SetVerts(self.cells)
        if self.scalars is not None:
            self.point_poly_data.GetPointData().SetScalars(self.scalars)

        self.set_color_mode(self.color_mode)
        self.point_poly_data.Modified()

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
