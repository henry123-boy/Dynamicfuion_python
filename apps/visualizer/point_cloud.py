import vtk
import vtk.util.numpy_support as vtk_np
import numpy as np


class PointCloud:
    FACTOR = 1.0

    def __init__(self):
        self.array_numpy = None

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

    def update(self, path_to_npy, render=False):
        points_np = np.load(path_to_npy)
        point_count = points_np.shape[0]

        self.points.SetData(vtk_np.numpy_to_vtk(points_np))
        cells_numpy = np.vstack([np.ones(point_count, dtype=np.int64),
                                 np.arange(point_count, dtype=np.int64)]).T.flatten()
        self.cells.SetCells(point_count, vtk_np.numpy_to_vtkIdTypeArray(cells_numpy))
        self.points.Modified()
        self.cells.Modified()
        self.point_poly_data.SetPoints(self.points)
        self.point_poly_data.SetVerts(self.cells)
        self.point_poly_data.Modified()

        if render:
            self.render_window.Render()

    def toggle_visibility(self):
            self.actor.SetVisibility(not self.actor.GetVisibility())

    def hide(self):
        self.actor.SetVisibility(False)

    def show(self):
        self.actor.SetVisibility(True)

    def is_visible(self):
        return self.actor.GetVisibility()