from pathlib import Path

import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as vtk_np
import numpy as np


class Graph:

    def __init__(self, renderer, render_window, color):
        self.renderer = renderer
        self.render_window = render_window

        self.points = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        self.lines = vtk.vtkCellArray()
        self.scalars = None

        # Create a polydata object
        self.node_poly_data = vtk.vtkPolyData()
        self.line_poly_data = vtk.vtkPolyData()

        # Create source geometry
        self.point_geometry = vtk.vtkSphereSource()
        self.point_geometry.SetRadius(0.01)

        # Create mappers and actors
        self.edge_mapper = vtk.vtkPolyDataMapper()
        self.edge_mapper.SetInputData(self.line_poly_data)
        self.node_mapper = vtk.vtkGlyph3DMapper()
        self.node_mapper.SetSourceConnection(self.point_geometry.GetOutputPort())
        self.node_mapper.ScalarVisibilityOff()
        self.node_mapper.ScalingOff()
        self.node_mapper.SetInputData(self.node_poly_data)

        self.edge_actor = vtk.vtkActor()
        self.edge_actor.SetMapper(self.edge_mapper)
        self.edge_actor.SetOrientation(0, 0.0, 180)
        self.edge_actor.GetProperty().SetColor(color)
        self.edge_actor.GetProperty().SetPointSize(2.0)

        self.node_actor = vtk.vtkActor()
        self.node_actor.SetMapper(self.node_mapper)
        self.node_actor.SetOrientation(0, 0.0, 180)
        self.node_actor.GetProperty().SetColor(color)
        self.node_actor.GetProperty().SetPointSize(2.0)
        
        self.renderer.AddActor(self.node_actor)
        self.renderer.AddActor(self.edge_actor)

    def update(self,
               path_to_nodes_npy: Path,
               path_to_edges_npy: Path,
               path_to_translations_npy: Path,
               render: bool = False) -> None:
        nodes = np.load(str(path_to_nodes_npy))
        translations = np.load(str(path_to_translations_npy))
        edges = np.load(str(path_to_edges_npy))

        points_np = nodes + translations

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

        self.lines.Reset()
        for node_id, edges in enumerate(edges):
            for neighbor_id in edges:
                if neighbor_id == -1:
                    break
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, node_id)
                line.GetPointIds().SetId(1, neighbor_id)
                self.lines.InsertNextCell(line)
                # edges_pairs.append([node_id, neighbor_id])

        self.points.Modified()
        self.cells.Modified()
        self.lines.Modified()
        self.node_poly_data.SetPoints(self.points)
        self.node_poly_data.SetVerts(self.cells)
        self.line_poly_data.SetPoints(self.points)
        self.line_poly_data.SetLines(self.lines)

        if self.scalars is not None:
            self.node_poly_data.GetPointData().SetScalars(self.scalars)

        self.node_poly_data.Modified()

        if render:
            self.render_window.Render()

    def toggle_visibility(self) -> None:
        self.node_actor.SetVisibility(not self.node_actor.GetVisibility())
        self.edge_actor.SetVisibility(not self.edge_actor.GetVisibility())

    def hide(self) -> None:
        self.node_actor.SetVisibility(False)
        self.edge_actor.SetVisibility(False)

    def show(self) -> None:
        self.node_actor.SetVisibility(True)
        self.edge_actor.SetVisibility(True)

    def is_visible(self) -> bool:
        return self.node_actor.GetVisibility()
