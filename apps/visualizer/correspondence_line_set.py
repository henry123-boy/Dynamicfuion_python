from pathlib import Path

import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as vtk_np
import numpy as np
from enum import Enum


class CorrespondenceColorMode(Enum):
    UNIFORM = 0
    PREDICTION_WEIGHTED = 1

    def next(self):
        return CorrespondenceColorMode((self.value + 1) % len(self._member_map_))

    def previous(self):
        return CorrespondenceColorMode((self.value - 1) % len(self._member_map_))


class CorrespondenceLineSet:

    def __init__(self, renderer, render_window, color):
        self.correspondence_weight_threshold = 0.3
        self.renderer = renderer
        self.render_window = render_window

        # setup the look up tabe
        self.lut = vtk.vtkLookupTable()
        self.build_lut()

        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        self.scalars = None

        # Create a polydata object
        self.line_poly_data = vtk.vtkPolyData()

        # Create mapper and actor
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.line_poly_data)
        self.mapper.SetLookupTable(self.lut)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.SetOrientation(0, 0.0, 180)
        self.actor.GetProperty().SetColor(color)
        self.actor.GetProperty().SetPointSize(2.0)
        self.renderer.AddActor(self.actor)
        self.color_mode = CorrespondenceColorMode.UNIFORM
        self.set_color_mode(self.color_mode)

    def build_lut(self):
        """
        Creates the lookup table
        Returns:
            - lut (vtkLookupTable): lookup table with red=max, blue=min
        """

        self.lut.SetHueRange(0.667, 0)
        self.lut.Build()

    def set_color_mode(self, mode: CorrespondenceColorMode) -> None:
        if mode is CorrespondenceColorMode.UNIFORM:
            self.mapper.ScalarVisibilityOff()
        elif mode is CorrespondenceColorMode.PREDICTION_WEIGHTED:
            if self.scalars is None:
                print("Can't change correspondence line set to colored by prediction weight, "
                      "weight-based colors are missing")
                return
            self.mapper.ScalarVisibilityOn()
        else:
            raise ValueError("Unsupported color mode: " + mode.name)
        self.color_mode = mode

    def update(self,
               path_to_source_rgbxyz: Path,
               path_to_target_matches: Path,
               path_to_correspondence_mask: Path,
               path_to_prediction_mask: Path,
               render: bool = False) -> None:

        source_points_np = np.load(str(path_to_source_rgbxyz))[:, 3:]
        # these are like motion vectors of the source points
        target_matches_np = np.load(str(path_to_target_matches))
        correspondence_mask = np.load(str(path_to_correspondence_mask)).flatten()
        correspondence_weights = np.load(str(path_to_prediction_mask)).flatten()
        good_match_mask = correspondence_mask & (correspondence_weights > self.correspondence_weight_threshold)

        good_source_points_np = source_points_np[good_match_mask]
        good_target_matches_np = target_matches_np[good_match_mask]
        correspondence_weights = correspondence_weights[good_match_mask]

        good_match_points = np.concatenate([good_source_points_np, good_target_matches_np], axis=0)
        line_count = len(good_source_points_np)

        self.points.SetData(vtk_np.numpy_to_vtk(good_match_points))

        # cells_numpy = np.vstack([np.arange(line_count, dtype=np.int64),
        #                          np.arange(line_count, line_count * 2, dtype=np.int64)]).T.flatten()
        # self.lines.SetCells(line_count, vtk_np.numpy_to_vtkIdTypeArray(cells_numpy))
        self.lines.Reset()
        for i_line in range(line_count):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i_line)
            line.GetPointIds().SetId(1, i_line+line_count)
            self.lines.InsertNextCell(line)

        self.points.Modified()
        self.lines.Modified()
        self.line_poly_data.SetPoints(self.points)
        self.line_poly_data.SetLines(self.lines)

        colors_np = np.tile(np.tile((correspondence_weights * 255).astype(np.uint8), (3, 1)).T, (2, 1))
        self.scalars = vtk_np.numpy_to_vtk(colors_np)
        self.scalars.SetName("Colors")

        self.line_poly_data.GetPointData().SetScalars(self.scalars)

        self.set_color_mode(self.color_mode)
        self.line_poly_data.Modified()

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
