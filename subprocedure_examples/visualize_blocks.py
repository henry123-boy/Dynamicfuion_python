# third-party
import numpy as np
import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as vtk_np

# local
from apps.shared.generic_3d_viewer_app import Generic3DViewerApp


class BlockVisualizerApp(Generic3DViewerApp):
    def __init__(self, block_positions: np.ndarray, block_size: float) -> None:
        super().__init__()
        # set up camera
        self.offset_cam = (0.2562770766576, 0.13962609403401335, -0.2113334598208764)
        self.camera.SetPosition(self.offset_cam[0], self.offset_cam[1], self.offset_cam[2])
        self.camera.SetPosition(0, 0, -1)
        self.camera.SetViewUp(0, 1.0, 0)
        self.camera.SetFocalPoint(0, 0, 1.5)
        self.camera.SetClippingRange(0.01, 10.0)

        # set up block polydata, locations, "verts", and labels
        self.block_polydata = vtk.vtkPolyData()
        self.block_locations = vtk.vtkPoints()
        self.block_locations.SetData(vtk_np.numpy_to_vtk(block_positions * block_size))
        self.vertices = vtk.vtkCellArray()
        self.block_labels = vtk.vtkStringArray()
        self.block_labels.SetName("label")
        self.block_labels.SetNumberOfValues(len(block_positions))

        for i_block, block_position in enumerate(block_positions):
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(i_block)
            label = "({:d}, {:d}, {:d})".format(block_position[0],
                                                block_position[1],
                                                block_position[2])
            self.block_labels.SetValue(i_block, label)

        self.block_polydata.SetPoints(self.block_locations)
        self.block_polydata.SetVerts(self.vertices)
        self.block_polydata.GetPointData().AddArray(self.block_labels)

        self.block_label_hierarchy = vtk.vtkPointSetToLabelHierarchy()
        self.block_label_hierarchy.SetInputData(self.block_polydata)
        self.block_label_hierarchy.SetLabelArrayName("label")

        self.block_label_placement_mapper = vtk.vtkLabelPlacementMapper()
        self.block_label_placement_mapper.SetInputConnection(self.block_label_hierarchy.GetOutputPort())
        self.block_label_placement_mapper.SetRenderStrategy(vtk.vtkFreeTypeLabelRenderStrategy())
        self.block_label_placement_mapper.SetShapeToRoundedRect()
        self.block_label_placement_mapper.SetBackgroundColor(1.0, 1.0, 0.7)
        self.block_label_placement_mapper.SetBackgroundOpacity(0.4)
        self.block_label_placement_mapper.SetMargin(3)

        self.block_mapper = vtk.vtkGlyph3DMapper()
        self.block_mapper.SetInputData(self.block_polydata)

        # set up geometry source for all the blocks
        block = vtk.vtkCubeSource()
        block.SetXLength(block_size)
        block.SetYLength(block_size)
        block.SetZLength(block_size)
        self.block_mapper.SetSourceConnection(block.GetOutputPort())

        colors = vtk.vtkNamedColors()

        # block actor
        self.block_actor = vtk.vtkActor()
        self.block_actor.SetMapper(self.block_mapper)
        self.block_actor.GetProperty().SetColor(colors.GetColor3d("PowderBlue"))
        self.block_actor.GetProperty().SetLineWidth(0.1)
        # self.block_actor.GetProperty().SetRepresentationToWireframe()

        # label actor
        self.block_label_actor = vtk.vtkActor2D()
        self.block_label_actor.SetMapper(self.block_label_placement_mapper)

        self.renderer.AddActor(self.block_actor)
        self.renderer.AddActor(self.block_label_actor)

        self.render_window.Render()


def visualize_grid_aligned_blocks(block_positions_in_blocks: np.ndarray, block_size: float) -> None:
    app = BlockVisualizerApp(block_positions_in_blocks, block_size)
    app.launch()


if __name__ == "__main__":
    visualize_grid_aligned_blocks(np.array([[0, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 1],
                                            [1, 1, 0]]), 0.2)
