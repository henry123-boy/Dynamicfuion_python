import vtk


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

    def update(self, path, render=False):
        self.reader.SetFileName(path)
        self.reader.Update()
        self.mapper.SetInputConnection(self.reader.GetOutputPort())
        self.mapper.Modified()

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
