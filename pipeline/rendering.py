import open3d as o3d
import numpy as np
import sys
import vtk
import OpenGL.GL as gl
import vtkmodules.all as vtk

PROGRAM_EXIT_SUCCESS = 0




def make_vtk_id_list(it):
    """
    Makes a vtkIdList from a Python iterable. I'm kinda surprised that
     this is necessary, since I assumed that this kind of thing would
     have been built into the wrapper and happen transparently, but it
     seems not.

    :param it: A python iterable.
    :return: A vtkIdList
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


def main():
    # mesh : o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("output/mesh_000000_red_shorts.ply")
    # vertex_positions = np.array(mesh.vertices)
    # triangle_indices = np.array(mesh.triangles)
    # width = 500
    # height = 500
    # vertex_buffer = gl.glGenBuffers(1)
    # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
    # gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_positions.nbytes, vertex_positions, gl.GL_STREAM_DRAW)
    # gl.glClear(gl.GL_COLOR_BUFFER_BIT, gl.GL_DEPTH_BUFFER_BIT)
    #

    colors = vtk.vtkNamedColors()

    # x = array of 8 3-tuples of float representing the vertices of a cube:
    x = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
         (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]

    # pts = array of 6 4-tuples of vtkIdType (int) representing the faces
    #     of the cube in terms of the above vertices
    pts = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
           (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]

    points = vtk.vtkPoints()
    polygons = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()

    # Load the point, cell, and data attributes.
    for i, xi in enumerate(x):
        points.InsertPoint(i, xi)
    for pt in pts:
        polygons.InsertNextCell(make_vtk_id_list(pt))
    for i, _ in enumerate(x):
        scalars.InsertTuple1(i, i)

    cube = vtk.vtkPolyData()
    # We now assign the pieces to the vtkPolyData.
    cube.SetPoints(points)
    cube.SetPolys(polygons)
    # cube.GetPointData().SetScalars(scalars)

    filename = "output/mesh_000000_red_shorts.ply"
    #
    # reader = vtk.vtkPLYReader()
    # reader.SetFileName(filename)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(cube)

    # uncomment to use PLY mesh from disk
    # mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuse(1.0)
    # actor.GetProperty().SetDiffuse(0.8)
    # actor.GetProperty().SetDiffuseColor(colors.GetColor3d('LightSteelBlue'))
    # actor.GetProperty().SetSpecular(0.0)
    # actor.GetProperty().SetSpecularPower(0.0)

    camera_and_light_position = (2, 2, 2)

    light = vtk.vtkLight()
    light.PositionalOn()
    light.SetPosition(*camera_and_light_position)
    light.SetColor(1.0, 1.0, 1.0)
    light.SetIntensity(1.0)
    light.SetAttenuationValues(0.0, 1.0, 0.0)

    # Create a rendering window and renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(render_window.GetScreenSize())
    render_window.AddRenderer(renderer)
    render_window.SetWindowName('Test')

    # Create a renderwindowinteractor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.AddObserver('KeyPressEvent', keypress_callback, 1.0)
    interactor.SetRenderWindow(render_window)

    # Assign actor to the renderer
    renderer.AddActor(actor)
    renderer.AddLight(light)

    camera = renderer.GetActiveCamera()
    camera.SetPosition(*camera_and_light_position)
    camera.SetFocalPoint(0, 0, 0)
    # camera.SetPosition(0, 0, -3.5)
    # camera.SetFocalPoint(0.0, 0.0, 2.0)
    # camera.SetViewUp(0, -1, 0)

    # Enable user interface interactor
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
