import open3d as o3d
import numpy as np
import sys
from OpenGL import GL as gl
import glfw
import glm
import ctypes
from pipeline.rendering.shader import Shader, init_shader_from_glsl

PROGRAM_EXIT_SUCCESS = 0

g_vertex_buffer_data = [
    -1.0, -1.0, -1.0,
    -1.0, -1.0, 1.0,
    -1.0, 1.0, 1.0,
    1.0, 1.0, -1.0,
    -1.0, -1.0, -1.0,
    -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0,
    -1.0, -1.0, -1.0,
    1.0, -1.0, -1.0,
    1.0, 1.0, -1.0,
    1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0,
    -1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0,
    -1.0, -1.0, 1.0,
    -1.0, -1.0, -1.0,
    -1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0,
    1.0, -1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, -1.0, -1.0,
    1.0, 1.0, -1.0,
    1.0, -1.0, -1.0,
    1.0, 1.0, 1.0,
    1.0, -1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0,
    1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0,
    -1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    -1.0, 1.0, 1.0,
    1.0, -1.0, 1.0]
g_color_buffer_data = [
    0.583, 0.771, 0.014,
    0.609, 0.115, 0.436,
    0.327, 0.483, 0.844,
    0.822, 0.569, 0.201,
    0.435, 0.602, 0.223,
    0.310, 0.747, 0.185,
    0.597, 0.770, 0.761,
    0.559, 0.436, 0.730,
    0.359, 0.583, 0.152,
    0.483, 0.596, 0.789,
    0.559, 0.861, 0.639,
    0.195, 0.548, 0.859,
    0.014, 0.184, 0.576,
    0.771, 0.328, 0.970,
    0.406, 0.615, 0.116,
    0.676, 0.977, 0.133,
    0.971, 0.572, 0.833,
    0.140, 0.616, 0.489,
    0.997, 0.513, 0.064,
    0.945, 0.719, 0.592,
    0.543, 0.021, 0.978,
    0.279, 0.317, 0.505,
    0.167, 0.620, 0.077,
    0.347, 0.857, 0.137,
    0.055, 0.953, 0.042,
    0.714, 0.505, 0.345,
    0.783, 0.290, 0.734,
    0.722, 0.645, 0.174,
    0.302, 0.455, 0.848,
    0.225, 0.587, 0.040,
    0.517, 0.713, 0.338,
    0.053, 0.959, 0.120,
    0.393, 0.621, 0.362,
    0.673, 0.211, 0.457,
    0.820, 0.883, 0.371,
    0.982, 0.099, 0.879
]


def main():
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("output/mesh_000000_red_shorts.ply")
    vertex_positions = np.array(mesh.vertices)
    triangle_indices = np.array(mesh.triangles)
    width = 500
    height = 500

    # initialize window
    glfw.init()
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(1024, 768, "title of the window", None, None)
    glfw.make_context_current(window)

    # set up OpenGL background & options
    gl.glClearColor(0.0, 0, 0.4, 0)
    gl.glDepthFunc(gl.GL_LESS)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # init shaders
    shader = init_shader_from_glsl(["vertex.glsl"], ["fragment.glsl"])

    # vertex_buffer = gl.glGenBuffers(1)
    # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
    # gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_positions.nbytes, vertex_positions, gl.GL_STREAM_DRAW)

    mvp_id = gl.glGetUniformLocation(shader.program, "MVP")

    projection_matrix = glm.perspective(glm.radians(45.0), 800.0 / 480.0, 0.1, 100.0)
    view_matrix = glm.lookAt(glm.vec3(4, 3, -3),  # Camera is at (4,3,-3), in World Space
                      glm.vec3(0, 0, 0),  # and looks at the (0.0.0))
                      glm.vec3(0, 1, 0))  # Head is up (set to 0,-1,0 to look upside-down)

    model_matrix = glm.mat4(1.0)
    movel_view_projection_matrix = projection_matrix * view_matrix * model_matrix
    # print context.MVP
    # exit(0)

    vertex_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, len(g_vertex_buffer_data) * 4, (gl.GLfloat * len(g_vertex_buffer_data))(*g_vertex_buffer_data), gl.GL_STATIC_DRAW)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    color_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_buffer)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, len(g_color_buffer_data) * 4, (gl.GLfloat * len(g_color_buffer_data))(*g_color_buffer_data), gl.GL_STATIC_DRAW)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 1)

    while True:
        # print self.context.MVP
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        shader.begin()
        # gl.glUseProgram(shader.program)
        gl.glUniformMatrix4fv(mvp_id, 1, gl.GL_FALSE, glm.value_ptr(movel_view_projection_matrix))

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_voidp(0))  # None means ctypes.c_voidp(0)

        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_voidp(0))

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 12 * 3)  # 12*3 indices starting at 0 -> 12 triangles

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)


        # shader.end()
        glfw.swap_buffers(window)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
