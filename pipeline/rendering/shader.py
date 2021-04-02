import os
from OpenGL import GL as gl
from OpenGL import GLU as glu
from typing import List


def check_opengl_error():
    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        raise RuntimeError('GLERROR: ', glu.gluErrorString(error))


class Shader(object):
    def __init__(self, vertex_shader_source_list, fragment_shader_source_list):
        # create program
        self.program = gl.glCreateProgram()  # pylint: disable=E1111
        # print('create program ',self.program)
        check_opengl_error()

        # vertex shader
        # print('compile vertex shader...')
        self.vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)  # pylint: disable=E1111
        gl.glShaderSource(self.vertex_shader, vertex_shader_source_list)
        gl.glCompileShader(self.vertex_shader)
        if gl.GL_TRUE != gl.glGetShaderiv(self.vertex_shader, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(self.vertex_shader)
            raise Exception(error)
        gl.glAttachShader(self.program, self.vertex_shader)
        check_opengl_error()

        # fragment shader
        # print('compile fragment shader...')
        self.fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)  # pylint: disable=E1111
        gl.glShaderSource(self.fragment_shader, fragment_shader_source_list)
        gl.glCompileShader(self.fragment_shader)
        if gl.GL_TRUE != gl.glGetShaderiv(self.fragment_shader, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(self.fragment_shader)
            raise Exception(error)
        gl.glAttachShader(self.program, self.fragment_shader)
        check_opengl_error()

        # print('link...')
        gl.glLinkProgram(self.program)
        if gl.GL_TRUE != gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS):
            error = gl.glGetShaderInfoLog(self.vertex_shader)
            raise Exception(error)
        check_opengl_error()

    def begin(self):
        if gl.glUseProgram(self.program):
            check_opengl_error()

    def end(self):
        gl.glUseProgram(0)


def init_shader_from_glsl(vertex_shader_paths: List[str], fragment_shader_paths: List[str]) -> Shader:
    vertex_shader_source_list = []
    fragment_shader_source_list = []
    if isinstance(vertex_shader_paths, list):
        for GLSL in vertex_shader_paths:
            absolute_directory = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), "shaders"), GLSL))
            file = open(absolute_directory, 'rb')
            vertex_shader_source_list.append(file.read())
            file.close()
        for GLSL in fragment_shader_paths:
            absolute_directory = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), "shaders"), GLSL))
            file = open(absolute_directory, 'rb')
            fragment_shader_source_list.append(file.read())
            file.close()
        return Shader(vertex_shader_source_list, fragment_shader_source_list)
    else:
        raise ValueError("Expected vertex_shader_paths to be of type " + str(list))
