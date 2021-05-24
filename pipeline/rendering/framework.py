# Mario Rosasco, 2016
# adapted from framework.cpp, Copyright (C) 2010-2012 by Jason L. McKesson
# This file is licensed under the MIT License.
#
# NB: Unlike in the framework.cpp organization, the main loop is contained
# in the tutorial files, not in this framework file. Additionally, a copy of
# this module file must exist in the same directory as the tutorial files
# to be imported properly.


import os

from OpenGL.GL import *


# Function that creates and compiles shaders according to the given type (a GL enum value) and
# shader program (a file containing a GLSL program).
def load_shader(shader_type, shader_file):
    # check if file exists, get full path name
    str_filename = find_file_or_throw(shader_file)
    shader_data = None
    with open(str_filename, 'r') as f:
        shader_data = f.read()

    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_data)  # note that this is a simpler function call than in C

    # This shader compilation is more explicit than the one used in
    # framework.cpp, which relies on a glutil wrapper function.
    # This is made explicit here mainly to decrease dependence on pyOpenGL
    # utilities and wrappers, which docs caution may change in future versions.
    glCompileShader(shader)

    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status == GL_FALSE:
        # Note that getting the error log is much simpler in Python than in C/C++
        # and does not require explicit handling of the string buffer
        str_info_log = glGetShaderInfoLog(shader)
        str_shader_type = ""
        if shader_type is GL_VERTEX_SHADER:
            str_shader_type = "vertex"
        elif shader_type is GL_GEOMETRY_SHADER:
            str_shader_type = "geometry"
        elif shader_type is GL_FRAGMENT_SHADER:
            str_shader_type = "fragment"

        print("Compilation failure for " + str_shader_type + " shader:\n" + str(str_info_log))

    return shader


# Function that accepts a list of shaders, compiles them, and returns a handle to the compiled program
def create_program(shader_list):
    program = glCreateProgram()

    for shader in shader_list:
        glAttachShader(program, shader)

    glLinkProgram(program)

    status = glGetProgramiv(program, GL_LINK_STATUS)
    if status == GL_FALSE:
        # Note that getting the error log is much simpler in Python than in C/C++
        # and does not require explicit handling of the string buffer
        str_info_log = glGetProgramInfoLog(program)
        print("Linker failure: \n" + str(str_info_log))

    for shader in shader_list:
        glDetachShader(program, shader)

    return program


# Helper function to locate and open the target file (passed in as a string).
# Returns the full path to the file as a string.
def find_file_or_throw(str_basename):
    # Keep constant names in C-style convention, for readability
    # when comparing to C(/C++) code.
    if os.path.isfile(str_basename):
        return str_basename

    local_file_dir = "." + os.sep
    global_file_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep

    str_filename = local_file_dir + str_basename
    if os.path.isfile(str_filename):
        return str_filename

    str_filename = global_file_dir + str_basename
    if os.path.isfile(str_filename):
        return str_filename

    str_filename = os.path.join(global_file_dir, "shaders", str_basename)
    if os.path.isfile(str_filename):
        return str_filename

    raise IOError('Could not find target file ' + str_basename)
