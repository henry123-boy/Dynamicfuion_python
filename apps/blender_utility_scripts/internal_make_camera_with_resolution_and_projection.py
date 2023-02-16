#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 2/9/23.
#  Copyright (c) 2023 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
# script intended for usage inside blender

import bpy
import numpy as np
def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")


def main():
    # input section

    render_resolution = (100,100) # (height, width)

    projection_matrix = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0,   0.0,   1.0],
    ])

    # computation

    scene = bpy.context.scene

    scene.render.resolution_percentage = 100
    scene.render.resolution_x = render_resolution[1]
    scene.render.resolution_y = render_resolution[0]

    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)

    camera_data.sensor_width = 50 # mm
    camera_data.sensor_height = 50 # mm
    camera_data.lens_unit = 'FOV'



    f_x = projection_matrix[0,0]
    f = f_x * camera_data.sensor_width / render_resolution[1]
    camera_data.lens = f


    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object





if __name__ == "__main__":
    main()