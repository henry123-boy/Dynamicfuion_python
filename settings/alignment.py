#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 10/27/21.
#  Copyright (c) 2021 Gregory Kramida
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
from ext_argparse import ParameterEnum, Parameter


class AlignmentParameters(ParameterEnum):
    image_width = \
        Parameter(default=640, arg_type=int,
                  arg_help="Input image/point cloud height for the non-rigid alignment portion of the algorithm. "
                           "The actual image / point cloud will be cropped down to this height and intrinsic matrix "
                           "adjusted accordingly.")
    image_height = \
        Parameter(default=448, arg_type=int,
                  arg_help="Input image/point cloud width for the non-rigid alignment portion of the algorithm. "
                           "The actual image / point cloud will be cropped down to this width and intrinsic matrix "
                           "adjusted accordingly.")
    max_boundary_distance = \
        Parameter(default=0.10, arg_type=float,
                  arg_help="Used in marking up boundaries within an incoming RGB-D image pair. When neighboring pixel "
                           "points within the point cloud based on the depth image exceed this distance from each other, "
                           "the boundaries are drawn along the break.")
