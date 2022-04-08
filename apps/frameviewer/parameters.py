#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 9/24/21.
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
from typing import Type

from settings.tsdf import TsdfParameters


class FrameviewerParameters(ParameterEnum):
    input = \
        Parameter(default="datasets/DeepDeform/train/seq070", arg_type=str,
                  arg_help="Path to folder with frame data.")
    output = \
        Parameter(default="output/train/seq070", arg_type=str,
                  arg_help="Path to folder with output generated from the sequence (optional).")

    start_frame_index = \
        Parameter(default=0, arg_type=int,
                  arg_help="The frame of the sequence to start viewing from.")

    masking_threshold = \
        Parameter(default=254, arg_type=int,
                  arg_help="Threshold to use for masking. Masks are typically greyscale 8-bit images (with values "
                           "between 0 and 255. A threshold of 0 would show everything, a threshold of 255 would "
                           "obscure everything.")

    frame_count = \
        Parameter(default=0, arg_type=int,
                  arg_help="The total number of frames to view.")

    tsdf: Type[TsdfParameters] = TsdfParameters
