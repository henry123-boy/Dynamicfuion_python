#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 10/11/21.
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


class VisualizerParameters(ParameterEnum):
    base_output_path = Parameter(arg_type=str,
                                 default="/mnt/Data/Reconstruction/output/NeuralTracking_experiment_output",
                                 arg_help="Path to the output folder for all fusion pipeline run linear_loss.")
    experiment_folder = \
        Parameter(arg_type=str, default="!LATEST!",
                  arg_help=
                  "Folder (relative to base_output_path) with output from a single fusion pipeline app run. If "
                  "\"!LATEST!\" is specified instead of a folder name, the app will retrieve the folder that's using "
                  "the latest date & time in it's name (or, if no folders have such a name in the proper sortable "
                  "timestamp format, the folder with the latest modification date.)")
    start_frame = \
        Parameter(arg_type=int, default=-1, shorthand="start",
                  arg_help="Frame in the sequence to start from. Passing '-1' will cause the program "
                           "to automatically infer and utilize the entire available frame range by analyzing "
                           "experiment_folder contents.")
    add_synchronized_frameviewer = \
        Parameter(arg_type='bool_flag', default=False, shorthand="fv",
                  arg_help="Start a frameviewer to show sequence input synchronized to the output frames.")
