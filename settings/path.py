#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph).
#  Copyright (c) 2021 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

import os
from collections import namedtuple
from pathlib import Path
from ext_argparse import ParameterEnum, Parameter

LocalPathCollection = namedtuple("LocalPathCollection", "deep_deform_root output")

# to add your own root DeepDeform data directory, run the sha256 cypher on your MAC address and add the hash &
# local directory as a key/value pair to the dict below
REPOSITORY_ROOT = Path(__file__).parent.parent.resolve().absolute()

DEFAULT_OUTPUT_DIRECTORY = os.path.join(REPOSITORY_ROOT, "output")
DEFAULT_NN_DATA_DIRECTORY = os.path.join(REPOSITORY_ROOT, "nn_data")


class PathParameters(ParameterEnum):
    dataset_base_directory = \
        Parameter(default="datasets/DeepDeform", arg_type=str,
                  arg_help="Path to the base of the DeepDeform dataset root.")
    output_directory = \
        Parameter(default=DEFAULT_OUTPUT_DIRECTORY, arg_type=str,
                  arg_help="Path to the directory where reconstruction output & telemetry will be placed.")
    nn_data_directory = \
        Parameter(default=DEFAULT_NN_DATA_DIRECTORY, arg_type=str,
                  arg_help="Path to the directory where trained DeformNet models & other neural network data are stored.")



