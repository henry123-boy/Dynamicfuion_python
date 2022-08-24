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

from ext_argparse import ParameterEnum, Parameter


# Info for a saved alignment
# - In train.py, this info is only used if use_pretrained_model=True
# - In generate.py, evaluate.py or example_viz.py, it is used regardless of the value of use_pretrained_model
class ModelParameters(ParameterEnum):
    # TODO: switch to an Enum parameter
    model_module_to_load = \
        Parameter(default="full_model", arg_type=str,
                  arg_help="Must be set to one of ['only_flow_net', 'full_model']. Dictates whether the model will be"
                           "loaded in full or only the flow_net part will be loaded.")
    model_name = \
        Parameter(default="model_A", arg_type=str,
                  arg_help="Name of the pre-trained model to use.")
    model_iteration = \
        Parameter(default=0, arg_type=int,
                  arg_help="Iteration number of the model to load.")


def get_saved_model():
    from settings.path import PathParameters
    import os
    return os.path.join(PathParameters.nn_data_directory.value, "models",
                        ModelParameters.model_name.value,
                        f"{ModelParameters.model_name.value}_{ModelParameters.model_iteration.value}.pt")
