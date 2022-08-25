#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 11/5/21.
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
from enum import Enum


class PenaltyFunction(Enum):
    SQUARE = 1,
    TUKEY = 2,
    HUBER = 3


class RenderingAlignmentParameters(ParameterEnum):
    data_term_penalty_function = \
        Parameter(default=PenaltyFunction.TUKEY, arg_type=PenaltyFunction,
                  arg_help="What penalty function to use for the energy of the data term (point-to-plane) energy.")
    data_term_penalty_constant = \
        Parameter(default=0.01, arg_type=float,
                  arg_help="Controls how quickly the data term penalty function tapers off away from the origin if "
                           "TUKEY or HUBER are used for the penalty function.")
    regularization_term_penalty_function = \
        Parameter(default=PenaltyFunction.HUBER, arg_type=PenaltyFunction,
                  arg_help="What penalty function to use for the energy of the data term (point-to-plane) energy.")
    regularization_term_penalty_constant = \
        Parameter(default=0.0001, arg_type=float,
                  arg_help="Controls how quickly the data term penalty function tapers off away from the origin if "
                           "TUKEY or HUBER are used for the penalty function.")
