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
import torch
from settings.rendering_alignment import PenaltyFunction, RenderingAlignmentParameters


def square_data_penalty(linear_loss: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.square(linear_loss)


def linear_data_residuals(linear_residuals: torch.Tensor):
    return linear_residuals


def square_regularization_penalty(linear_loss: torch.Tensor):
    return linear_loss.norm(dim=1) * 0.5


def linear_regularization_residuals(linear_residuals: torch.Tensor):
    return linear_residuals

# TODO: extend torch.autograd.Function and make these into forward()/backward() functions w/ torch.no_grad()
# for use on computing energy/loss
def robust_tukey_penalty(linear_loss: torch.Tensor):
    constant = RenderingAlignmentParameters.tukey_penalty_constant.value
    constant_factor = torch.tensor([constant * constant / 6.0], dtype=torch.float32, device=linear_loss.device)
    quotients: torch.Tensor = linear_loss / constant
    no_cutoff = constant_factor - (1.0 - torch.pow(1.0 - (quotients * quotients), 3)) / 6.0
    return torch.where(linear_loss <= constant, no_cutoff, constant_factor)


# for use on computing gradients
def robust_tukey_penalty_grad(linear_residuals: torch.Tensor):
    constant = RenderingAlignmentParameters.tukey_penalty_constant.value
    no_cutoff = linear_residuals * torch.square(1.0 - torch.square(linear_residuals / constant))
    zero = torch.zeros([1], dtype=torch.float32, device=linear_residuals.device)
    return torch.where(linear_residuals.abs() <= constant, no_cutoff, zero)


def huber_penalty(linear_loss: torch.Tensor):
    constant = RenderingAlignmentParameters.huber_penalty_constant.value
    norms = linear_loss.norm(dim=1)
    return torch.where(norms < constant, norms * norms * 0.5, constant * (norms - constant * 0.5))


def huber_penalty_grad(linear_residuals: torch.Tensor):
    constant = RenderingAlignmentParameters.huber_penalty_constant.value
    norms = linear_residuals.norm(dim=1)
    return torch.where(norms < constant, linear_residuals, linear_residuals * constant / norms)


DATA_PENALTY_FUNCTION_MAP = {
    PenaltyFunction.SQUARE: square_data_penalty,
    PenaltyFunction.TUKEY: robust_tukey_penalty,
    PenaltyFunction.HUBER: huber_penalty
}


def apply_data_residuals_penalty(linear_loss: torch.Tensor):
    return DATA_PENALTY_FUNCTION_MAP[RenderingAlignmentParameters.data_term_penalty.value](linear_loss)


REGULARIZATION_PENALTY_FUNCTION_MAP = {
    PenaltyFunction.SQUARE: square_regularization_penalty,
    PenaltyFunction.TUKEY: square_data_penalty,
    PenaltyFunction.HUBER: huber_penalty
}


def apply_regularization_penalty(linear_loss: torch.Tensor):
    return REGULARIZATION_PENALTY_FUNCTION_MAP[RenderingAlignmentParameters.regularization_term_penalty_function.value](linear_loss)
