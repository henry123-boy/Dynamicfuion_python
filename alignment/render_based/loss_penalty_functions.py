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
from settings.rendering_alignment import DataTermPenaltyFunction, RegularizationTermPenaltyFunction, RenderingAlignmentParameters


def square_data_loss(linear_loss: torch.Tensor):
    return 0.5 * linear_loss * linear_loss


def linear_data_residuals(linear_residuals: torch.Tensor):
    return linear_residuals


def square_regularization_penalty(linear_loss: torch.Tensor):
    return linear_loss.norm(dim=1) * 0.5


def linear_regularization_residuals(linear_residuals: torch.Tensor):
    return linear_residuals


# for use on computing energies
def robust_tukey_data_penalty(linear_loss: torch.Tensor):
    constant = RenderingAlignmentParameters.tukey_penalty_constant.value
    constant_factor = torch.tensor([constant * constant / 6.0], dtype=torch.float32, device=linear_loss.device)
    quotients: torch.Tensor = linear_loss / constant
    no_cutoff = constant_factor - (1.0 - torch.pow(1.0 - (quotients * quotients), 3)) / 6.0
    return torch.where(linear_loss <= constant, no_cutoff, constant_factor)


# for use on computing energy residuals
def robust_tukey_data_penalty_derivative(linear_residuals: torch.Tensor):
    constant = RenderingAlignmentParameters.tukey_penalty_constant.value
    no_cutoff = linear_residuals / constant
    no_cutoff = 1.0 - no_cutoff * no_cutoff
    no_cutoff = linear_residuals * no_cutoff * no_cutoff
    zero = torch.zeros([1], dtype=torch.float32, device=linear_residuals.device)
    return torch.where(linear_residuals.abs() <= constant, no_cutoff, zero)


def huber_regularization_penalty(linear_loss: torch.Tensor):
    constant = RenderingAlignmentParameters.huber_penalty_constant.value
    norms = linear_loss.norm(dim=1)
    return torch.where(norms < constant, norms * norms * 0.5, constant * (norms - constant * 0.5))


def huber_regularization_penalty_derivative(linear_residuals: torch.Tensor):
    constant = RenderingAlignmentParameters.huber_penalty_constant.value
    norms = linear_residuals.norm(dim=1)
    return torch.where(norms < constant, linear_residuals, linear_residuals * constant / norms)


DATA_ENERGY_PENALTY_FUNCTION_MAP = {
    DataTermPenaltyFunction.NONE: square_data_loss,
    DataTermPenaltyFunction.ROBUST_TUKEY: robust_tukey_data_penalty
}


def apply_data_energy_penalty(linear_loss: torch.Tensor):
    return DATA_ENERGY_PENALTY_FUNCTION_MAP[RenderingAlignmentParameters.data_term_penalty.value](linear_loss)


DATA_RESIDUAL_PENALTY_FUNCTION_MAP = {
    DataTermPenaltyFunction.NONE: linear_data_residuals,
    DataTermPenaltyFunction.ROBUST_TUKEY: robust_tukey_data_penalty_derivative
}


def apply_data_residual_penalty(linear_residual: torch.Tensor):
    return DATA_RESIDUAL_PENALTY_FUNCTION_MAP[RenderingAlignmentParameters.data_term_penalty.value](linear_residual)


REGULARIZATION_ENERGY_PENALTY_FUNCTION_MAP = {
    RegularizationTermPenaltyFunction.ROBUST_TUKEY: square_regularization_penalty,
    RegularizationTermPenaltyFunction.ROBUST_TUKEY_GRADIENT: huber_regularization_penalty
}


def apply_regularization_energy_penalty(linear_loss: torch.Tensor):
    return REGULARIZATION_ENERGY_PENALTY_FUNCTION_MAP[RenderingAlignmentParameters.regularization_term_penalty.value](linear_loss)


REGULARIZATION_RESIDUAL_PENALTY_FUNCTION_MAP = {
    RegularizationTermPenaltyFunction.ROBUST_TUKEY: linear_regularization_residuals,
    RegularizationTermPenaltyFunction.ROBUST_TUKEY_GRADIENT: huber_regularization_penalty_derivative
}


def apply_regularization_residual_penalty(linear_residual: torch.Tensor):
    return REGULARIZATION_RESIDUAL_PENALTY_FUNCTION_MAP[RenderingAlignmentParameters.regularization_term_penalty.value](linear_residual)
