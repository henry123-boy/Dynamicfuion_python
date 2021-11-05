#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 11/5/21.
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
import torch


# for use on computing energies
def robust_tukey_gradient_penalty(linear_loss: torch.Tensor, constant: float):
    constant_factor = torch.tensor([constant * constant / 6.0], dtype=torch.float32, device=linear_loss.device)
    quotients: torch.Tensor = linear_loss / constant
    no_cutoff = constant_factor - (1.0 - torch.pow(1.0 - (quotients * quotients), 3)) / 6.0
    return torch.where(linear_loss <= constant, no_cutoff, constant_factor)


# for use on computing energy residuals
def robust_tukey_gradient_penalty_derivative(linear_residuals: torch.Tensor, constant: float):
    no_cutoff = linear_residuals / constant
    no_cutoff = 1.0 - no_cutoff * no_cutoff
    no_cutoff = linear_residuals * no_cutoff * no_cutoff
    zero = torch.zeros([1], dtype=torch.float32, device=linear_residuals.device)
    return torch.where(linear_residuals.abs() <= constant, no_cutoff, zero)


def huber_gradient_penalty(linear_loss: torch.Tensor, constant: float):
    norms = linear_loss.norm(dim=1)
    return torch.where(norms < constant, norms * norms * 0.5, constant * (norms - constant * 0.5))


def huber_gradient_penalty_derivative(linear_residuals: torch.Tensor, constant: float):
    norms = linear_residuals.norm(dim=1)
    return torch.where(norms < constant, linear_residuals, linear_residuals * constant / norms)
