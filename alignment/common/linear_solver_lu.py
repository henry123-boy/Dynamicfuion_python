#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 10/29/21.
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
from typing import Tuple, Union, List, Any


class LinearSolverLU(torch.autograd.Function):
    """
    LU linear solver.
    """

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, A, b):
        A_LU, pivots = torch.lu(A)
        x = torch.lu_solve(b, A_LU, pivots)

        ctx.save_for_backward(A_LU, pivots, x)

        return x

    @staticmethod
    def backward(ctx, grad_x):
        A_LU, pivots, x = ctx.saved_tensors

        # Math:
        # A * grad_b = grad_x
        # grad_A = -grad_b * x^T

        grad_b = torch.lu_solve(grad_x, A_LU, pivots)
        grad_A = -torch.matmul(grad_b, x.view(1, -1))

        return grad_A, grad_b
