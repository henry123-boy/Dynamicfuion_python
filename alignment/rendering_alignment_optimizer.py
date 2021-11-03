#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 10/29/21.
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
from warp_field.graph import DeformationGraphOpen3D
from nnrt.geometry import WarpableTSDFVoxelGrid
from linear_solver_lu import LinearSolverLU
import torch
import nnrt


class RenderingAlignmentOptimizer:

    def __init__(self):
        nnrt.geometry
        pass

    def optimize_graph(self, graph: DeformationGraphOpen3D, tsdf: WarpableTSDFVoxelGrid):
        pass
