#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 2/13/23.
#  Copyright (c) 2023 Gregory Kramida
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
import numpy as np


def edge_endpoint_array_to_ordered_origin_endpoint_matrix(edge_endpoint_array: np.ndarray, max_degree=4) -> np.ndarray:
    if len(edge_endpoint_array.shape) != 2 and edge_endpoint_array.shape[1] != 2:
        raise ValueError(f"edge_endpoint_array must be an 2d np.ndarray with two columns. "
                         f"Got shape: {edge_endpoint_array.shape}")

    max_vertex_index = np.max(edge_endpoint_array)
    origin_endpoints = -np.ones(shape=(max_vertex_index+1, max_degree), dtype=np.int32)
    origin_list_lengths = np.zeros((max_vertex_index+1), dtype=np.int32)

    for edge in edge_endpoint_array:
        origin_endpoint_list = origin_endpoints[edge[0]]
        origin_endpoint_list_length = origin_list_lengths[edge[0]]
        origin_endpoint_list[origin_endpoint_list_length] = edge[1]
        origin_list_lengths[edge[0]] += 1

    return origin_endpoints
