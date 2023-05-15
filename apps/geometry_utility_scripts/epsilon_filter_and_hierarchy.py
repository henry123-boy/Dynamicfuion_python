#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 5/15/23.
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
import sys
from typing import Tuple, List

import numpy as np


def epsilon_sample(array: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    array_filtered = np.sort(array)
    sample = np.arange(len(array_filtered))
    distances = array_filtered[1:] - array_filtered[:-1]
    min_distance = distances.min()
    while min_distance < epsilon:
        indexes_remaining = []
        indexes_filtered = set()
        for i_element, element in enumerate(array_filtered):
            if i_element not in indexes_filtered:
                indexes_remaining.append(i_element)
                if i_element < len(array_filtered) - 1:
                    next_element = array_filtered[i_element + 1]
                    if next_element - element < epsilon:
                        indexes_filtered.add(i_element + 1)
        array_filtered = array_filtered[indexes_remaining]
        sample = sample[indexes_remaining]
        if len(array_filtered) > 1:
            distances = array_filtered[1:] - array_filtered[:-1]
            min_distance = distances.min()
        else:
            break
    return array_filtered, sample


class Layer():
    def __init__(self, node_positions, node_indices, edges_to_coarser=np.array([])):
        self.node_positions = node_positions
        self.node_indices = node_indices
        self.edges_to_coarser = edges_to_coarser

    def __repr__(self):
        return f"===Layer===\n  Node positions: {self.node_positions}\n  Node indices: {self.node_indices}\n  Edges: {self.edges_to_coarser}\n"


def build_node_hierarchy(node_positions: np.ndarray, L: int = 4, B=4, epsilon=0.015) -> List[Layer]:
    layers = []
    layer0 = Layer(node_positions=node_positions, node_indices=np.arange(0, len(node_positions)))
    layers.append(layer0)

    for l in range(1, L):
        previous_layer = layers[l - 1]
        layer_epsilon = epsilon * (B ** l)
        layer_node_positions, previous_layer_node_sample = epsilon_sample(previous_layer.node_positions, layer_epsilon)
        layer = Layer(node_positions=layer_node_positions,
                      node_indices=previous_layer.node_indices[previous_layer_node_sample])
        previous_layer_mask = np.ones(previous_layer.node_positions.shape, bool)
        previous_layer_mask[previous_layer_node_sample] = False
        previous_layer.node_positions = previous_layer.node_positions[previous_layer_mask]
        previous_layer.node_indices = previous_layer.node_indices[previous_layer_mask]

        layers.append(layer)
    return layers


def reindex_node_hierarchy(hierarchy: List[Layer], node_count) -> np.ndarray:
    new_to_old_index_map = np.zeros((node_count), np.int64)
    next_new_node_index = 0
    hierarchy.reverse()
    for layer in hierarchy:
        for i_node_index in range(0, len(layer.node_indices)):
            old_index = layer.node_indices[i_node_index]
            layer.node_indices[i_node_index] = next_new_node_index
            new_to_old_index_map[next_new_node_index] = old_index
            next_new_node_index += 1

    return new_to_old_index_map


def k_nearest_neighbors(query_points, data, k):
    k_nearest_for_each_query = []
    for query_point in query_points:
        result = []
        for i_point in range(0, len(data)):
            distance = abs(query_point - data[i_point])
            result.append([i_point, distance])
        sorted_result = sorted(result)
        indices = []
        if k < len(data):
            for i_result in range(0, k):
                indices.append(sorted_result[i_result][0])
        else:
            indices = [result[0] for result in sorted_result]
        k_nearest_for_each_query.append(indices)
    return k_nearest_for_each_query


def define_hierarchy_edges(hierarchy: List[Layer], k) -> None:
    for l in range(1, len(hierarchy)):
        coarser_layer = hierarchy[l-1]
        finer_layer = hierarchy[l]
        edges = k_nearest_neighbors(finer_layer.node_positions, coarser_layer.node_positions, k)

        print(edges)


def main():
    nodes = np.array(
        [0.00950328, 0.02285199, 0.13555111, 0.15677549, 0.16505296, 0.18583595, 0.22895041, 0.33687065, 0.33997254,
         0.39501414, 0.41348084, 0.41525062, 0.42536172, 0.43980097, 0.44716107, 0.50197847, 0.51962451, 0.53292511,
         0.55830996, 0.58350063, 0.65435996, 0.66131234, 0.68292744, 0.70779239, 0.73901546, 0.78387486, 0.84494837,
         0.90642201, 0.93449647, 0.93959651, 0.97725148])

    layers = build_node_hierarchy(nodes, L=4, B=4, epsilon=0.01)
    new_to_old_index_map = reindex_node_hierarchy(layers, len(nodes))
    define_hierarchy_edges(layers, 2)
    # print(layers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
