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


# TODO: combine with related TrackingParameters in settings/fusion.py -- reorganize
class GraphParameters(ParameterEnum):
    node_coverage = \
        Parameter(default=0.05, arg_type=float,
                  arg_help="This is the maximum distance between any point in the source point cloud and at least one "
                           "of the resulting graph nodes. Allows to control graph sparsity and influences a number of "
                           "other operations that have to do with node influence over surrounding points.")
    # TODO: refactor all parameters with 'graph_' prefix -- remove the prefix

    graph_debug = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Show & print debug output during graph generation.")
    # TODO: refactor to max_mesh_triangle_edge_length
    graph_max_triangle_distance = \
        Parameter(default=0.05, arg_type=float,
                  arg_help="This is actually the maximum edge length allowed for any triangles generated from an"
                           "RGB-D image pair / resulting point cloud during graph construction.")
    graph_erosion_num_iterations = \
        Parameter(default=4, arg_type=int,
                  arg_help="Number of erosion iterations applied to the graph during generation.")
    graph_erosion_min_neighbors = \
        Parameter(default=4, arg_type=int,
                  arg_help="While the graph is being eroded (during generation), the nodes not having the required"
                           "minimum neighbor count will be removed.")
    graph_use_only_valid_vertices = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Whether to use eroded nodes during sampling or not.")
    # TODO: refactor to max_neighbor_count
    graph_neighbor_count = \
        Parameter(default=8, arg_type=int,
                  arg_help="Maximum possible number of neighbors each node in the graph has after generation. "
                           "Corresponds to the width of the edge table/2d array.")
    graph_enforce_neighbor_count = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Whether to enforce the neighbor count during graph generation. If set to true,"
                           "even neighbors beyond the maximal edge influence (2*node_coverage) will be filled "
                           "in the edge table, so that each node has exactly neighbor_count neighbors. ")
    graph_sample_random_shuffle = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Whether to use random node shuffling during node sampling for graph generation.")
    graph_remove_nodes_with_too_few_neighbors = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Whether to remove nodes with \"too few\" neighbors before rolling out the graph. "
                           "Currently, the \"too few\" condition is hard-coded as \"one or fewer\".")
