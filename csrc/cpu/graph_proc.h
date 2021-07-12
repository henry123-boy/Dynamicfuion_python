#pragma once


#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define GRAPH_K 4

namespace py = pybind11;

namespace graph_proc {


py::array_t<bool>
get_vertex_erosion_mask(const py::array_t<float>& vertex_positions, const py::array_t<int>& face_indices, int iteration_count, int min_neighbors);

/**
 * Samples canonical_node_positions that cover all vertex positions with given node coverage.
 * Nodes are sampled from vertices, resulting node vertex indices are returned.
 */
py::tuple sample_nodes(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<bool>& vertex_erosion_mask_in,
		float node_coverage, bool use_only_non_eroded_indices,
		bool random_shuffle);


void compute_edges_shortest_path(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<bool>& vertex_mask_in,
		const py::array_t<int>& face_indices_in,
		const py::array_t<int>& node_indices_in,
		int max_neighbor_count, const float node_coverage,
		py::array_t<int>& graph_edges_out,
		py::array_t<float>& graph_edges_weights_out,
		py::array_t<float>& graph_edges_distances_out,
		py::array_t<float>& node_to_vertex_distances_out,
		bool enforce_total_num_neighbors
);

void compute_edges_shortest_path(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<int>& face_indices_in,
		const py::array_t<int>& node_indices_in,
		int max_neighbor_count, float node_coverage,
		py::array_t<int>& graph_edges_out,
		py::array_t<float>& graph_edges_weights_out,
		py::array_t<float>& graph_edges_distances_out,
		py::array_t<float>& node_to_vertex_distances_out,
		bool enforce_total_num_neighbors
);

py::tuple compute_edges_shortest_path(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<bool>& vertex_mask_in,
		const py::array_t<int>& face_indices_in,
		const py::array_t<int>& node_indices_in,
		int max_neighbor_count, float node_coverage,
		bool enforce_total_num_neighbors
);

py::tuple compute_edges_shortest_path(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<int>& face_indices_in,
		const py::array_t<int>& node_indices_in,
		int max_neighbor_count, float node_coverage,
		bool enforce_total_num_neighbors
);


/**
 * Compute the graph edges between nodes, connecting nearest nodes using Euclidean
 * distances.
 */
py::array_t<int> compute_edges_euclidean(const py::array_t<float>& node_positions, int max_neighbor_count);

/**
 * Removes invalid nodes (with less than 2 neighbors).
 */
void node_and_edge_clean_up(const py::array_t<int>& graph_edges, py::array_t<bool>& valid_nodes_mask);

/**
 * Compute node clusters based on connectivity.
 * @returns Sizes (number of nodes) of each cluster.
 */
py::tuple compute_clusters(const py::array_t<int>& graph_edges_in);

/**
 * Computes node clusters based on connectivity.
 * @returns Sizes (number of nodes) of each cluster.
 */
std::vector<int> compute_clusters_legacy(
		const py::array_t<int> graph_edges,
		py::array_t<int> graph_clusters
);


/**
 * For each input pixel it computes 4 nearest anchors, following graph edges.
 * It also compute skinning weights for every pixel.
 */
void compute_pixel_anchors_shortest_path(
		const py::array_t<float>& node_to_vertex_distance,
		const py::array_t<int>& valid_nodes_mask,
		const py::array_t<float>& vertices,
		const py::array_t<int>& vertex_pixels,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights,
		int width, int height,
		float node_coverage
);

py::tuple compute_pixel_anchors_shortest_path(
		const py::array_t<float>& node_to_vertex_distance,
		const py::array_t<int>& valid_nodes_mask,
		const py::array_t<float>& vertices,
		const py::array_t<int>& vertex_pixels,
		int width, int height,
		float node_coverage
);


void compute_pixel_anchors_euclidean(
		const py::array_t<float>& graph_nodes,
		const py::array_t<float>& point_image,
		float node_coverage,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights
);

py::tuple compute_pixel_anchors_euclidean(
		const py::array_t<float>& graph_nodes,
		const py::array_t<float>& point_image,
		float node_coverage
);

py::tuple compute_vertex_anchors_euclidean(
		const py::array_t<float>& graph_nodes,
		const py::array_t<float>& vertices,
		float node_coverage
);

/**
 * Updates pixel anchor after node id change.
 */
void update_pixel_anchors(
		const std::map<int, int>& node_id_mapping,
		py::array_t<int>& pixel_anchors
);

/**
 * It samples graph regularly from the image, using pixel-wise connectivity
 * (connecting each pixel with at most 8 neighbors).
 */
void construct_regular_graph(
		const py::array_t<float>& point_image,
		int x_nodes, int y_nodes,
		float edge_threshold,
		float max_point_to_node_distance,
		float max_depth,
		py::array_t<float>& graph_nodes,
		py::array_t<int>& graph_edges,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights
);

py::tuple construct_regular_graph(
		const py::array_t<float>& point_image,
		int x_nodes, int y_nodes,
		float edge_threshold,
		float max_point_to_node_distance,
		float max_depth
);

void compute_vertex_anchors_shortest_path(
		const py::array_t<float>& vertices,
		const py::array_t<float>& nodes,
		const py::array_t<int>& edges,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights,
		int anchor_count,
		float node_coverage
);

void compute_pixel_anchors_shortest_path(
		const py::array_t<float>& point_image,
		const py::array_t<float>& nodes,
		const py::array_t<int>& edges,
		py::array_t<int>& anchors,
		py::array_t<float>& weights,
		int anchor_count,
		float node_coverage
);

py::tuple compute_vertex_anchors_shortest_path(
		const py::array_t<float>& vertices,
		const py::array_t<float>& nodes,
		const py::array_t<int>& edges,
		int anchor_count,
		float node_coverage
);

py::tuple compute_pixel_anchors_shortest_path(
		const py::array_t<float>& point_image,
		const py::array_t<float>& nodes,
		const py::array_t<int>& edges,
		int anchor_count,
		float node_coverage
);


} // namespace graph_proc