#pragma once

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define GRAPH_K 4


namespace graph_proc {

    /**
	 * Erode mesh
	 */
	py::array_t<bool> get_vertex_erosion_mask(const py::array_t<float>& vertex_positions, const py::array_t<int>& face_indices, int iteration_count, int min_neighbors);
	
    /**
	 * Samples nodes that cover all vertex positions with given node coverage.
	 * Nodes are sampled from vertices, resulting node vertex indices are returned.
	 */
	int sample_nodes(
		    const py::array_t<float>& vertex_positions_in, const py::array_t<bool>& vertex_erosion_mask_in,
		    py::array_t<float>& node_positions_out, py::array_t<int>& node_indices_out,
		    float node_coverage, const bool use_only_non_eroded_indices, const bool random_shuffle);


	/**
	 * Computes the graph edges between nodes, connecting nearest nodes using geodesic
	 * distances.
	 */
	py::array_t<int> compute_edges_geodesic(
			const py::array_t<float>& vertex_positions,
			const py::array_t<bool>& valid_vertices,
			const py::array_t<int>& face_indices,
			const py::array_t<int>& node_indices,
			int max_neighbor_count,
			float node_coverage,
	        py::array_t<int>& graph_edges,
	        py::array_t<float>& graph_edge_weights,
	        py::array_t<float>& graph_edge_distances,
	        py::array_t<float>& node_to_vertex_distances,
	        const bool allow_only_valid_vertices,
	        const bool enforce_total_num_neighbors
	);

	/**
	 * Computes the graph edges between nodes, connecting nearest nodes using Euclidean
	 * distances.
	 */
	py::array_t<int> compute_edges_euclidean(const py::array_t<float>& node_positions, int max_neighbor_count);
	
    /**
     * Removes invalid nodes (with less than 2 neighbors).
     */
    void node_and_edge_clean_up(const py::array_t<int>& graph_edges, py::array_t<bool>& valid_nodes_mask);

    /**
     * Computes node clusters based on connectivity.
     * @returns Sizes (number of nodes) of each cluster.
     */
    std::vector<int> compute_clusters(
        const py::array_t<int> graph_edges,
        py::array_t<int> graph_clusters
    );


    /**
	 * For each input pixel it computes 4 nearest anchors, following graph edges. 
	 * It also compute skinning weights for every pixel. 
	 */ 
	void compute_pixel_anchors_geodesic(
        const py::array_t<float> &node_to_vertex_distance,
        const py::array_t<int> &valid_nodes_mask,
        const py::array_t<float> &vertices,
        const py::array_t<int> &vertex_pixels,
        py::array_t<int>& pixel_anchors,
        py::array_t<float>& pixel_weights,
        const int width, const int height,
        const float node_coverage
    );

	/**
	 * For each input pixel it computes 4 nearest anchors, using Euclidean distances. 
	 * It also compute skinning weights for every pixel. 
	 */ 
	void compute_pixel_anchors_euclidean(
        const py::array_t<float>& graph_nodes,
        const py::array_t<float>& point_image,
        float node_coverage,
        py::array_t<int>& pixel_anchors,
        py::array_t<float>& pixel_weights
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

} // namespace graph_proc