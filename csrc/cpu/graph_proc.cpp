#include "cpu/graph_proc.h"

#include <set>
#include <vector>
#include <numeric> //std::iota
#include <algorithm>
#include <random>
#include <map>
#include <queue>
#include <functional>

#include <Eigen/Dense>
#include <cfloat>

using std::vector;

namespace graph_proc {

/**
 * \brief Compile a vertex mask that can be used to erode the provided mesh (iteratively mask out vertices at surface discontinuities, leave only non-eroded vertices)
 * \param vertex_positions array with vertex positions of dimentions (N, 3)
 * \param face_indices
 * \param iteration_count
 * \param min_neighbors minimum number of neighbors considered for erosion
 * \return (N, 1) array containing the erosion mask, such that, if applied to the original vertex postions array, would leave only non-eroded vertices
 */
py::array_t<bool> get_vertex_erosion_mask(const py::array_t<float>& vertex_positions, const py::array_t<int>& face_indices,
                                          int iteration_count, int min_neighbors) {
	int vertex_count = vertex_positions.shape(0);
	int face_count = face_indices.shape(0);

	// Init output
	py::array_t<bool> non_eroded_vertices = py::array_t<bool>({vertex_count, 1});
	std::vector<bool> non_eroded_vertices_vec(vertex_count, false);

	// Init list of eroded face indices with original list
	std::vector<Eigen::Vector3i> non_erorded_face_indices_vec;
	non_erorded_face_indices_vec.reserve(face_count);
	for (int face_idx = 0; face_idx < face_count; ++face_idx) {
		Eigen::Vector3i face(*face_indices.data(face_idx, 0), *face_indices.data(face_idx, 1), *face_indices.data(face_idx, 2));
		non_erorded_face_indices_vec.push_back(face);
	}

	// Erode mesh for a total of iteration_count
	for (int i = 0; i < iteration_count; i++) {
		face_count = non_erorded_face_indices_vec.size();

		// We compute the number of neighboring vertices for each vertex.
		vector<int> vertex_neighbor_counts(vertex_count, 0);
		for (int face_index = 0; face_index < face_count; face_index++) {
			const auto& face = non_erorded_face_indices_vec[face_index];
			vertex_neighbor_counts[face[0]] += 1;
			vertex_neighbor_counts[face[1]] += 1;
			vertex_neighbor_counts[face[2]] += 1;
		}

		std::vector<Eigen::Vector3i> tmp;
		tmp.reserve(face_count);

		for (int face_index = 0; face_index < face_count; face_index++) {
			const auto& face = non_erorded_face_indices_vec[face_index];
			if (vertex_neighbor_counts[face[0]] >= min_neighbors &&
			    vertex_neighbor_counts[face[1]] >= min_neighbors &&
			    vertex_neighbor_counts[face[2]] >= min_neighbors) {
				tmp.push_back(face);
			}
		}

		// We kill the faces with border vertices.
		non_erorded_face_indices_vec.clear();
		non_erorded_face_indices_vec = std::move(tmp);
	}

	// Mark non isolated vertices as not eroded.
	face_count = non_erorded_face_indices_vec.size();

	for (int i = 0; i < face_count; i++) {
		const auto& face = non_erorded_face_indices_vec[i];
		non_eroded_vertices_vec[face[0]] = true;
		non_eroded_vertices_vec[face[1]] = true;
		non_eroded_vertices_vec[face[2]] = true;
	}

	// Store into python array
	for (int i = 0; i < vertex_count; i++) {
		*non_eroded_vertices.mutable_data(i, 0) = non_eroded_vertices_vec[i];
	}

	return non_eroded_vertices;
}

py::tuple sample_nodes(
		const py::array_t<float>& vertex_positions_in, const py::array_t<bool>& vertex_erosion_mask_in,
		float node_coverage, const bool use_only_non_eroded_indices = true,
		const bool random_shuffle = true
) {
	assert(vertex_positions_in.ndim() == 2);
	assert(vertex_positions_in.shape(1) == 3);

	float node_coverage_squared = node_coverage * node_coverage;
	int vertex_count = vertex_positions_in.shape(0);


	// create list of shuffled indices
	std::vector<int> shuffled_vertices(vertex_count);
	std::iota(std::begin(shuffled_vertices), std::end(shuffled_vertices), 0);

	if (random_shuffle) {
		std::default_random_engine re{std::random_device{}()};
		std::shuffle(std::begin(shuffled_vertices), std::end(shuffled_vertices), re);
	}

	struct NodeInformation {
		Eigen::Vector3f vertex_position;
		int vertex_index;
	};

	std::vector<NodeInformation> node_information_vector;

	for (int vertex_index : shuffled_vertices) {
		Eigen::Vector3f point(*vertex_positions_in.data(vertex_index, 0),
		                      *vertex_positions_in.data(vertex_index, 1),
		                      *vertex_positions_in.data(vertex_index, 2));

		if (use_only_non_eroded_indices && !(*vertex_erosion_mask_in.data(vertex_index))) {
			continue;
		}

		bool is_node = true;
		for (const auto& node_information : node_information_vector) {
			if ((point - node_information.vertex_position).squaredNorm() <= node_coverage_squared) {
				is_node = false;
				break;
			}
		}

		if (is_node) {
			node_information_vector.push_back({point, vertex_index});
		}
	}

	py::array_t<float> node_positions_out({static_cast<ssize_t>(node_information_vector.size()), static_cast<ssize_t>(3)});
	py::array_t<int> node_indices_out({static_cast<ssize_t>(node_information_vector.size()), static_cast<ssize_t>(1)});

	int node_index = 0;
	for (const auto& node_information : node_information_vector) {
		*node_positions_out.mutable_data(node_index, 0) = node_information.vertex_position.x();
		*node_positions_out.mutable_data(node_index, 1) = node_information.vertex_position.y();
		*node_positions_out.mutable_data(node_index, 2) = node_information.vertex_position.z();
		*node_indices_out.mutable_data(node_index, 0) = node_information.vertex_index;
		node_index++;
	}

	return py::make_tuple(node_positions_out, node_indices_out);
}

/**
 * Custom comparison operator for geodesic priority queue.
 */
struct CustomCompare {
	bool operator()(const std::pair<int, float>& left, const std::pair<int, float>& right) {
		return left.second > right.second;
	}
};


inline float compute_anchor_weight(const Eigen::Vector3f& point_position, const Eigen::Vector3f& node_position, float node_coverage) {
	return std::exp(-(node_position - point_position).squaredNorm() / (2.f * node_coverage * node_coverage));
}

inline float compute_anchor_weight(float dist, float node_coverage) {
	return std::exp(-(dist * dist) / (2.f * node_coverage * node_coverage));
}

inline float compute_anchor_weight_square_distance(float square_distance, float node_coverage) {
	return std::exp(-(square_distance) / (2.f * node_coverage * node_coverage));
}


template<typename TVertexCheckLambda>
inline void compute_edges_geodesic_generic(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<int>& face_indices_in,
		const py::array_t<int>& node_indices_in,
		const int max_neighbor_count,
		const float node_coverage,
		py::array_t<int>& graph_edges_out,
		py::array_t<float>& graph_edges_weights_out,
		py::array_t<float>& graph_edges_distances_out,
		py::array_t<float>& node_to_vertex_distances_out,
		const bool enforce_total_num_neighbors,
		TVertexCheckLambda&& vertex_is_valid
) {

	assert(graph_edges_out.ndim() == 2);
	assert(graph_edges_weights_out.ndim() == 2);
	assert(graph_edges_distances_out.ndim() == 2);
	assert(node_to_vertex_distances_out.ndim() == 2);

	int vertex_count = vertex_positions_in.shape(0);
	int face_count = face_indices_in.shape(0);
	int node_count = node_indices_in.shape(0);

	assert(graph_edges_out.shape(0) == node_count);
	assert(graph_edges_out.shape(1) == max_neighbor_count);
	assert(graph_edges_weights_out.shape(0) == node_count);
	assert(graph_edges_weights_out.shape(1) == max_neighbor_count);
	assert(graph_edges_distances_out.shape(0) == node_count);
	assert(graph_edges_distances_out.shape(1) == max_neighbor_count);
	assert(node_to_vertex_distances_out.shape(0) == node_count);
	assert(graph_edges_distances_out.shape(1) == vertex_count);

	float max_influence = 2.f * node_coverage;

	// Preprocess vertex neighbors.
	vector<std::set<int>> vertex_neighbors(vertex_count);
	for (int face_index = 0; face_index < face_count; face_index++) {
		for (int face_vertex_index = 0; face_vertex_index < 3; face_vertex_index++) {
			int vertex_index = *face_indices_in.data(face_index, face_vertex_index);

			for (int other_face_vertex_index = 0; other_face_vertex_index < 3; other_face_vertex_index++) {
				int other_vertex_index = *face_indices_in.data(face_index, other_face_vertex_index);

				if (vertex_index == other_vertex_index) continue;
				vertex_neighbors[vertex_index].insert(other_vertex_index);
			}
		}
	}

	// Compute inverse vertex -> node relationship.
	vector<int> vertex_to_node_map(vertex_count, -1);

	for (int node_index = 0; node_index < node_count; node_index++) {
		int vertex_index = *node_indices_in.data(node_index);
		if (vertex_index >= 0) {
			vertex_to_node_map[vertex_index] = node_index;
		}
	}

	//TODO: parallelize or remove pragma
	// #pragma omp parallel for
	for (int node_index = 0; node_index < node_count; node_index++) {
		// vertex queue
		std::priority_queue<
				std::pair<int, float>,
				vector<std::pair<int, float>>,
				CustomCompare
		> next_vertices_with_distances;

		std::set<int> visited_vertices;

		// Add node vertex as the first vertex to be visited.
		int node_vertex_index = *node_indices_in.data(node_index);
		if (node_vertex_index < 0) continue;
		next_vertices_with_distances.push(std::make_pair(node_vertex_index, 0.f));

		// Traverse all neighbors in the monotonically increasing order.
		vector<int> neighbor_node_indices;
		vector<float> neighbor_node_weights;
		vector<float> neighbor_node_distances;
		while (!next_vertices_with_distances.empty()) {
			auto next_vertex = next_vertices_with_distances.top();
			next_vertices_with_distances.pop();

			int next_vertex_index = next_vertex.first;
			float next_vertex_distance = next_vertex.second;

			// We skip the vertex, if it was already visited before.
			if (visited_vertices.find(next_vertex_index) != visited_vertices.end()) continue;

			assert(std::forward<TVertexCheckLambda>(vertex_is_valid)(next_vertex_index));

			// We check if the vertex is a node.
			int next_node_id = vertex_to_node_map[next_vertex_index];
			if (next_node_id >= 0 && next_node_id != node_index) {
				neighbor_node_indices.push_back(next_node_id);
				neighbor_node_weights.push_back(compute_anchor_weight(next_vertex_distance, node_coverage));
				neighbor_node_distances.push_back(next_vertex_distance);
				if (neighbor_node_indices.size() >= max_neighbor_count) break;
			}

			// Note down the node-vertex distance.
			*node_to_vertex_distances_out.mutable_data(node_index, next_vertex_index) = next_vertex_distance;

			// We visit the vertex, and check all his neighbors.
			// We add only valid vertices under a certain distance
			visited_vertices.insert(next_vertex_index);
			Eigen::Vector3f next_vertex_position(*vertex_positions_in.data(next_vertex_index, 0),
			                                     *vertex_positions_in.data(next_vertex_index, 1),
			                                     *vertex_positions_in.data(next_vertex_index, 2));

			const auto& next_neighbors = vertex_neighbors[next_vertex_index];
			for (int neighbor_index : next_neighbors) {

				if (!std::forward<TVertexCheckLambda>(vertex_is_valid)(neighbor_index)) {
					continue;
				}

				Eigen::Vector3f neighbor_vertex_position(*vertex_positions_in.data(neighbor_index, 0),
				                                         *vertex_positions_in.data(neighbor_index, 1),
				                                         *vertex_positions_in.data(neighbor_index, 2));
				float distance = next_vertex_distance + (next_vertex_position - neighbor_vertex_position).norm();

				if (enforce_total_num_neighbors) {
					next_vertices_with_distances.push(std::make_pair(neighbor_index, distance));
				} else {
					if (distance <= max_influence) {
						next_vertices_with_distances.push(std::make_pair(neighbor_index, distance));
					}
				}
			}
		}

		// Store the nearest neighbors.
		int neighbor_count = neighbor_node_indices.size();

		float weight_sum = 0.f;
		for (int neighbor_index = 0; neighbor_index < neighbor_count; neighbor_index++) {
			*graph_edges_out.mutable_data(node_index, neighbor_index) = neighbor_node_indices[neighbor_index];
			weight_sum += neighbor_node_weights[neighbor_index];
		}

		// Normalize weights
		if (weight_sum > 0) {
			for (int neighbor_index = 0; neighbor_index < neighbor_count; neighbor_index++) {
				*graph_edges_weights_out.mutable_data(node_index, neighbor_index) = neighbor_node_weights[neighbor_index] / weight_sum;
			}
		} else if (neighbor_count > 0) {
			for (int neighbor_index = 0; neighbor_index < neighbor_count; neighbor_index++) {
				*graph_edges_weights_out.mutable_data(node_index, neighbor_index) = neighbor_node_weights[neighbor_index] / neighbor_count;
			}
		}

		// Store edge distance.
		for (int neighbor_index = 0; neighbor_index < neighbor_count; neighbor_index++) {
			*graph_edges_distances_out.mutable_data(node_index, neighbor_index) = neighbor_node_distances[neighbor_index];
		}
	}
}

/**
 * \brief Compute the graph edges between nodes, connecting nearest nodes using geodesic distances.
 * \param vertex_positions_in
 * \param vertex_mask_in
 * \param face_indices_in
 * \param node_indices_in
 * \param max_neighbor_count
 * \param node_coverage
 * \param graph_edges_out
 * \param graph_edges_weights_out
 * \param graph_edges_distances_out
 * \param node_to_vertex_distances_out
 * \param enforce_total_num_neighbors
 */
void compute_edges_geodesic(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<bool>& vertex_mask_in,
		const py::array_t<int>& face_indices_in,
		const py::array_t<int>& node_indices_in,
		const int max_neighbor_count, const float node_coverage,
		py::array_t<int>& graph_edges_out,
		py::array_t<float>& graph_edges_weights_out,
		py::array_t<float>& graph_edges_distances_out,
		py::array_t<float>& node_to_vertex_distances_out,
		const bool enforce_total_num_neighbors
) {
	compute_edges_geodesic_generic(vertex_positions_in, face_indices_in, node_indices_in, max_neighbor_count, node_coverage,
	                               graph_edges_out, graph_edges_weights_out, graph_edges_distances_out, node_to_vertex_distances_out,
	                               enforce_total_num_neighbors,
	                               [&vertex_mask_in](int vertex_index) { return *vertex_mask_in.data(vertex_index); });
}

void compute_edges_geodesic(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<int>& face_indices_in,
		const py::array_t<int>& node_indices_in,
		const int max_neighbor_count, const float node_coverage,
		py::array_t<int>& graph_edges_out,
		py::array_t<float>& graph_edges_weights_out,
		py::array_t<float>& graph_edges_distances_out,
		py::array_t<float>& node_to_vertex_distances_out,
		const bool enforce_total_num_neighbors
) {
	compute_edges_geodesic_generic(vertex_positions_in, face_indices_in, node_indices_in, max_neighbor_count, node_coverage,
	                               graph_edges_out, graph_edges_weights_out, graph_edges_distances_out, node_to_vertex_distances_out,
	                               enforce_total_num_neighbors,
	                               [](int vertex_index) { return true; });
}

py::tuple compute_edges_geodesic(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<bool>& vertex_mask_in,
		const py::array_t<int>& face_indices_in,
		const py::array_t<int>& node_indices_in,
		const int max_neighbor_count,
		const float node_coverage,
		const bool enforce_total_num_neighbors) {


	const ssize_t node_count = node_indices_in.shape(0);
	const ssize_t vertex_count = vertex_positions_in.shape(0);

	py::array_t<int> graph_edges_out({node_count, static_cast<ssize_t>(max_neighbor_count)});
	std::fill_n(graph_edges_out.mutable_data(0, 0), graph_edges_out.size(), -1);

	py::array_t<float> graph_edges_weights_out({node_count, static_cast<ssize_t>(max_neighbor_count)});
	memset(graph_edges_weights_out.mutable_data(0, 0), 0, graph_edges_weights_out.size() * sizeof(float));

	py::array_t<float> graph_edges_distances_out({node_count, static_cast<ssize_t>(max_neighbor_count)});
	memset(graph_edges_distances_out.mutable_data(0, 0), 0, graph_edges_distances_out.size() * sizeof(float));

	py::array_t<float> node_to_vertex_distances_out({node_count, vertex_count});
	std::fill_n(node_to_vertex_distances_out.mutable_data(0, 0), node_to_vertex_distances_out.size(), -1.f);

	compute_edges_geodesic(vertex_positions_in, vertex_mask_in, face_indices_in, node_indices_in, max_neighbor_count, node_coverage,
	                       graph_edges_out, graph_edges_weights_out, graph_edges_distances_out, node_to_vertex_distances_out,
	                       enforce_total_num_neighbors);

	return py::make_tuple(graph_edges_out, graph_edges_weights_out, graph_edges_distances_out, node_to_vertex_distances_out);
}

py::tuple compute_edges_geodesic(
		const py::array_t<float>& vertex_positions_in,
		const py::array_t<int>& face_indices_in,
		const py::array_t<int>& node_indices_in,
		const int max_neighbor_count, const float node_coverage,
		const bool enforce_total_num_neighbors
) {
	const ssize_t node_count = node_indices_in.shape(0);
	const ssize_t vertex_count = vertex_positions_in.shape(0);

	py::array_t<int> graph_edges_out({node_count, static_cast<ssize_t>(max_neighbor_count)});
	std::fill_n(graph_edges_out.mutable_data(0, 0), graph_edges_out.size(), -1);

	py::array_t<float> graph_edges_weights_out({node_count, static_cast<ssize_t>(max_neighbor_count)});
	memset(graph_edges_weights_out.mutable_data(0, 0), 0, graph_edges_weights_out.size() * sizeof(float));

	py::array_t<float> graph_edges_distances_out({node_count, static_cast<ssize_t>(max_neighbor_count)});
	memset(graph_edges_distances_out.mutable_data(0, 0), 0, graph_edges_distances_out.size() * sizeof(float));

	py::array_t<float> node_to_vertex_distances_out({node_count, vertex_count});
	std::fill_n(node_to_vertex_distances_out.mutable_data(0, 0), node_to_vertex_distances_out.size(), -1.f);

	compute_edges_geodesic(vertex_positions_in, face_indices_in, node_indices_in, max_neighbor_count, node_coverage,
	                       graph_edges_out, graph_edges_weights_out, graph_edges_distances_out, node_to_vertex_distances_out,
	                       enforce_total_num_neighbors);

	return py::make_tuple(graph_edges_out, graph_edges_weights_out, graph_edges_distances_out, node_to_vertex_distances_out);
}


py::array_t<int> compute_edges_euclidean(const py::array_t<float>& node_positions, int max_neighbor_count) {
	int node_count = node_positions.shape(0);

	py::array_t<int> graph_edges = py::array_t<int>({node_count, max_neighbor_count});

	// Find nearest Euclidean neighbors for each node.
	for (int source_node_index = 0; source_node_index < node_count; source_node_index++) {
		Eigen::Vector3f node_position(*node_positions.data(source_node_index, 0),
		                              *node_positions.data(source_node_index, 1),
		                              *node_positions.data(source_node_index, 2));

		// Keep only the k nearest Euclidean neighbors.
		std::list<std::pair<int, float>> nearest_nodes_with_squared_distances;

		for (int neighbor_index = 0; neighbor_index < node_count; neighbor_index++) {
			if (neighbor_index == source_node_index) continue;

			Eigen::Vector3f neighbor_position(*node_positions.data(neighbor_index, 0), *node_positions.data(neighbor_index, 1),
			                                  *node_positions.data(neighbor_index, 2));

			float squared_distance = (node_position - neighbor_position).squaredNorm();
			bool neighbor_inserted = false;
			for (auto it = nearest_nodes_with_squared_distances.begin(); it != nearest_nodes_with_squared_distances.end(); ++it) {
				// We insert the element at the first position where its distance is smaller than the other
				// element's distance, which enables us to always keep a sorted list of at most k nearest
				// neighbors.
				if (squared_distance <= it->second) {
					it = nearest_nodes_with_squared_distances.insert(it, std::make_pair(neighbor_index, squared_distance));
					neighbor_inserted = true;
					break;
				}
			}

			if (!neighbor_inserted && nearest_nodes_with_squared_distances.size() < max_neighbor_count) {
				nearest_nodes_with_squared_distances.emplace_back(std::make_pair(neighbor_index, squared_distance));
			}

			// We keep only the list of k nearest elements.
			if (neighbor_inserted && nearest_nodes_with_squared_distances.size() > max_neighbor_count) {
				nearest_nodes_with_squared_distances.pop_back();
			}
		}

		// Store nearest neighbor indices.
		int neighbor_index = 0;
		for (auto& nearest_nodes_with_squared_distance : nearest_nodes_with_squared_distances) {
			int destination_node_index = nearest_nodes_with_squared_distance.first;
			*graph_edges.mutable_data(source_node_index, neighbor_index) = destination_node_index;
			neighbor_index++;
		}

		for (neighbor_index = nearest_nodes_with_squared_distances.size(); neighbor_index < max_neighbor_count; neighbor_index++) {
			*graph_edges.mutable_data(source_node_index, neighbor_index) = -1;
		}
	}

	return graph_edges;
}

inline int traverse_neighbors(const std::vector<std::set<int>>& node_neighbors, std::vector<int>& cluster_ids, int cluster_id, int node_id) {
	if (cluster_ids[node_id] != -1) return 0;

	std::set<int> active_node_indices;

	// Initialize with current node.
	int cluster_size = 0;
	active_node_indices.insert(node_id);

	// Process until we have no active nodes anymore.
	while (!active_node_indices.empty()) {
		int active_node_id = *active_node_indices.begin();
		active_node_indices.erase(active_node_indices.begin());

		if (cluster_ids[active_node_id] == -1) {
			cluster_ids[active_node_id] = cluster_id;
			++cluster_size;
		}

		// Look if we need to process any of the neighbors
		for (const auto& n_idx : node_neighbors[active_node_id]) {
			if (cluster_ids[n_idx] == -1) {    // If it doesn't have a cluster yet
				active_node_indices.insert(n_idx);
			}
		}
	}

	return cluster_size;
}

void node_and_edge_clean_up(const py::array_t<int>& graph_edges, py::array_t<bool>& valid_nodes_mask) {
	int num_nodes = graph_edges.shape(0);
	int max_num_neighbors = graph_edges.shape(1);

	std::list<int> removed_nodes;

	while (true) {
		int num_newly_removed_nodes = 0;

		for (int node_id = 0; node_id < num_nodes; ++node_id) {

			if (*valid_nodes_mask.data(node_id, 0) == false) {
				// if node has been already removed, continue
				continue;
			}

			int num_neighbors = 0;
			for (int i = 0; i < max_num_neighbors; ++i) {

				int neighbor_id = *graph_edges.data(node_id, i);

				// if neighboring node is -1, break, since by design 'graph_edges' has
				// the shape [2, 3, 6, -1, -1, -1, -1, -1]
				if (neighbor_id == -1) {
					break;
				}

				// if neighboring node has been marked as invalid, continue
				if (std::find(removed_nodes.begin(), removed_nodes.end(), neighbor_id) != removed_nodes.end()) {
					continue;
				}

				++num_neighbors;
			}

			if (num_neighbors <= 1) {
				// remove node
				*valid_nodes_mask.mutable_data(node_id, 0) = false;
				removed_nodes.emplace_back(node_id);
				// std::cout << "\tremoving node_id " << node_id << std::endl;
				++num_newly_removed_nodes;
			}
		}

		// std::cout << "num_newly_removed_nodes: " << num_newly_removed_nodes << std::endl;

		if (num_newly_removed_nodes == 0) {
			break;
		}
	}
}

py::tuple compute_clusters(
		const py::array_t<int>& graph_edges_in
) {

	int node_count = graph_edges_in.shape(0);
	int max_neighbor_count = graph_edges_in.shape(1);

	py::array_t<int> graph_clusters_out({static_cast<ssize_t>(node_count), static_cast<ssize_t>(1)});
	std::fill_n(graph_clusters_out.mutable_data(0, 0), graph_clusters_out.size(), -1);

	// convert graph_edges to a vector of sets
	std::vector<std::set<int>> node_neighbors(node_count);

	for (int node_id = 0; node_id < node_count; ++node_id) {
		for (int neighbor_idx = 0; neighbor_idx < max_neighbor_count; ++neighbor_idx) {

			int neighbor_id = *graph_edges_in.data(node_id, neighbor_idx);

			if (neighbor_id == -1) {
				break;
			}

			node_neighbors[node_id].insert(neighbor_id);
			node_neighbors[neighbor_id].insert(node_id);
		}
	}

	std::vector<int> cluster_ids(node_count, -1);
	std::vector<int> cluster_sizes_out;

	int cluster_id = 0;
	for (int node_id = 0; node_id < node_count; ++node_id) {
		int cluster_size = traverse_neighbors(node_neighbors, cluster_ids, cluster_id, node_id);
		if (cluster_size > 0) {
			cluster_id++;
			cluster_sizes_out.push_back(cluster_size);
		}
	}

	for (int node_id = 0; node_id < node_count; ++node_id) {
		*graph_clusters_out.mutable_data(node_id, 0) = cluster_ids[node_id];
	}

	return py::make_tuple(cluster_sizes_out, graph_clusters_out);
}

inline void find_nearest_nodes(
		const py::array_t<float>& node_to_vertex_distance,
		const py::array_t<int>& valid_nodes_mask,
		const int vertex_id,
		std::vector<int>& nearest_node_indexes,
		std::vector<float>& nearest_node_distances
) {
	int num_nodes = node_to_vertex_distance.shape(0);

	std::map<int, float> node_map;

	for (int node_index = 0; node_index < num_nodes; ++node_index) {

		// discard node if it was marked as invalid (due to not having enough neighbors)
		if (*valid_nodes_mask.data(node_index, 0) == false) {
			continue;
		}

		float dist = *node_to_vertex_distance.data(node_index, vertex_id);

		if (dist >= 0) {
			node_map.emplace(node_index, dist);
		}
	}

	// Sort the map by distance
	// Declaring the type of Predicate that accepts 2 pairs and return a bool
	typedef std::function<bool(std::pair<int, float>, std::pair<int, float>)> IntFloatPairComparator;

	// Defining a lambda function to compare two pairs. It will compare two pairs using second field
	IntFloatPairComparator compare_by_float =
			[](std::pair<int, float> node1, std::pair<int, float> node2) {
				return node1.second < node2.second;
			};

	// Declaring a set that will store the pairs using above comparison logic
	std::set<std::pair<int, float>, IntFloatPairComparator> node_set(
			node_map.begin(), node_map.end(), compare_by_float
	);

	for (auto node : node_set) {
		nearest_node_indexes.push_back(node.first);
		nearest_node_distances.push_back(node.second);

		if (nearest_node_indexes.size() == GRAPH_K) {
			break;
		}
	}
}

void compute_pixel_anchors_geodesic(
		const py::array_t<float>& node_to_vertex_shortest_path_distance,
		const py::array_t<int>& valid_nodes_mask,
		const py::array_t<float>& vertices,
		const py::array_t<int>& vertex_pixels,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights,
		const int width, const int height,
		const float node_coverage
) {
	// Allocate graph node ids and corresponding skinning weights.
	// Initialize with invalid anchors.
	pixel_anchors.resize({height, width, GRAPH_K}, false);
	pixel_weights.resize({height, width, GRAPH_K}, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int k = 0; k < GRAPH_K; k++) {
				*pixel_anchors.mutable_data(y, x, k) = -1;
				*pixel_weights.mutable_data(y, x, k) = 0.f;
			}
		}
	}

	const int vertex_count = vertices.shape(0);

#pragma omp parallel for default(none) shared(vertex_pixels, pixel_anchors, pixel_weights, valid_nodes_mask, \
    node_to_vertex_shortest_path_distance) firstprivate(vertex_count, node_coverage)
	for (int vertex_id = 0; vertex_id < vertex_count; vertex_id++) {
		// Get corresponding pixel location
		int u = *vertex_pixels.data(vertex_id, 0);
		int v = *vertex_pixels.data(vertex_id, 1);

		// Initialize some variables
		std::vector<int> nearest_geodesic_node_ids;
		std::vector<float> dist_to_nearest_geodesic_nodes;
		std::vector<float> skinning_weights;

		nearest_geodesic_node_ids.reserve(GRAPH_K);
		dist_to_nearest_geodesic_nodes.reserve(GRAPH_K);
		skinning_weights.reserve(GRAPH_K);

		// Find closest geodesic nodes
		find_nearest_nodes(
				node_to_vertex_shortest_path_distance, valid_nodes_mask, vertex_id,
				nearest_geodesic_node_ids, dist_to_nearest_geodesic_nodes
		);

		int anchor_count = nearest_geodesic_node_ids.size();

		// Compute skinning weights.
		float weight_sum{0.f};
		for (int i = 0; i < anchor_count; ++i) {
			float geodesic_dist_to_node = dist_to_nearest_geodesic_nodes[i];

			float weight = compute_anchor_weight(geodesic_dist_to_node, node_coverage);
			weight_sum += weight;

			skinning_weights.push_back(weight);
		}

		// Normalize the skinning weights.
		if (weight_sum > 0) {
			for (int anchor_index = 0; anchor_index < anchor_count; anchor_index++) {
				skinning_weights[anchor_index] /= weight_sum;
			}
		} else if (anchor_count > 0) {
			for (int anchor_index = 0; anchor_index < anchor_count; anchor_index++) {
				skinning_weights[anchor_index] = 1.f / static_cast<float>(anchor_count);
			}
		}

		// Store the results.
		for (int i = 0; i < anchor_count; i++) {
			*pixel_anchors.mutable_data(v, u, i) = nearest_geodesic_node_ids[i];
			*pixel_weights.mutable_data(v, u, i) = skinning_weights[i];
		}
	}
}

py::tuple compute_pixel_anchors_geodesic(
		const py::array_t<float>& node_to_vertex_distance,
		const py::array_t<int>& valid_nodes_mask,
		const py::array_t<float>& vertices,
		const py::array_t<int>& vertex_pixels,
		int width, int height,
		float node_coverage) {
	py::array_t<int> pixel_anchors;
	py::array_t<float> pixel_weights;
	compute_pixel_anchors_geodesic(node_to_vertex_distance, valid_nodes_mask, vertices, vertex_pixels,
	                               pixel_anchors, pixel_weights, width, height, node_coverage);
	return py::make_tuple(pixel_anchors, pixel_weights);
}

/**
 * \brief For each input pixel it computes 4 nearest anchors, using Euclidean distances.
 * It also compute skinning weights for every pixel.
 * \param graph_nodes N X 3 graph node positions
 * \param point_image height x width x 3 point cloud
 * \param node_coverage the maximal distance of each point in the point cloud to the nearest node used to generate the graph
 * \param pixel_anchors [out] height x width x 4 array, where each 2d sampling with four values holds the indices of the nodes
 * chosen as (controlling) anchor points for the pixel at the same 2d location
 * \param pixel_weights [out] height x width x 4 array, where each 2d sampling with four values holds the
 * influence weights of the nodes assigned to the pixel in pixel_anchors
 */
void compute_pixel_anchors_euclidean(
		const py::array_t<float>& graph_nodes,
		const py::array_t<float>& point_image,
		const float node_coverage,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights
) {
	const int node_count = graph_nodes.shape(0);
	const int width = point_image.shape(1);
	const int height = point_image.shape(0);

	// Allocate graph node ids and corresponding skinning weights.
	// Initialize with invalid anchors.
	pixel_anchors.resize({height, width, GRAPH_K}, false);
	pixel_weights.resize({height, width, GRAPH_K}, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int k = 0; k < GRAPH_K; k++) {
				*pixel_anchors.mutable_data(y, x, k) = -1;
				*pixel_weights.mutable_data(y, x, k) = 0.f;
			}
		}
	}

	// Compute anchors for every pixel.
#pragma omp parallel for default(none) shared(point_image, pixel_anchors, pixel_weights, graph_nodes), firstprivate(height, width, node_count, node_coverage)
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Query 3d pixel position.
			Eigen::Vector3f pixel_position(*point_image.data(y, x, 0),
			                               *point_image.data(y, x, 1),
			                               *point_image.data(y, x, 2));
			if (pixel_position.z() <= 0) continue;

			// Keep only the k nearest Euclidean neighbors.
			std::list<std::pair<int, float>> nearest_nodes_with_squared_distances;

			for (int node_index = 0; node_index < node_count; node_index++) {
				Eigen::Vector3f node_position(*graph_nodes.data(node_index, 0),
				                              *graph_nodes.data(node_index, 1),
				                              *graph_nodes.data(node_index, 2));

				float squared_distance = (pixel_position - node_position).squaredNorm();
				bool nodes_inserted = false;
				for (auto it = nearest_nodes_with_squared_distances.begin(); it != nearest_nodes_with_squared_distances.end(); ++it) {
					// We insert the element at the first position where its distance is smaller than the other
					// element's distance, which enables us to always keep a sorted list of at most k nearest
					// neighbors.
					if (squared_distance <= it->second) {
						it = nearest_nodes_with_squared_distances.insert(it, std::make_pair(node_index, squared_distance));
						nodes_inserted = true;
						break;
					}
				}

				if (!nodes_inserted && nearest_nodes_with_squared_distances.size() < GRAPH_K) {
					nearest_nodes_with_squared_distances.emplace_back(std::make_pair(node_index, squared_distance));
				}

				// We keep only the list of k nearest elements.
				if (nodes_inserted && nearest_nodes_with_squared_distances.size() > GRAPH_K) {
					nearest_nodes_with_squared_distances.pop_back();
				}
			}

			// Compute skinning weights.
			std::vector<int> nearest_euclidean_node_indices;
			nearest_euclidean_node_indices.reserve(nearest_nodes_with_squared_distances.size());

			std::vector<float> skinning_weights;
			skinning_weights.reserve(nearest_nodes_with_squared_distances.size());

			float weight_sum{0.f};
			for (auto& nearest_nodes_with_squared_distance : nearest_nodes_with_squared_distances) {
				int node_index = nearest_nodes_with_squared_distance.first;

				Eigen::Vector3f node_position(*graph_nodes.data(node_index, 0),
				                              *graph_nodes.data(node_index, 1),
				                              *graph_nodes.data(node_index, 2));
				float weight = compute_anchor_weight(pixel_position, node_position, node_coverage);
				weight_sum += weight;

				nearest_euclidean_node_indices.push_back(node_index);
				skinning_weights.push_back(weight);
			}

			// Normalize the skinning weights.
			int anchor_count = static_cast<int>(nearest_euclidean_node_indices.size());

			if (weight_sum > 0) {
				for (int i = 0; i < anchor_count; i++) skinning_weights[i] /= weight_sum;
			} else if (anchor_count > 0) {
				for (int i = 0; i < anchor_count; i++) skinning_weights[i] = 1.f / static_cast<float>(anchor_count);
			}

			// Store the results.
			for (int i = 0; i < anchor_count; i++) {
				*pixel_anchors.mutable_data(y, x, i) = nearest_euclidean_node_indices[i];
				*pixel_weights.mutable_data(y, x, i) = skinning_weights[i];
			}
		}
	}
}

// All-pass filter for easy decltype and thread-friendly default arguments
const auto TruePredicate = [](int x) { return true; };

inline decltype(TruePredicate) GetTruePredicate() {
	return TruePredicate;
}

template<typename Predicate = decltype(TruePredicate)>
inline std::list<std::pair<int, float>>
get_knn_with_squared_distances_brute_force(const Eigen::Vector3f& vertex, const py::array_t<float>& graph_nodes,
                                           const int neighbor_count, Predicate filter = GetTruePredicate()) {
// Keep only the k nearest Euclidean neighbors.
	std::list<std::pair<int, float>> nearest_nodes_with_squared_distances;

	for (int node_index = 0; node_index < graph_nodes.shape(0); node_index++) {
		if (!filter(node_index))
			continue;

		Eigen::Vector3f node_position(*graph_nodes.data(node_index, 0),
		                              *graph_nodes.data(node_index, 1),
		                              *graph_nodes.data(node_index, 2));

		float squared_distance = (vertex - node_position).squaredNorm();
		bool nodes_inserted = false;
		for (auto it = nearest_nodes_with_squared_distances.begin(); it != nearest_nodes_with_squared_distances.end(); ++it) {
			// We insert the element at the first position where its distance is smaller than the other
			// element's distance, which enables us to always keep a sorted list of at most k nearest
			// neighbors.
			if (squared_distance <= it->second) {
				it = nearest_nodes_with_squared_distances.insert(it, std::make_pair(node_index, squared_distance));
				nodes_inserted = true;
				break;
			}
		}

		if (!nodes_inserted && nearest_nodes_with_squared_distances.size() < neighbor_count) {
			nearest_nodes_with_squared_distances.emplace_back(std::make_pair(node_index, squared_distance));
		}

		// We keep only the list of k nearest elements.
		if (nodes_inserted && nearest_nodes_with_squared_distances.size() > neighbor_count) {
			nearest_nodes_with_squared_distances.pop_back();
		}
	}
	return nearest_nodes_with_squared_distances;
}

py::tuple compute_pixel_anchors_euclidean(
		const py::array_t<float>& graph_nodes,
		const py::array_t<float>& point_image,
		float node_coverage
) {
	py::array_t<int> pixel_anchors;
	py::array_t<float> pixel_weights;

	compute_pixel_anchors_euclidean(graph_nodes, point_image, node_coverage, pixel_anchors, pixel_weights);

	return py::make_tuple(pixel_anchors, pixel_weights);
}

py::tuple compute_vertex_anchors_euclidean(
		const py::array_t<float>& graph_nodes,
		const py::array_t<float>& vertices,
		float node_coverage
) {
	py::array_t<int> vertex_anchors({vertices.shape(0), static_cast<ssize_t>(GRAPH_K)});
	py::array_t<float> vertex_weights({vertices.shape(0), static_cast<ssize_t>(GRAPH_K)});
	std::fill_n(vertex_anchors.mutable_data(0), vertex_anchors.size(), -1);
	std::fill_n(vertex_weights.mutable_data(0), vertex_weights.size(), 0.f);

#pragma omp parallel for default(none) shared(vertices, graph_nodes, vertex_anchors, vertex_weights), firstprivate(node_coverage)
	for (int i_vertex = 0; i_vertex < vertices.shape(0); i_vertex++) {
		// Query 3d pixel position.
		Eigen::Vector3f vertex(*vertices.data(i_vertex, 0),
		                       *vertices.data(i_vertex, 1),
		                       *vertices.data(i_vertex, 2));

		// Keep only the k nearest Euclidean neighbors.
		std::list<std::pair<int, float>> nearest_nodes_with_squared_distances =
				get_knn_with_squared_distances_brute_force(vertex, graph_nodes,GRAPH_K);

		// Compute skinning weights.
		std::vector<int> nearest_euclidean_node_indices;
		nearest_euclidean_node_indices.reserve(nearest_nodes_with_squared_distances.size());

		std::vector<float> skinning_weights;
		skinning_weights.reserve(nearest_nodes_with_squared_distances.size());

		float weight_sum{0.f};
		for (auto& nearest_nodes_with_squared_distance : nearest_nodes_with_squared_distances) {
			int node_index = nearest_nodes_with_squared_distance.first;

			Eigen::Vector3f node_position(*graph_nodes.data(node_index, 0),
			                              *graph_nodes.data(node_index, 1),
			                              *graph_nodes.data(node_index, 2));
			float weight = compute_anchor_weight(vertex, node_position, node_coverage);
			weight_sum += weight;

			nearest_euclidean_node_indices.push_back(node_index);
			skinning_weights.push_back(weight);
		}

		// Normalize the skinning weights.
		int anchor_count = static_cast<int>(nearest_euclidean_node_indices.size());

		if (weight_sum > 0) {
			for (int i = 0; i < anchor_count; i++) skinning_weights[i] /= weight_sum;
		} else if (anchor_count > 0) {
			for (int i = 0; i < anchor_count; i++) skinning_weights[i] = 1.f / static_cast<float>(anchor_count);
		}

		// Store the results.
		for (int i_anchor_node = 0; i_anchor_node < anchor_count; i_anchor_node++) {
			*vertex_anchors.mutable_data(i_vertex, i_anchor_node) = nearest_euclidean_node_indices[i_anchor_node];
			*vertex_weights.mutable_data(i_vertex, i_anchor_node) = skinning_weights[i_anchor_node];
		}
	}
	return py::make_tuple(vertex_anchors, vertex_weights);
}

/**
  * Sample graph regularly from the image, using pixel-wise connectivity
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
) {
	int width = point_image.shape(1);
	int height = point_image.shape(0);

	float x_step = float(width - 1) / static_cast<float>(x_nodes - 1);
	float y_step = float(height - 1) / static_cast<float>(y_nodes - 1);

	// Sample graph nodes.
	// We need to maintain the mapping from all -> valid nodes ids.
	int node_count = x_nodes * y_nodes;
	std::vector<int> sampled_node_mapping(node_count, -1);

	std::vector<Eigen::Vector3f> node_positions;
	node_positions.reserve(node_count);

	int sampled_node_count = 0;
	for (int y = 0; y < y_nodes; y++) {
		for (int x = 0; x < x_nodes; x++) {
			int linear_node_index = y * x_nodes + x;

			// We use nearest neighbor interpolation for node position
			// computation.
			int x_pixel = static_cast<int>(std::round(static_cast<float>(x) * x_step));
			int y_pixel = static_cast<int>(std::round(static_cast<float>(y) * y_step));

			Eigen::Vector3f pixel_position(*point_image.data(y_pixel, x_pixel, 0),
			                               *point_image.data(y_pixel, x_pixel, 1),
			                               *point_image.data(y_pixel, x_pixel, 2));
			if (pixel_position.z() <= 0 || pixel_position.z() > max_depth) continue;

			node_positions.push_back(pixel_position);
			sampled_node_mapping[linear_node_index] = sampled_node_count;
			sampled_node_count++;
		}
	}

	// Compute graph edges using pixel-wise connectivity. Each node
	// is connected with at most 8 neighboring pixels.
	int max_neighbor_count = 8;
	float edge_threshold_squared = edge_threshold * edge_threshold;

	std::vector<int> sampled_node_edges(sampled_node_count * max_neighbor_count, -1);
	std::vector<bool> connected_nodes(sampled_node_count, false);

	int connected_node_count = 0;
	for (int y = 0; y < y_nodes; y++) {
		for (int x = 0; x < x_nodes; x++) {
			int node_index = y * x_nodes + x;
			int sampled_node_index = sampled_node_mapping[node_index];

			if (sampled_node_index >= 0) {
				Eigen::Vector3f node_position = node_positions[sampled_node_index];

				int neighbor_count = 0;
				for (int y_delta = -1; y_delta <= 1; y_delta++) {
					for (int x_delta = -1; x_delta <= 1; x_delta++) {
						int x_neighbor = x + x_delta;
						int y_neighbor = y + y_delta;
						if (x_neighbor < 0 || x_neighbor >= x_nodes || y_neighbor < 0 || y_neighbor >= y_nodes)
							continue;

						int neighbor_index = y_neighbor * x_nodes + x_neighbor;

						if (neighbor_index == node_index || neighbor_index < 0)
							continue;

						int sampled_neighbor_index = sampled_node_mapping[neighbor_index];
						if (sampled_neighbor_index >= 0) {
							Eigen::Vector3f neighbor_position = node_positions[sampled_neighbor_index];

							if ((neighbor_position - node_position).squaredNorm() <= edge_threshold_squared) {
								sampled_node_edges[sampled_node_index * max_neighbor_count + neighbor_count] = sampled_neighbor_index;
								neighbor_count++;
							}
						}
					}
				}

				for (int i = neighbor_count; i < max_neighbor_count; i++) {
					sampled_node_edges[sampled_node_index * max_neighbor_count + i] = -1;
				}

				if (neighbor_count > 0) {
					connected_nodes[sampled_node_index] = true;
					connected_node_count += 1;
				}
			}
		}
	}

	// Filter out canonical_node_positions with no edges.
	// After changing node ids the edge ids need to be changed as well.
	std::vector<int> valid_node_mapping(sampled_node_count, -1);
	{
		graph_nodes.resize({connected_node_count, 3}, false);
		graph_edges.resize({connected_node_count, max_neighbor_count}, false);

		int valid_node_index = 0;
		for (int y = 0; y < y_nodes; y++) {
			for (int x = 0; x < x_nodes; x++) {
				int node_index = y * x_nodes + x;
				int sampled_node_index = sampled_node_mapping[node_index];

				if (sampled_node_index >= 0 && connected_nodes[sampled_node_index]) {
					valid_node_mapping[sampled_node_index] = valid_node_index;

					Eigen::Vector3f node_position = node_positions[sampled_node_index];
					*graph_nodes.mutable_data(valid_node_index, 0) = node_position.x();
					*graph_nodes.mutable_data(valid_node_index, 1) = node_position.y();
					*graph_nodes.mutable_data(valid_node_index, 2) = node_position.z();

					valid_node_index++;
				}
			}
		}
	}

	for (int y = 0; y < y_nodes; y++) {
		for (int x = 0; x < x_nodes; x++) {
			int node_index = y * x_nodes + x;
			int sampled_node_index = sampled_node_mapping[node_index];

			if (sampled_node_index >= 0 && connected_nodes[sampled_node_index]) {
				int valid_node_index = valid_node_mapping[sampled_node_index];

				if (valid_node_index >= 0) {
					for (int i = 0; i < max_neighbor_count; i++) {
						int sampledNeighborId = sampled_node_edges[sampled_node_index * max_neighbor_count + i];
						if (sampledNeighborId >= 0) {
							*graph_edges.mutable_data(valid_node_index, i) = valid_node_mapping[sampledNeighborId];
						} else {
							*graph_edges.mutable_data(valid_node_index, i) = -1;
						}
					}
				}
			}
		}
	}

	// Compute pixel anchors and weights.
	pixel_anchors.resize({height, width, 4}, false);
	pixel_weights.resize({height, width, 4}, false);

	float max_point_to_node_distance_squared = max_point_to_node_distance * max_point_to_node_distance;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Initialize with invalid values.
			for (int k = 0; k < 4; k++) {
				*pixel_anchors.mutable_data(y, x, k) = -1;
				*pixel_weights.mutable_data(y, x, k) = 0.f;
			}

			// Compute 4 nearest canonical_node_positions.
			float x_node = float(x) / x_step;
			float y_node = float(y) / y_step;

			int x0 = std::floor(x_node), x1 = x0 + 1;
			int y0 = std::floor(y_node), y1 = y0 + 1;

			// Check that all neighboring nodes are valid.
			if (x0 < 0 || x1 >= x_nodes || y0 < 0 || y1 >= y_nodes)
				continue;

			int sampled_node_00 = sampled_node_mapping[y0 * x_nodes + x0];
			int sampled_node_01 = sampled_node_mapping[y1 * x_nodes + x0];
			int sampled_node_10 = sampled_node_mapping[y0 * x_nodes + x1];
			int sampled_node_11 = sampled_node_mapping[y1 * x_nodes + x1];

			if (sampled_node_00 < 0 || sampled_node_01 < 0 || sampled_node_10 < 0 || sampled_node_11 < 0)
				continue;

			int valid_node_00 = valid_node_mapping[sampled_node_00];
			int valid_node_01 = valid_node_mapping[sampled_node_01];
			int valid_node_10 = valid_node_mapping[sampled_node_10];
			int valid_node_11 = valid_node_mapping[sampled_node_11];

			if (valid_node_00 < 0 || valid_node_01 < 0 || valid_node_10 < 0 || valid_node_11 < 0)
				continue;

			// Check that all nodes are close enough to the point.
			Eigen::Vector3f pixel_position(*point_image.data(y, x, 0),
			                               *point_image.data(y, x, 1),
			                               *point_image.data(y, x, 2));
			if (pixel_position.z() <= 0 || pixel_position.z() > max_depth) continue;

			if ((pixel_position - node_positions[sampled_node_00]).squaredNorm() > max_point_to_node_distance_squared ||
			    (pixel_position - node_positions[sampled_node_01]).squaredNorm() > max_point_to_node_distance_squared ||
			    (pixel_position - node_positions[sampled_node_10]).squaredNorm() > max_point_to_node_distance_squared ||
			    (pixel_position - node_positions[sampled_node_11]).squaredNorm() > max_point_to_node_distance_squared) {
				continue;
			}

			// Compute bilinear weights.
			float dx = x_node - static_cast<float>(x0);
			float dy = y_node - static_cast<float>(y0);

			float w00 = (1 - dx) * (1 - dy);
			float w01 = (1 - dx) * dy;
			float w10 = dx * (1 - dy);
			float w11 = dx * dy;

			*pixel_anchors.mutable_data(y, x, 0) = valid_node_00;
			*pixel_weights.mutable_data(y, x, 0) = w00;
			*pixel_anchors.mutable_data(y, x, 1) = valid_node_01;
			*pixel_weights.mutable_data(y, x, 1) = w01;
			*pixel_anchors.mutable_data(y, x, 2) = valid_node_10;
			*pixel_weights.mutable_data(y, x, 2) = w10;
			*pixel_anchors.mutable_data(y, x, 3) = valid_node_11;
			*pixel_weights.mutable_data(y, x, 3) = w11;
		}
	}
}

py::tuple construct_regular_graph(
		const py::array_t<float>& point_image,
		int x_nodes, int y_nodes,
		float edge_threshold,
		float max_point_to_node_distance,
		float max_depth
) {
	py::array_t<float> graph_nodes;
	py::array_t<int> graph_edges;
	py::array_t<int> pixel_anchors;
	py::array_t<float> pixel_weights;
	construct_regular_graph(point_image, x_nodes, y_nodes, edge_threshold,
	                        max_point_to_node_distance, max_depth,
	                        graph_nodes, graph_edges, pixel_anchors, pixel_weights);
	return py::make_tuple(graph_nodes, graph_edges, pixel_anchors, pixel_weights);
}


void update_pixel_anchors(
		const std::map<int, int>& node_id_mapping,
		py::array_t<int>& pixel_anchors
) {
	int height = pixel_anchors.shape(0);
	int width = pixel_anchors.shape(1);
	int num_anchors = pixel_anchors.shape(2);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			for (int a = 0; a < num_anchors; a++) {

				int current_anchor_id = *pixel_anchors.data(y, x, a);

				if (current_anchor_id != -1) {
					int mapped_anchor_id = node_id_mapping.at(current_anchor_id);

					// update anchor only if it would actually change something
					if (mapped_anchor_id != current_anchor_id) {
						*pixel_anchors.mutable_data(y, x, a) = mapped_anchor_id;
					}
				}
			}
		}
	}
}

/**
 * \brief compute point anchors based on shortest path within the graph
 * \param vertex_to_node_distance
 * \param vertices
 * \param pixel_anchors
 * \param pixel_weights
 * \param width
 * \param height
 * \param node_coverage
 */
void compute_vertex_anchors_shortest_path(const py::array_t<float>& vertices,
                                          const py::array_t<float>& nodes,
		// edges are assumed to have been computed in a shortest-path manner,
		// i.e. what the NNRT authors called "geodesically" -- like via the
		// compute_edges_geodesic function in the original source,
		// for each vertex - target vertices sharing an edge, ordered by shortest edge
		                                 const py::array_t<int>& edges,
		                                  py::array_t<int>& anchors,
		                                  py::array_t<float>& weights,
		                                  int anchor_count,
		                                  float node_coverage) {
// Allocate graph node ids and corresponding skinning weights.

	int node_count = static_cast<int>(nodes.shape(0));
	int vertex_count = static_cast<int>(vertices.shape(0));

	assert(edges.shape(1) == GRAPH_K);
	assert(edges.shape(0) == node_count);

	// Initialize with invalid anchors.
	anchors.resize({node_count, anchor_count}, false);
	weights.resize({vertex_count, anchor_count}, false);

	for (int i_node = 0; i_node < node_count; i_node++) {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			*anchors.mutable_data(i_node, i_anchor) = -1;
			*weights.mutable_data(i_node, i_anchor) = 0.f;
		}
	}

#pragma omp parallel for default(none) shared(vertices, nodes, edges, anchors, weights)\
    firstprivate(vertex_count, node_coverage, anchor_count)
	for (int i_vertex = 0; i_vertex < vertex_count; i_vertex++) {
		// Query 3d pixel position.
		Eigen::Vector3f vertex(*vertices.data(i_vertex, 0),
		                       *vertices.data(i_vertex, 1),
		                       *vertices.data(i_vertex, 2));

		std::map<int, float> distance_by_shortest_path_anchor;
		typedef std::pair<int, float> index_and_distance;
		auto node_geodesic_comparator = [](const index_and_distance& a, const index_and_distance& b) {
			return a.second > b.second;
		};
		std::priority_queue<index_and_distance, std::vector<index_and_distance>,
				decltype(node_geodesic_comparator)> queue(node_geodesic_comparator);

		while (distance_by_shortest_path_anchor.size() < anchor_count) {
			// Keep only the anchor_count nearest Euclidean neighbors.
			std::list<std::pair<int, float>> nearest_nodes_with_squared_distances =
					get_knn_with_squared_distances_brute_force(vertex, nodes, 1,
					                                           [distance_by_shortest_path_anchor](int i) {
						                                           return distance_by_shortest_path_anchor.find(i) ==
						                                                  distance_by_shortest_path_anchor.end();
					                                           });
			if(nearest_nodes_with_squared_distances.empty()){
				break; // no node to initialize queue with, we've got no more valid anchors to consider
			}

			auto closest_euclidean_node_and_distance = nearest_nodes_with_squared_distances.front();
			queue.emplace(closest_euclidean_node_and_distance.first, sqrt(closest_euclidean_node_and_distance.second));

			while (!queue.empty()) {
				auto graph_node_and_distance = queue.top();
				queue.pop();

				int source_node_index = graph_node_and_distance.first;
				float source_path_distance = graph_node_and_distance.second;

				auto it = distance_by_shortest_path_anchor.find(source_node_index);
				if (it != distance_by_shortest_path_anchor.end()){
					if(it->second > source_path_distance){
						// update distance to the node if a shorter alternative path is found
						distance_by_shortest_path_anchor[source_node_index] = source_path_distance;
					}
					continue;
				}

				// insert new node in the weight map
				distance_by_shortest_path_anchor[source_node_index] = source_path_distance;
				if (distance_by_shortest_path_anchor.size() >= anchor_count)
					break;

				Eigen::Map<const Eigen::Vector3f> source_node(nodes.data(source_node_index, 0));

				for (int i_edge = 0; i_edge < GRAPH_K; i_edge++) {
					int target_node_index = *edges.data(i_vertex, i_edge);
					if (target_node_index > -1) {
						Eigen::Map<const Eigen::Vector3f> target_node(nodes.data(target_node_index, 0));
						float distance_source_to_target = (target_node - source_node).norm();
						queue.emplace(target_node_index, distance_source_to_target + source_path_distance);
					}
				}
			}
		}

		// Write to output directly
		auto vertex_anchors = anchors.mutable_data(i_vertex, 0);
		auto vertex_weights = weights.mutable_data(i_vertex, 0);

		float weight_sum = 0;
		int valid_anchor_count = 0;
		for (const auto anchor : distance_by_shortest_path_anchor) {
			vertex_anchors[valid_anchor_count] = anchor.first;
			float weight = compute_anchor_weight(anchor.second, node_coverage);
			vertex_weights[valid_anchor_count] = weight;
			weight_sum += weight;
			valid_anchor_count++;
		}

		// Normalize weights
		if (weight_sum > 0) {
			for (int i_anchor = 0; i_anchor < valid_anchor_count; i_anchor++){
				vertex_weights[i_anchor] /= weight_sum;
			}
		} else if (valid_anchor_count > 0) {
			for (int i_anchor = 0; i_anchor < valid_anchor_count; i_anchor++){
				vertex_weights[i_anchor] = 1.f / static_cast<float>(valid_anchor_count);
			}
		}
	} // end vertex loop
}

py::tuple compute_vertex_anchors_shortest_path(const py::array_t<float>& vertices, const py::array_t<float>& nodes, const py::array_t<int>& edges,
                                                int anchor_count, float node_coverage) {
	py::array_t<int> anchors;
	py::array_t<float> weights;
	compute_vertex_anchors_shortest_path(vertices, nodes, edges, anchors, weights, anchor_count, node_coverage);
	return py::make_tuple(anchors, weights);
}


} // namespace graph_proc
