import os
import sys
import typing

import numpy as np
import open3d as o3d
import open3d.core as o3c

import io as dio
import skimage.io

from settings import Parameters
import image_processing
from warp_field.graph_warp_field import GraphWarpFieldOpen3DNative

from nnrt import compute_mesh_from_depth_and_flow as compute_mesh_from_depth_and_flow_c
from nnrt import compute_mesh_from_depth as compute_mesh_from_depth_c
from nnrt import get_vertex_erosion_mask as erode_mesh_c
from nnrt import sample_nodes as sample_nodes_c
from nnrt import compute_edges_shortest_path as compute_edges_shortest_path_c
from nnrt import node_and_edge_clean_up as node_and_edge_clean_up_c
from nnrt import compute_pixel_anchors_shortest_path as compute_pixel_anchors_shortest_path_c
from nnrt import compute_clusters as compute_clusters_c
from nnrt import update_pixel_anchors as update_pixel_anchors_c


def build_graph_warp_field_from_depth_image(depth_image: np.ndarray, mask_image: np.ndarray,
                                            intrinsic_matrix: np.ndarray, device: o3d.core.Device,
                                            max_triangle_distance: float = 0.05, depth_scale_reciprocal: float = 1000.0,
                                            erosion_num_iterations: int = 10, erosion_min_neighbors: int = 4,
                                            remove_nodes_with_too_few_neighbors: bool = True,
                                            use_only_valid_vertices: bool = True,
                                            sample_random_shuffle: bool = False, neighbor_count: int = 8,
                                            enforce_neighbor_count: bool = True,
                                            scene_flow_path: typing.Union[str, None] = None,
                                            enable_visual_debugging: bool = False,
                                            node_coverage: float = 0.05,
                                            minimum_valid_anchor_count: int = 3) -> \
        typing.Tuple[GraphWarpFieldOpen3DNative, typing.Union[None, np.ndarray], np.ndarray, np.ndarray]:
    # options

    node_coverage = Parameters.graph.node_coverage.value
    graph_debug = Parameters.graph.graph_debug.value

    # extract intrinsic coefficients

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    #########################################################################
    # Convert depth to mesh.
    #########################################################################
    width = depth_image.shape[1]
    height = depth_image.shape[0]

    # Invalidate depth residuals outside object mask.
    # We only define graph over dynamic object (inside the object mask).
    mask_image[mask_image > 0] = 1
    depth_image = depth_image * mask_image

    # Backproject depth images into 3D.
    point_image = image_processing.backproject_depth(depth_image, fx, fy, cx, cy, depth_scale=depth_scale_reciprocal)
    point_image = point_image.astype(np.float32)

    # Convert depth image into mesh, using pixel-wise connectivity.
    # We also compute flow residuals, and invalidate any vertex with non-finite
    # flow residuals.

    if scene_flow_path is None:
        vertices, vertex_pixels, faces = \
            compute_mesh_from_depth_c(
                point_image, max_triangle_distance
            )
    else:
        # Load scene flow image.
        scene_flow_image = dio.load_flow(scene_flow_path)
        scene_flow_image = np.moveaxis(scene_flow_image, 0, 2)

        vertices, vertex_flows, vertex_pixels, faces = \
            compute_mesh_from_depth_and_flow_c(
                point_image, scene_flow_image,
                max_triangle_distance
            )

    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]

    assert num_vertices > 0 and num_faces > 0

    # Erode mesh, to not sample unstable nodes on the mesh boundary.
    non_eroded_vertices = erode_mesh_c(
        vertices, faces, erosion_num_iterations, erosion_min_neighbors
    )

    # Just for debugging.
    if enable_visual_debugging:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[non_eroded_vertices.reshape(-1), :]))

        o3d.visualization.draw_geometries([mesh, pcd], mesh_show_back_face=True)

        if scene_flow_path is None:
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        else:
            mesh_transformed = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices + vertex_flows),
                                                         o3d.utility.Vector3iVector(faces))
            mesh_transformed.compute_vertex_normals()
            mesh_transformed.paint_uniform_color([0.0, 1.0, 0.0])

            o3d.visualization.draw_geometries([mesh, mesh_transformed], mesh_show_back_face=True)

    #########################################################################
    # Sample graph nodes.
    #########################################################################
    valid_vertices = non_eroded_vertices

    # Sample graph nodes.
    node_coords, node_indices = sample_nodes_c(
        vertices, valid_vertices,
        node_coverage, use_only_valid_vertices,
        sample_random_shuffle
    )

    num_nodes = node_coords.shape[0]

    node_coords = node_coords[:num_nodes, :]
    node_indices = node_indices[:num_nodes, :]

    if scene_flow_path is not None:
        # Get node deformation.
        node_deformations = vertex_flows[node_indices.squeeze()]
        node_deformations = node_deformations.reshape(-1, 3)

        assert np.isfinite(node_deformations).all(), "All deformations should be valid."
        assert node_deformations.shape[0] == node_coords.shape[0] == node_indices.shape[0]
    else:
        node_deformations = None

    if enable_visual_debugging:
        pcd_nodes = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node_coords))
        o3d.visualization.draw_geometries([pcd_nodes], mesh_show_back_face=True)

    #########################################################################
    # Compute graph edges.
    #########################################################################
    # Compute edges between nodes.

    visible_vertices = np.ones_like(valid_vertices)

    graph_edges, graph_edge_weights, graph_edges_distances, node_to_vertex_distances = \
        compute_edges_shortest_path_c(
            vertices, visible_vertices, faces, node_indices,
            neighbor_count, Parameters.graph.node_coverage.value, enforce_neighbor_count
        )

    # Remove nodes 
    valid_nodes_mask = np.ones((num_nodes, 1), dtype=bool)

    if remove_nodes_with_too_few_neighbors:
        # Mark nodes with not enough neighbors
        node_and_edge_clean_up_c(graph_edges, valid_nodes_mask)

        # Get the list of invalid nodes
        node_id_black_list = np.where(valid_nodes_mask == False)[0].tolist()
    else:
        node_id_black_list = []
        if graph_debug:
            print("You're allowing nodes with not enough neighbors!")

    if graph_debug:
        print("Node filtering: initial num nodes", num_nodes, "| invalid nodes", len(node_id_black_list),
              "({})".format(node_id_black_list))

    #########################################################################
    # Compute pixel anchors.
    #########################################################################
    pixel_anchors, pixel_weights = compute_pixel_anchors_shortest_path_c(
        node_to_vertex_distances, valid_nodes_mask,
        vertices, vertex_pixels,
        width, height, node_coverage
    )

    if graph_debug:
        print("Valid pixels:", np.sum(np.all(pixel_anchors != -1, axis=2)))

    if enable_visual_debugging:
        pixel_anchors_image = np.sum(pixel_anchors, axis=2)
        pixel_anchors_mask_ed = np.copy(pixel_anchors_image).astype(np.uint8)
        pixel_anchors_mask_ed[...] = 1
        pixel_anchors_mask_ed[pixel_anchors_image == -4] = 0
        dio.save_grayscale_image("pixel_anchors_mask_ed.jpeg", pixel_anchors_mask_ed)

    # Get only valid nodes and their corresponding info
    node_coords = node_coords[valid_nodes_mask.squeeze()]
    node_indices = node_indices[valid_nodes_mask.squeeze()]

    # Apply node mask to the computed node deformations
    if node_deformations is not None:
        node_deformations = node_deformations[valid_nodes_mask.squeeze()]

    graph_edges = graph_edges[valid_nodes_mask.squeeze()]
    graph_edge_weights = graph_edge_weights[valid_nodes_mask.squeeze()]
    graph_edges_distances = graph_edges_distances[valid_nodes_mask.squeeze()]

    #########################################################################
    # Graph checks.
    #########################################################################
    num_nodes = node_coords.shape[0]

    # Check that we have enough nodes
    if num_nodes == 0:
        print("No nodes! Exiting ...")
        exit()

    # Update node ids only if we actually removed nodes
    if len(node_id_black_list) > 0:
        # 1. Mapping old indices to new indices
        count = 0
        node_id_mapping = {}
        for i, is_node_valid in enumerate(valid_nodes_mask):
            if not is_node_valid:
                node_id_mapping[i] = -1
            else:
                node_id_mapping[i] = count
                count += 1

        # 2. Update graph_edges using the id mapping
        for node_id, graph_edge in enumerate(graph_edges):
            # compute mask of valid neighbors
            valid_neighboring_nodes = np.invert(np.isin(graph_edge, node_id_black_list))

            # make a copy of the current neighbors' ids
            graph_edge_copy = np.copy(graph_edge)
            graph_edge_weights_copy = np.copy(graph_edge_weights[node_id])
            graph_edge_distances_copy = np.copy(graph_edges_distances[node_id])

            # set the neighbors' ids to -1
            graph_edges[node_id] = -np.ones_like(graph_edge_copy)
            graph_edge_weights[node_id] = np.zeros_like(graph_edge_weights_copy)
            graph_edges_distances[node_id] = np.zeros_like(graph_edge_distances_copy)

            count_valid_neighbors = 0
            for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
                if is_valid_neighbor:
                    # current neighbor id
                    current_neighbor_id = graph_edge_copy[neighbor_idx]

                    # get mapped neighbor id       
                    if current_neighbor_id == -1:
                        mapped_neighbor_id = -1
                    else:
                        mapped_neighbor_id = node_id_mapping[current_neighbor_id]

                    graph_edges[node_id, count_valid_neighbors] = mapped_neighbor_id
                    graph_edge_weights[node_id, count_valid_neighbors] = graph_edge_weights_copy[neighbor_idx]
                    graph_edges_distances[node_id, count_valid_neighbors] = graph_edge_distances_copy[neighbor_idx]

                    count_valid_neighbors += 1

            # normalize edges' weights
            sum_weights = np.sum(graph_edge_weights[node_id])
            if sum_weights > 0:
                graph_edge_weights[node_id] /= sum_weights
            else:
                raise ValueError(
                    f"Weight sum for node anchors is {sum_weights}. Weights: {str(graph_edge_weights[node_id])}.")

        # 3. Update pixel anchors using the id mapping (note that, at this point, pixel_anchors is already free of "bad" nodes, since
        # 'compute_pixel_anchors_shortest_path_c' was given 'valid_nodes_mask')
        update_pixel_anchors_c(node_id_mapping, pixel_anchors)
    #########################################################################
    # Compute clusters.
    #########################################################################
    cluster_sizes, graph_clusters = compute_clusters_c(graph_edges)

    for i, cluster_size in enumerate(cluster_sizes):
        if cluster_size <= 2:
            raise ValueError(
                f"Cluster is too small: {cluster_size}, it only has nodes: {str(np.where(graph_clusters == i)[0])}")

    nodes_o3d = o3c.Tensor(node_coords, device=device)
    edges_o3d = o3c.Tensor(graph_edges, device=device)
    edge_weights_o3d = o3c.Tensor(graph_edge_weights, device=device)
    clusters_o3d = o3c.Tensor(graph_clusters.flatten(), device=device)

    return GraphWarpFieldOpen3DNative(nodes_o3d, edges_o3d, edge_weights_o3d,
                                      clusters_o3d, node_coverage=node_coverage,
                                      threshold_nodes_by_distance=minimum_valid_anchor_count > 0,
                                      minimum_valid_anchor_count=minimum_valid_anchor_count), \
           node_deformations, pixel_anchors, pixel_weights


def generate_paths(seq_dir: str):
    dst_graph_nodes_dir = os.path.join(seq_dir, "graph_nodes")
    if not os.path.exists(dst_graph_nodes_dir): os.makedirs(dst_graph_nodes_dir)

    dst_graph_edges_dir = os.path.join(seq_dir, "graph_edges")
    if not os.path.exists(dst_graph_edges_dir):
        os.makedirs(dst_graph_edges_dir)

    dst_graph_edges_weights_dir = os.path.join(seq_dir, "graph_edges_weights")
    if not os.path.exists(dst_graph_edges_weights_dir):
        os.makedirs(dst_graph_edges_weights_dir)

    dst_graph_clusters_dir = os.path.join(seq_dir, "graph_clusters")
    if not os.path.exists(dst_graph_clusters_dir):
        os.makedirs(dst_graph_clusters_dir)

    dst_node_deformations_dir = os.path.join(seq_dir, "graph_node_deformations")
    if not os.path.exists(dst_node_deformations_dir):
        os.makedirs(dst_node_deformations_dir)

    dst_pixel_anchors_dir = os.path.join(seq_dir, "pixel_anchors")
    if not os.path.exists(dst_pixel_anchors_dir):
        os.makedirs(dst_pixel_anchors_dir)

    dst_pixel_weights_dir = os.path.join(seq_dir, "pixel_weights")
    if not os.path.exists(dst_pixel_weights_dir):
        os.makedirs(dst_pixel_weights_dir)

    return dst_graph_nodes_dir, dst_graph_edges_dir, dst_graph_edges_weights_dir, dst_pixel_weights_dir, \
           dst_graph_clusters_dir, dst_node_deformations_dir, dst_pixel_anchors_dir, dst_pixel_weights_dir


def save_graph_data(seq_dir: str, pair_name: str, node_coords: np.ndarray, graph_edges: np.ndarray,
                    graph_edges_weights: np.ndarray, graph_clusters: np.ndarray,
                    node_deformations: typing.Union[None, np.ndarray] = None,
                    pixel_anchors: typing.Union[None, np.ndarray] = None,
                    pixel_weights: typing.Union[None, np.ndarray] = None):
    node_coverage = Parameters.graph.node_coverage.value

    dst_graph_nodes_dir, dst_graph_edges_dir, dst_pixel_weights_dir, dst_graph_edges_weights_dir, dst_graph_clusters_dir, \
    dst_node_deformations_dir, dst_pixel_anchors_dir, dst_pixel_weights_dir = generate_paths(seq_dir)

    output_graph_nodes_path = os.path.join(dst_graph_nodes_dir,
                                           pair_name + "_{}_{:.2f}.bin".format("geodesic", node_coverage))
    output_graph_edges_path = os.path.join(dst_graph_edges_dir,
                                           pair_name + "_{}_{:.2f}.bin".format("geodesic", node_coverage))
    output_graph_edges_weights_path = os.path.join(dst_graph_edges_weights_dir,
                                                   pair_name + "_{}_{:.2f}.bin".format("geodesic", node_coverage))
    output_node_deformations_path = os.path.join(dst_node_deformations_dir,
                                                 pair_name + "_{}_{:.2f}.bin".format("geodesic", node_coverage))
    output_graph_clusters_path = os.path.join(dst_graph_clusters_dir,
                                              pair_name + "_{}_{:.2f}.bin".format("geodesic", node_coverage))

    output_pixel_anchors_path = os.path.join(dst_pixel_anchors_dir,
                                             pair_name + "_{}_{:.2f}.bin".format("geodesic", node_coverage))
    output_pixel_weights_path = os.path.join(dst_pixel_weights_dir,
                                             pair_name + "_{}_{:.2f}.bin".format("geodesic", node_coverage))

    dio.save_graph_nodes(output_graph_nodes_path, node_coords)
    dio.save_graph_edges(output_graph_edges_path, graph_edges)
    dio.save_graph_edges_weights(output_graph_edges_weights_path, graph_edges_weights)
    if node_deformations is not None:
        dio.save_graph_node_deformations(output_node_deformations_path, node_deformations)
    dio.save_graph_clusters(output_graph_clusters_path, graph_clusters)

    if pixel_anchors is not None:
        dio.save_int_image(output_pixel_anchors_path, pixel_anchors)
    if pixel_weights is not None:
        dio.save_float_image(output_pixel_weights_path, pixel_weights)


def check_graph_data_against_ground_truth(seq_dir: str, ground_truth_pair_name: str,
                                          node_coords: np.ndarray, graph_edges: np.ndarray,
                                          graph_edges_weights: np.ndarray, graph_clusters: np.ndarray,
                                          node_deformations: typing.Union[None, np.ndarray] = None,
                                          pixel_anchors: typing.Union[None, np.ndarray] = None,
                                          pixel_weights: typing.Union[None, np.ndarray] = None):
    node_coverage = Parameters.graph.node_coverage.value

    dst_graph_nodes_dir, dst_graph_edges_dir, dst_pixel_weights_dir, dst_graph_edges_weights_dir, dst_graph_clusters_dir, \
    dst_node_deformations_dir, dst_pixel_anchors_dir, dst_pixel_weights_dir = generate_paths(seq_dir)

    gt_output_graph_nodes_path = os.path.join(dst_graph_nodes_dir,
                                              ground_truth_pair_name + "_{}_{:.2f}.bin".format("geodesic",
                                                                                               node_coverage))
    gt_output_graph_edges_path = os.path.join(dst_graph_edges_dir,
                                              ground_truth_pair_name + "_{}_{:.2f}.bin".format("geodesic",
                                                                                               node_coverage))
    gt_output_graph_edges_weights_path = os.path.join(dst_graph_edges_weights_dir,
                                                      ground_truth_pair_name + "_{}_{:.2f}.bin".format("geodesic",
                                                                                                       node_coverage))
    gt_output_node_deformations_path = os.path.join(dst_node_deformations_dir,
                                                    ground_truth_pair_name + "_{}_{:.2f}.bin".format("geodesic",
                                                                                                     node_coverage))
    gt_output_graph_clusters_path = os.path.join(dst_graph_clusters_dir,
                                                 ground_truth_pair_name + "_{}_{:.2f}.bin".format("geodesic",
                                                                                                  node_coverage))
    gt_output_pixel_anchors_path = os.path.join(dst_pixel_anchors_dir,
                                                ground_truth_pair_name + "_{}_{:.2f}.bin".format("geodesic",
                                                                                                 node_coverage))
    gt_output_pixel_weights_path = os.path.join(dst_pixel_weights_dir,
                                                ground_truth_pair_name + "_{}_{:.2f}.bin".format("geodesic",
                                                                                                 node_coverage))

    assert np.array_equal(node_coords, dio.load_graph_nodes(gt_output_graph_nodes_path))
    assert np.array_equal(graph_edges, dio.load_graph_edges(gt_output_graph_edges_path))
    assert np.array_equal(graph_edges_weights, dio.load_graph_edges_weights(gt_output_graph_edges_weights_path))
    assert np.array_equal(graph_clusters, dio.load_graph_clusters(gt_output_graph_clusters_path))

    if node_deformations is not None:
        assert np.allclose(node_deformations, dio.load_graph_node_deformations(gt_output_node_deformations_path))
    if pixel_anchors is not None:
        assert np.array_equal(pixel_anchors, dio.load_int_image(gt_output_pixel_anchors_path))
    if pixel_weights is not None:
        assert np.array_equal(pixel_weights, dio.load_float_image(gt_output_pixel_weights_path))


PROGRAM_EXIT_SUCCESS = 0


def main():
    #########################################################################
    # Options
    #########################################################################
    VISUAL_DEBUGGING = False

    # Scene flow data is assumed to be only known at training time. To compute graphs for test time,
    # this should be set to false.
    USE_SCENE_FLOW_DATA = False

    # Depth-to-mesh conversion
    DEPTH_SCALE_RECIPROCAL = 1000.0
    MAX_TRIANGLE_DISTANCE = 0.05

    # Erosion of vertices in the boundaries
    EROSION_NUM_ITERATIONS = 4  # original authors' value: 10. 4 works better for the berlin sequence
    EROSION_MIN_NEIGHBORS = 4

    # Node sampling and edges computation
    USE_ONLY_VALID_VERTICES = True
    NEIGHBOR_COUNT = 8
    ENFORCE_NEIGHBOR_COUNT = False
    SAMPLE_RANDOM_SHUFFLE = False

    # Pixel anchors
    NEIGHBORHOOD_DEPTH = 2  # unused in code. Is this set as default parameter C++-side?

    MIN_CLUSTER_SIZE = 3  # unused in code. Is this set as default parameter C++-side?
    MIN_NUM_NEIGHBORS = 2  # unused in code. Is this set as default parameter C++-side?

    # Node clean-up
    REMOVE_NODES_WITH_TOO_FEW_NEIGHBORS = True

    #########################################################################
    # Paths.
    #########################################################################
    slice_name = "train"
    sequence_number = 70
    seq_dir = os.path.join(Parameters.path.dataset_base_directory.value, slice_name, f"seq{sequence_number:03d}")

    start_frame_number = 0
    end_frame_number = 100
    segment_name = "adult0"

    depth_image_path = os.path.join(seq_dir, "depth", f"{start_frame_number:06d}.png")
    mask_image_path = os.path.join(seq_dir, "mask", f"{start_frame_number:06d}_{segment_name:s}.png")
    scene_flow_path = \
        os.path.join(seq_dir, "scene_flow",
                     f"{segment_name:s}_{start_frame_number:06d}_{end_frame_number:06d}.sflow") if USE_SCENE_FLOW_DATA else None
    intrinsics_path = os.path.join(seq_dir, "intrinsics.txt")

    prefix = "generated"
    pair_name = f"{prefix:s}_{segment_name:s}_{start_frame_number:06d}_{end_frame_number:06d}"

    SAVE_GRAPH_DATA = True

    # enables/disables optional checks at end of script
    CHECK_AGAINST_GROUND_TRUTH = False
    # both prefixes can be set to the same value to simply check functions for the loading / saving of the graph
    ground_truth_prefix = "5c8446e47ef76a0addc6d0d1"
    ground_truth_pair_name = f"{ground_truth_prefix:s}_{segment_name:s}_{start_frame_number:06d}_{end_frame_number:06d}"

    #########################################################################
    # Load data.
    #########################################################################
    # Load intrinsics.
    intrinsic_matrix = np.loadtxt(intrinsics_path)

    # Load depth image.
    depth_image = skimage.io.imread(depth_image_path)

    # Load mask image.
    mask_image = skimage.io.imread(mask_image_path)

    graph, node_deformations, pixel_anchors, pixel_weights = \
        build_graph_warp_field_from_depth_image(
            depth_image, mask_image, o3d.core.Device("CPU:0"), intrinsic_matrix,
            MAX_TRIANGLE_DISTANCE, DEPTH_SCALE_RECIPROCAL, EROSION_NUM_ITERATIONS, EROSION_MIN_NEIGHBORS,
            REMOVE_NODES_WITH_TOO_FEW_NEIGHBORS, USE_ONLY_VALID_VERTICES, SAMPLE_RANDOM_SHUFFLE, NEIGHBOR_COUNT,
            ENFORCE_NEIGHBOR_COUNT, scene_flow_path, VISUAL_DEBUGGING
        )

    if SAVE_GRAPH_DATA:
        save_graph_data(seq_dir, pair_name, graph.nodes, graph.edges, graph.edge_weights, graph.clusters,
                        node_deformations, pixel_anchors, pixel_weights)

    if CHECK_AGAINST_GROUND_TRUTH:
        check_graph_data_against_ground_truth(seq_dir, ground_truth_pair_name,
                                              graph.nodes, graph.edges, graph.edge_weights, graph.clusters,
                                              node_deformations, pixel_anchors, pixel_weights)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
