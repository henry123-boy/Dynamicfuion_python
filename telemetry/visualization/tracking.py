import typing

import numpy as np
import open3d as o3d

import telemetry.visualization.geometry.merge_meshes
import image_processing
from telemetry.visualization import example_viewer, line_mesh as line_mesh_utils
from telemetry.visualization.coordinate_transformations import transform_pointcloud_to_opengl_coords


def draw_node_graph(graph_nodes, graph_edges):
    # Graph canonical_node_positions
    rendered_graph_nodes = []
    for node in graph_nodes:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        mesh_sphere.translate(node)
        rendered_graph_nodes.append(mesh_sphere)

    # Merge all different sphere meshes
    rendered_graph_nodes = telemetry.visualization.geometry.merge_meshes.merge_meshes(rendered_graph_nodes)

    # Graph edges
    edges_pairs = []
    for node_id, edges in enumerate(graph_edges):
        for neighbor_id in edges:
            if neighbor_id == -1:
                break
            edges_pairs.append([node_id, neighbor_id])

    colors = [[0.2, 1.0, 0.2] for i in range(len(edges_pairs))]
    line_mesh = line_mesh_utils.LineMesh(graph_nodes, edges_pairs, colors, radius=0.003)
    line_mesh_geoms = line_mesh.cylinder_segments

    # Merge all different line meshes
    line_mesh_geoms = telemetry.visualization.geometry.merge_meshes.merge_meshes(line_mesh_geoms)

    # o3d.visualization.draw_geometries([rendered_graph_nodes, line_mesh_geoms, source_object_pcd])

    # Combined canonical_node_positions & edges
    rendered_graph = [rendered_graph_nodes, line_mesh_geoms]
    return rendered_graph


def visualize_tracking(
        source_rgbxyz: np.ndarray,
        target_rgbxyz: np.ndarray,
        pixel_anchors: np.ndarray,
        pixel_weights: np.ndarray,
        graph_nodes: np.ndarray,
        graph_edges: np.ndarray,
        rotations_pred: np.ndarray,
        translations_pred: np.ndarray,
        mask_pred: np.ndarray,
        valid_source_points: np.ndarray,
        valid_correspondences: np.ndarray,
        target_matches: np.ndarray):
    # Some params for coloring the predicted correspondence confidences
    weight_threshold = 0.3
    weight_scale = 1

    #####################################################################################################
    # region ======= Prepare data for visualization =====================================================
    #####################################################################################################
    #####################################################################################################
    # region >>>> Original Source Point Cloud <<<<
    #####################################################################################################
    source_flat = np.moveaxis(source_rgbxyz, 0, -1).reshape(-1, 6)
    source_points = transform_pointcloud_to_opengl_coords(source_flat[..., 3:])
    source_colors = source_flat[..., :3]

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    source_pcd.colors = o3d.utility.Vector3dVector(source_colors)

    # keep only object using the mask
    valid_source_mask = np.moveaxis(valid_source_points, 0, -1).reshape(-1).astype(bool)
    valid_source_points = source_points[valid_source_mask, :]
    valid_source_colors = source_colors[valid_source_mask, :]
    # source object PointCloud
    source_object_pcd = o3d.geometry.PointCloud()
    source_object_pcd.points = o3d.utility.Vector3dVector(valid_source_points)
    source_object_pcd.colors = o3d.utility.Vector3dVector(valid_source_colors)

    # endregion
    #####################################################################################################
    # region >>>> Warped Source Point Cloud<<<<
    #####################################################################################################
    warped_deform_pred_3d_np = image_processing.warp_deform_3d(
        source_rgbxyz, pixel_anchors, pixel_weights, graph_nodes, rotations_pred, translations_pred
    )

    source_warped = np.copy(source_rgbxyz)
    source_warped[3:, :, :] = warped_deform_pred_3d_np

    # (source) warped RGB-D image
    source_warped = np.moveaxis(source_warped, 0, -1).reshape(-1, 6)
    warped_points = transform_pointcloud_to_opengl_coords(source_warped[..., 3:])
    warped_colors = source_warped[..., :3]
    # Filter points at (0, 0, 0)
    warped_points = warped_points[valid_source_mask]
    warped_colors = warped_colors[valid_source_mask]
    # warped PointCloud
    warped_pcd = o3d.geometry.PointCloud()
    warped_pcd.points = o3d.utility.Vector3dVector(warped_points)
    warped_pcd.paint_uniform_color([1, 0.706, 0])  # warped_pcd.colors = o3d.utility.Vector3dVector(warped_colors)
    # endregion
    ####################################
    # region >>>> Target Point Cloud <<<<
    ####################################
    # target RGB-D image
    target_flat = np.moveaxis(target_rgbxyz, 0, -1).reshape(-1, 6)
    target_points = transform_pointcloud_to_opengl_coords(target_flat[..., 3:])
    target_colors = target_flat[..., :3]
    # target PointCloud
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    target_pcd.colors = o3d.utility.Vector3dVector(target_colors)
    # endregion
    ####################################
    # region >>>> Graph <<<<
    ####################################

    deformed_graph_nodes = graph_nodes + translations_pred

    # Transform to OpenGL coords
    graph_nodes = transform_pointcloud_to_opengl_coords(graph_nodes)
    deformed_graph_nodes = transform_pointcloud_to_opengl_coords(deformed_graph_nodes)

    source_graph = draw_node_graph(graph_nodes, graph_edges)
    target_graph = draw_node_graph(deformed_graph_nodes, graph_edges)

    # endregion
    ####################################
    # Mask
    ####################################
    mask_pred_flat = mask_pred.reshape(-1)
    valid_correspondences = valid_correspondences.reshape(-1).astype(bool)

    ####################################
    # Correspondences
    ####################################
    # target matches
    target_matches = np.moveaxis(target_matches, 0, -1).reshape(-1, 3)
    target_matches = transform_pointcloud_to_opengl_coords(target_matches)

    ################################
    # "Good" matches
    ################################
    good_mask = valid_correspondences & (mask_pred_flat >= weight_threshold)
    good_source_points_corresp = source_points[good_mask]
    good_target_matches_corresp = target_matches[good_mask]
    good_mask_pred = mask_pred_flat[good_mask]

    # number of good matches
    n_good_matches = good_source_points_corresp.shape[0]
    # Subsample
    subsample = True
    if subsample:
        N = 2000
        sampled_idxs = np.random.permutation(n_good_matches)[:N]
        good_source_points_corresp = good_source_points_corresp[sampled_idxs]
        good_target_matches_corresp = good_target_matches_corresp[sampled_idxs]
        good_mask_pred = good_mask_pred[sampled_idxs]
        n_good_matches = N

    # both good_source and good_target points together into one vector
    good_matches_points = np.concatenate([good_source_points_corresp, good_target_matches_corresp], axis=0)
    good_matches_lines = [[i, i + n_good_matches] for i in range(0, n_good_matches, 1)]

    # --> Create good (unweighted) lines
    good_matches_colors = [[201 / 255, 177 / 255, 14 / 255] for i in range(len(good_matches_lines))]
    good_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(good_matches_points),
        lines=o3d.utility.Vector2iVector(good_matches_lines),
    )
    good_matches_set.colors = o3d.utility.Vector3dVector(good_matches_colors)

    # --> Create good weighted lines
    # first, we need to get the proper color coding
    high_color, low_color = np.array([0.0, 0.8, 0]), np.array([0.8, 0, 0.0])

    good_weighted_matches_colors = np.ones_like(good_source_points_corresp)

    weights_normalized = np.maximum(np.minimum(0.5 + (good_mask_pred - weight_threshold) / weight_scale, 1.0), 0.0)
    weights_normalized_opposite = 1 - weights_normalized

    good_weighted_matches_colors[:, 0] = weights_normalized * high_color[0] + weights_normalized_opposite * low_color[0]
    good_weighted_matches_colors[:, 1] = weights_normalized * high_color[1] + weights_normalized_opposite * low_color[1]
    good_weighted_matches_colors[:, 2] = weights_normalized * high_color[2] + weights_normalized_opposite * low_color[2]

    good_weighted_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(good_matches_points),
        lines=o3d.utility.Vector2iVector(good_matches_lines),
    )
    good_weighted_matches_set.colors = o3d.utility.Vector3dVector(good_weighted_matches_colors)

    ################################
    # "Bad" matches
    ################################
    bad_mask = valid_correspondences & (mask_pred_flat < weight_threshold)
    bad_source_points_corresp = source_points[bad_mask]
    bad_target_matches_corresp = target_matches[bad_mask]
    bad_mask_pred = mask_pred_flat[bad_mask]

    # number of good matches
    n_bad_matches = bad_source_points_corresp.shape[0]

    # both good_source and good_target points together into one vector
    bad_matches_points = np.concatenate([bad_source_points_corresp, bad_target_matches_corresp], axis=0)
    bad_matches_lines = [[i, i + n_bad_matches] for i in range(0, n_bad_matches, 1)]

    # --> Create bad (unweighted) lines
    bad_matches_colors = [[201 / 255, 177 / 255, 14 / 255] for i in range(len(bad_matches_lines))]
    bad_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bad_matches_points),
        lines=o3d.utility.Vector2iVector(bad_matches_lines),
    )
    bad_matches_set.colors = o3d.utility.Vector3dVector(bad_matches_colors)

    # --> Create bad weighted lines
    # first, we need to get the proper color coding
    high_color, low_color = np.array([0.0, 0.8, 0]), np.array([0.8, 0, 0.0])

    bad_weighted_matches_colors = np.ones_like(bad_source_points_corresp)

    weights_normalized = np.maximum(np.minimum(0.5 + (bad_mask_pred - weight_threshold) / weight_scale, 1.0), 0.0)
    weights_normalized_opposite = 1 - weights_normalized

    bad_weighted_matches_colors[:, 0] = weights_normalized * high_color[0] + weights_normalized_opposite * low_color[0]
    bad_weighted_matches_colors[:, 1] = weights_normalized * high_color[1] + weights_normalized_opposite * low_color[1]
    bad_weighted_matches_colors[:, 2] = weights_normalized * high_color[2] + weights_normalized_opposite * low_color[2]

    bad_weighted_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bad_matches_points),
        lines=o3d.utility.Vector2iVector(bad_matches_lines),
    )
    bad_weighted_matches_set.colors = o3d.utility.Vector3dVector(bad_weighted_matches_colors)

    ####################################
    # Generate info for aligning source to target (by interpolating between source and warped source)
    ####################################
    assert warped_points.shape[0] == valid_source_points.shape[0]
    line_segments = warped_points - valid_source_points
    line_segments_unit, line_lengths = line_mesh_utils.normalized(line_segments)
    line_lengths = line_lengths[:, np.newaxis]
    line_lengths = np.repeat(line_lengths, 3, axis=1)

    # endregion
    ############################################################################
    # region ======= Draw / Launch Visualization ===============================
    ############################################################################
    geometry_dict = {
        "source_pcd": source_pcd,
        "source_obj": source_object_pcd,
        "target_pcd": target_pcd,
        "source_graph": source_graph,
        "target_graph": target_graph,
        "warped_pcd": warped_pcd
    }

    alignment_dict = {
        "valid_source_points": valid_source_points,
        "line_segments_unit": line_segments_unit,
        "line_lengths": line_lengths
    }

    matches_dict = {
        "good_matches_set": good_matches_set,
        "good_weighted_matches_set": good_weighted_matches_set,
        "bad_matches_set": bad_matches_set,
        "bad_weighted_matches_set": bad_weighted_matches_set
    }

    #####################################################################################################
    # Open viewer
    #####################################################################################################
    viewer = example_viewer.CustomDrawGeometryWithKeyCallbackViewer(
        geometry_dict, alignment_dict, matches_dict
    )
    viewer.custom_draw_geometry_with_key_callback()
    # endregion
