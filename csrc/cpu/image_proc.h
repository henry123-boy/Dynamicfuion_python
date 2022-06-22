#pragma once

#include <iostream>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace image_proc {

    py::array_t<float> compute_augmented_flow_from_rotation(py::array_t<float>& flow_image_rot_sa2so,
                                                            py::array_t<float>& flow_image_so2to,
                                                            py::array_t<float>& flow_image_rot_to2ta,
                                                            const int height, const int width);


    void backproject_depth_ushort(py::array_t<unsigned short>& image_in, py::array_t<float>& point_image_out, float fx, float fy, float cx, float cy, float normalizer);
	py::array_t<float> backproject_depth_ushort(py::array_t<unsigned short>& image_in, float fx, float fy, float cx, float cy, float normalizer);

    void backproject_depth_float(py::array_t<float>& image_in, py::array_t<float>& point_image_out, float fx, float fy, float cx, float cy);

    void compute_mesh_from_depth(
		    const py::array_t<float>& point_image_in, float max_triangle_edge_distance,
		    py::array_t<float>& vertex_positions_out, py::array_t<int>& face_indices_out
    );

	void compute_mesh_from_depth(
			const py::array_t<float>& point_image_in, float max_triangle_edge_distance,
			py::array_t<float>& vertex_positions_out, py::array_t<int>& vertex_pixels_out, py::array_t<int>& face_indices_out
	);

	py::tuple compute_mesh_from_depth(
			const py::array_t<float>& point_image_in, float max_triangle_edge_distance
	);

    void compute_mesh_from_depth_and_color(
		    const py::array_t<float>& point_image, const py::array_t<int>& color_image, float max_triangle_edge_distance,
		    py::array_t<float>& vertex_positions, py::array_t<int>& vertex_colors, py::array_t<int>& face_indices
    );

    void compute_mesh_from_depth_and_flow(
		    const py::array_t<float>& point_image_in, const py::array_t<float>& flow_image_in, float max_triangle_edge_distance,
		    py::array_t<float>& vertex_positions_out, py::array_t<float>& vertex_flows_out, py::array_t<int>& vertex_pixels_out, py::array_t<int>& face_indices_out
    );

	py::tuple compute_mesh_from_depth_and_flow(
			const py::array_t<float>& point_image_in, const py::array_t<float>& flow_image_in, float max_triangle_edge_distance
	);

    void filter_depth(py::array_t<unsigned short>& depth_image_in, py::array_t<unsigned short>& depth_image_out, int radius);
	py::array_t<unsigned short> filter_depth(py::array_t<unsigned short>& depth_image_in, int radius);

    py::array_t<float> warp_flow(const py::array_t<float>& image, const py::array_t<float>& flow, const py::array_t<float>& mask);

    py::array_t<float> warp_rigid(const py::array_t<float>& rgbxyz_image, const py::array_t<float>& rotation, const py::array_t<float>& translation, float fx, float fy, float cx, float cy);

    py::array_t<float> warp_3d(const py::array_t<float>& rgbxyz_image, const py::array_t<float>& points, const py::array_t<int>& mask, float fx, float fy, float cx, float cy);

} // namespace image_proc