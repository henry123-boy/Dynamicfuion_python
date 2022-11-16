#include "cpu/image_proc.h"

#include <Eigen/Dense>
#include <map>
#include <string>
#include <cmath>
#include <stdexcept>

namespace helper {

bool in_bounds(Eigen::Vector2f p, int h, int w) {
	return p.x() >= 0.0 && p.x() < static_cast<float>(w) && p.y() >= 0.0 && p.y() < static_cast<float>(h);
}

bool valid_flow_at(const Eigen::Vector2f& p, int h, int w, py::array_t<float>& flow_image, Eigen::Vector2f& flow) {
	if (!in_bounds(p, h, w)) {
		return false;
	}

	flow.x() = *flow_image.data(p.y(), p.x(), 0);
	flow.y() = *flow_image.data(p.y(), p.x(), 1);

	if (!flow.allFinite()) {
		return false;
	}

	return true;
}
} // namespace helper

namespace image_proc {

using Vec2f = Eigen::Vector2f;

template<class V, class L = std::less<std::string>,
		class A = Eigen::aligned_allocator<std::pair<const std::string, V>>>
using aligned_dict = std::map<std::string, V, L, A>;

py::array_t<float> compute_augmented_flow_from_rotation(py::array_t<float>& flow_image_rot_sa2so,
                                                        py::array_t<float>& flow_image_so2to,
                                                        py::array_t<float>& flow_image_rot_to2ta,
                                                        const int height, const int width) {
	// TODO: change to runtime asserts
	// assert(flow_image_rot_sa2so.ndim() == 3);
	// assert(flow_image_rot_sa2so.shape(0) == 2);
	// assert(flow_image_rot_sa2so.shape(1) == height);
	// assert(flow_image_rot_sa2so.shape(2) == width);

	// assert(flow_image_so2to.ndim() == 3);
	// assert(flow_image_so2to.shape(0) == 2);
	// assert(flow_image_so2to.shape(1) == height);
	// assert(flow_image_so2to.shape(2) == width);

	// assert(flow_image_rot_to2ta.ndim() == 3);
	// assert(flow_image_rot_to2ta.shape(0) == 2);
	// assert(flow_image_rot_to2ta.shape(1) == height);
	// assert(flow_image_rot_to2ta.shape(2) == width);

	// allocate memory for output array
	py::array_t<float> flow_image_rot_sa2ta = py::array_t<float>(flow_image_rot_sa2so.request().size);

	// reshape array to match input shape
	flow_image_rot_sa2ta.resize({height, width, 2});

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			// update output flow image
			*flow_image_rot_sa2ta.mutable_data(y, x, 0) = -std::numeric_limits<float>::infinity();
			*flow_image_rot_sa2ta.mutable_data(y, x, 1) = -std::numeric_limits<float>::infinity();

			Vec2f p_sa(x, y);

			/////////////////////////////////////////////////////////////////////////////////
			// 1. SOURCE AUGMENTED TO SOURCE ORIGINAL
			/////////////////////////////////////////////////////////////////////////////////

			// flow from source augmented to source original
			Vec2f flow_sa2so(*flow_image_rot_sa2so.data(y, x, 0), *flow_image_rot_sa2so.data(y, x, 1));

			// flow_sa2so should be dense and w/o any invalid value
			if (!flow_sa2so.allFinite()) {
				throw std::runtime_error("flow_sa2so should be dense and w/o any invalid residuals!");
			}

			// compute warped location on source original (so we're going from source augmented to source original)
			Vec2f p_so = p_sa + flow_sa2so;

			// init flow_sa2ta with the first contribution, i.e, flow_sa2so
			Vec2f flow_sa2ta = flow_sa2so;

			/////////////////////////////////////////////////////////////////////////////////
			// 2. SOURCE ORIGINAL TO TARGET ORIGINAL
			/////////////////////////////////////////////////////////////////////////////////
			int u0 = std::floor(p_so.x());
			int u1 = u0 + 1;
			int v0 = std::floor(p_so.y());
			int v1 = v0 + 1;

			Vec2f p00(u0, v0);
			Vec2f p01(u0, v1);
			Vec2f p10(u1, v0);
			Vec2f p11(u1, v1);

			aligned_dict<Vec2f> valid_coords;
			aligned_dict<Vec2f> valid_flows;

			Vec2f flow_00_so2to;
			if (helper::valid_flow_at(p00, height, width, flow_image_so2to, flow_00_so2to)) {
				valid_coords["p00"] = p00;
				valid_flows["p00"] = flow_00_so2to;
			}

			Vec2f flow_01_so2to;
			if (helper::valid_flow_at(p01, height, width, flow_image_so2to, flow_01_so2to)) {
				valid_coords["p01"] = p01;
				valid_flows["p01"] = flow_01_so2to;
			}

			Vec2f flow_10_so2to;
			if (helper::valid_flow_at(p10, height, width, flow_image_so2to, flow_10_so2to)) {
				valid_coords["p10"] = p10;
				valid_flows["p10"] = flow_10_so2to;
			}

			Vec2f flow_11_so2to;
			if (helper::valid_flow_at(p11, height, width, flow_image_so2to, flow_11_so2to)) {
				valid_coords["p11"] = p11;
				valid_flows["p11"] = flow_11_so2to;
			}

			// Depending on how many valid flows we have, do bilinear interpolation or nearest neighbor:
			Vec2f flow_so2to;

			if (valid_coords.empty()) {
				continue;
			} else if (valid_coords.size() == 4) {
				// Bilinear interpolation
				float du = p_so.x() - static_cast<float>(u0);
				float dv = p_so.y() - static_cast<float>(v0);

				float w00 = (1 - du) * (1 - dv);
				float w01 = (1 - du) * dv;
				float w10 = du * (1 - dv);
				float w11 = du * dv;

				flow_so2to = w00 * valid_flows["p00"] +
				             w01 * valid_flows["p01"] +
				             w10 * valid_flows["p10"] +
				             w11 * valid_flows["p11"];
			} else {
				// Nearest Neighbor
				std::string nn = "None";
				float min_dist = std::numeric_limits<float>::max();

				for (const auto& valid_coord : valid_coords) {
					const std::string k = valid_coord.first;
					const Vec2f& p = valid_coord.second;

					float dist = (p_so - p).norm();
					if (dist < min_dist) {
						min_dist = dist;
						nn = k;
					}
				}

				if (nn == "None") {
					throw std::runtime_error("Neighrest Neighbor 'nn' was not assigned...");
				}

				flow_so2to = valid_flows[nn];
			}

			// compute warped location on target original (so we're going from source original to target original)
			Vec2f p_to = p_so + flow_so2to;

			// add flow_so2to to flow_sa2ta
			flow_sa2ta += flow_so2to;

			/////////////////////////////////////////////////////////////////////////////////
			// 3. TARGET ORIGINAL TO TARGET AUGMENTED
			/////////////////////////////////////////////////////////////////////////////////
			u0 = std::floor(p_to.x());
			u1 = u0 + 1;
			v0 = std::floor(p_to.y());
			v1 = v0 + 1;

			p00 = Vec2f(u0, v0);
			p01 = Vec2f(u0, v1);
			p10 = Vec2f(u1, v0);
			p11 = Vec2f(u1, v1);

			valid_coords.clear();
			valid_flows.clear();

			Vec2f flow_00_to2ta;
			if (helper::valid_flow_at(p00, height, width, flow_image_rot_to2ta, flow_00_to2ta)) {
				valid_coords["p00"] = p00;
				valid_flows["p00"] = flow_00_to2ta;
			}

			Vec2f flow_01_to2ta;
			if (helper::valid_flow_at(p01, height, width, flow_image_rot_to2ta, flow_01_to2ta)) {
				valid_coords["p01"] = p01;
				valid_flows["p01"] = flow_01_to2ta;
			}

			Vec2f flow_10_to2ta;
			if (helper::valid_flow_at(p10, height, width, flow_image_rot_to2ta, flow_10_to2ta)) {
				valid_coords["p10"] = p10;
				valid_flows["p10"] = flow_10_to2ta;
			}

			Vec2f flow_11_to2ta;
			if (helper::valid_flow_at(p11, height, width, flow_image_rot_to2ta, flow_11_to2ta)) {
				valid_coords["p11"] = p11;
				valid_flows["p11"] = flow_11_to2ta;
			}

			// Depending on how many valid flows we have, do bilinear interpolation or nearest neighbor:
			Vec2f flow_to2ta;

			if (valid_coords.empty()) {
				continue;
			} else if (valid_coords.size() == 4) {
				// Bilinear interpolation
				float du = p_to.x() - static_cast<float>(u0);
				float dv = p_to.y() - static_cast<float>(v0);

				float w00 = (1 - du) * (1 - dv);
				float w01 = (1 - du) * dv;
				float w10 = du * (1 - dv);
				float w11 = du * dv;

				flow_to2ta = w00 * valid_flows["p00"] +
				             w01 * valid_flows["p01"] +
				             w10 * valid_flows["p10"] +
				             w11 * valid_flows["p11"];
			} else {
				// Nearest Neighbor
				std::string nn = "None";
				float min_dist = std::numeric_limits<float>::max();

				for (const auto& valid_coord : valid_coords) {
					const std::string k = valid_coord.first;
					const Vec2f& p = valid_coord.second;

					float dist = (p_to - p).norm();
					if (dist < min_dist) {
						min_dist = dist;
						nn = k;
					}
				}

				if (nn == "None") {
					throw std::runtime_error("Neighrest Neighbor 'nn' was not assigned...");
				}

				flow_to2ta = valid_flows[nn];
			}

			// add flow_to2ta to flow_sa2ta
			flow_sa2ta += flow_to2ta;

			// update output flow image
			*flow_image_rot_sa2ta.mutable_data(y, x, 0) = flow_sa2ta.x();
			*flow_image_rot_sa2ta.mutable_data(y, x, 1) = flow_sa2ta.y();
		}
	}

	return flow_image_rot_sa2ta;
}


void backproject_depth_ushort(py::array_t<unsigned short>& image_in, py::array_t<float>& point_image_out,
                              const float fx, const float fy, const float cx, const float cy, const float normalizer) {
	assert(image_in.ndim() == 2);
	assert(point_image_out.ndim() == 3);

	const int width = image_in.shape(1);
	const int height = image_in.shape(0);

	assert(point_image_out.shape(0) == height);
	assert(point_image_out.shape(1) == width);
	assert(point_image_out.shape(2) == 3);

#pragma omp parallel for default(none) shared(image_in, point_image_out) firstprivate(height, width, fx, fy, cx, cy, normalizer)
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float depth = float(*image_in.data(y, x)) / normalizer;

			if (depth > 0) {
				float pos_x = depth * (static_cast<float>(x) - cx) / fx;
				float pos_y = depth * (static_cast<float>(y) - cy) / fy;
				float pos_z = depth;

				*point_image_out.mutable_data(y, x, 0) = pos_x;
				*point_image_out.mutable_data(y, x, 1) = pos_y;
				*point_image_out.mutable_data(y, x, 2) = pos_z;
			}
		}
	}
}

py::array_t<float> backproject_depth_ushort(py::array_t<unsigned short>& image_in, float fx, float fy, float cx, float cy, float normalizer) {
	py::array_t<float> point_image_out({image_in.shape(0), image_in.shape(1), static_cast<ssize_t>(3)});
	memset(point_image_out.mutable_data(0, 0, 0), 0, point_image_out.size() * sizeof(float));
	backproject_depth_ushort(image_in, point_image_out, fx, fy, cx, cy, normalizer);
	return point_image_out;
}

void backproject_depth_float(py::array_t<float>& image_in, py::array_t<float>& point_image_out,
                             float fx, float fy, float cx, float cy) {
	assert(image_in.ndim() == 2);
	assert(point_image_out.ndim() == 3);

	int width = image_in.shape(1);
	int height = image_in.shape(0);
	assert(point_image_out.shape(0) == 3);
	assert(point_image_out.shape(1) == height);
	assert(point_image_out.shape(2) == width);

#pragma omp parallel for default(none) shared(image_in, point_image_out) firstprivate(height, width, fx, fy, cx, cy)
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float depth = *image_in.data(y, x);

			if (depth > 0) {
				float pos_x = depth * (static_cast<float>(x) - cx) / fx;
				float pos_y = depth * (static_cast<float>(y) - cy) / fy;
				float pos_z = depth;

				*point_image_out.mutable_data(y, x, 0) = pos_x;
				*point_image_out.mutable_data(y, x, 1) = pos_y;
				*point_image_out.mutable_data(y, x, 2) = pos_z;
			}
		}
	}
}

void compute_mesh_from_depth(const py::array_t<float>& point_image_in, float max_triangle_edge_distance,
                             py::array_t<float>& vertex_positions_out, py::array_t<int>& vertex_pixels_out,
                             py::array_t<int>& face_indices_out) {
	assert(point_image_in.ndim() == 3);
	assert(point_image_in.shape(2) == 3);

	int width = static_cast<int>(point_image_in.shape(1));
	int height = static_cast<int>(point_image_in.shape(0));

	// Compute valid pixel vertices and faces.
	// We also need to compute the pixel -> vertex index mapping for
	// computation of faces.
	// We connect neighboring pixels on the square into two triangles.
	// We only select valid triangles, i.e. with all valid vertices and
	// not too far apart.
	// Important: The triangle orientation is set such that the normals
	// point towards the camera.
	std::vector<Eigen::Vector3f> vertices;
	std::vector<Eigen::Vector3i> faces;
	std::vector<Eigen::Vector2i> pixels;

	int vertexIdx = 0;
	std::vector<int> mapPixelToVertexIdx(width * height, -1);

	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			Eigen::Vector3f obs00(*point_image_in.data(y + 0, x + 0, 0), *point_image_in.data(y + 0, x + 0, 1), *point_image_in.data(y + 0, x + 0, 2));
			Eigen::Vector3f obs01(*point_image_in.data(y + 1, x + 0, 0), *point_image_in.data(y + 1, x + 0, 1), *point_image_in.data(y + 1, x + 0, 2));
			Eigen::Vector3f obs10(*point_image_in.data(y + 0, x + 1, 0), *point_image_in.data(y + 0, x + 1, 1), *point_image_in.data(y + 0, x + 1, 2));
			Eigen::Vector3f obs11(*point_image_in.data(y + 1, x + 1, 0), *point_image_in.data(y + 1, x + 1, 1), *point_image_in.data(y + 1, x + 1, 2));

			int idx00 = y * width + x;
			int idx01 = (y + 1) * width + x;
			int idx10 = y * width + (x + 1);
			int idx11 = (y + 1) * width + (x + 1);

			bool valid00 = obs00.z() > 0;
			bool valid01 = obs01.z() > 0;
			bool valid10 = obs10.z() > 0;
			bool valid11 = obs11.z() > 0;

			if (valid00 && valid01 && valid10) {
				float d0 = (obs00 - obs01).norm();
				float d1 = (obs00 - obs10).norm();
				float d2 = (obs01 - obs10).norm();

				if (d0 <= max_triangle_edge_distance && d1 <= max_triangle_edge_distance && d2 <= max_triangle_edge_distance) {
					int vIdx0 = mapPixelToVertexIdx[idx00];
					int vIdx1 = mapPixelToVertexIdx[idx01];
					int vIdx2 = mapPixelToVertexIdx[idx10];

					if (vIdx0 == -1) {
						vIdx0 = vertexIdx;
						mapPixelToVertexIdx[idx00] = vertexIdx;
						vertices.push_back(obs00);
						vertexIdx++;
						pixels.emplace_back(x, y);
					}
					if (vIdx1 == -1) {
						vIdx1 = vertexIdx;
						mapPixelToVertexIdx[idx01] = vertexIdx;
						vertices.push_back(obs01);
						vertexIdx++;
						pixels.emplace_back(x, y + 1);
					}
					if (vIdx2 == -1) {
						vIdx2 = vertexIdx;
						mapPixelToVertexIdx[idx10] = vertexIdx;
						vertices.push_back(obs10);
						vertexIdx++;
						pixels.emplace_back(x + 1, y);
					}

					faces.emplace_back(vIdx0, vIdx1, vIdx2);
				}
			}

			if (valid01 && valid10 && valid11) {
				float d0 = (obs10 - obs01).norm();
				float d1 = (obs10 - obs11).norm();
				float d2 = (obs01 - obs11).norm();

				if (d0 <= max_triangle_edge_distance && d1 <= max_triangle_edge_distance && d2 <= max_triangle_edge_distance) {
					int vIdx0 = mapPixelToVertexIdx[idx11];
					int vIdx1 = mapPixelToVertexIdx[idx10];
					int vIdx2 = mapPixelToVertexIdx[idx01];

					if (vIdx0 == -1) {
						vIdx0 = vertexIdx;
						mapPixelToVertexIdx[idx11] = vertexIdx;
						vertices.push_back(obs11);
						vertexIdx++;
						pixels.emplace_back(x + 1, y + 1);
					}
					if (vIdx1 == -1) {
						vIdx1 = vertexIdx;
						mapPixelToVertexIdx[idx10] = vertexIdx;
						vertices.push_back(obs10);
						vertexIdx++;
						pixels.emplace_back(x + 1, y);
					}
					if (vIdx2 == -1) {
						vIdx2 = vertexIdx;
						mapPixelToVertexIdx[idx01] = vertexIdx;
						vertices.push_back(obs01);
						vertexIdx++;
						pixels.emplace_back(x, y + 1);
					}

					faces.emplace_back(vIdx0, vIdx1, vIdx2);
				}
			}
		}
	}

	// Convert to numpy array.
	int vertex_count = vertices.size();
	int face_count = faces.size();

	if (vertex_count > 0 && face_count > 0) {
		// Reference check should be set to false otherwise there is a runtime
		// error. Check why that is the case.
		vertex_positions_out.resize({vertex_count, 3}, false);
		face_indices_out.resize({face_count, 3}, false);
		vertex_pixels_out.resize({vertex_count, 2}, false);

		for (int i = 0; i < vertex_count; i++) {
			*vertex_positions_out.mutable_data(i, 0) = vertices[i].x();
			*vertex_positions_out.mutable_data(i, 1) = vertices[i].y();
			*vertex_positions_out.mutable_data(i, 2) = vertices[i].z();

			*vertex_pixels_out.mutable_data(i, 0) = pixels[i].x();
			*vertex_pixels_out.mutable_data(i, 1) = pixels[i].y();
		}

		for (int i = 0; i < face_count; i++) {
			*face_indices_out.mutable_data(i, 0) = faces[i].x();
			*face_indices_out.mutable_data(i, 1) = faces[i].y();
			*face_indices_out.mutable_data(i, 2) = faces[i].z();
		}
	}
}


void compute_mesh_from_depth(
		const py::array_t<float>& point_image_in, float max_triangle_edge_distance,
		py::array_t<float>& vertex_positions_out, py::array_t<int>& face_indices_out
) {
	py::array_t<int> vertex_pixels;
	compute_mesh_from_depth(point_image_in, max_triangle_edge_distance, vertex_positions_out, vertex_pixels, face_indices_out);
}

py::tuple compute_mesh_from_depth(const py::array_t<float>& point_image_in, float max_triangle_edge_distance) {
	py::array_t<float> vertex_positions_out;
	py::array_t<int> face_indices_out;
	py::array_t<int> vertex_pixels_out;
	compute_mesh_from_depth(point_image_in, max_triangle_edge_distance, vertex_positions_out, vertex_pixels_out, face_indices_out);
	return py::make_tuple(vertex_positions_out, vertex_pixels_out, face_indices_out);
}

void compute_mesh_from_depth_and_color(
		const py::array_t<float>& point_image, const py::array_t<int>& color_image, float max_triangle_edge_distance,
		py::array_t<float>& vertex_positions, py::array_t<int>& vertex_colors, py::array_t<int>& face_indices
) {
	int width = static_cast<int>(point_image.shape(1));
	int height = static_cast<int>(point_image.shape(0));

	// Compute valid pixel vertices and faces.
	// We also need to compute the pixel -> vertex index mapping for
	// computation of faces.
	// We connect neighboring pixels on the square into two triangles.
	// We only select valid triangles, i.e. with all valid vertices and
	// not too far apart.
	// Important: The triangle orientation is set such that the normals
	// point towards the camera.
	std::vector<Eigen::Vector3f> vertices;
	std::vector<Eigen::Vector3i> colors;
	std::vector<Eigen::Vector3i> faces;

	int vertexIdx = 0;
	std::vector<int> mapPixelToVertexIdx(width * height, -1);

	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			Eigen::Vector3f obs00(*point_image.data(y + 0, x + 0, 0), *point_image.data(y + 0, x + 0, 1), *point_image.data(y + 0, x + 0, 2));
			Eigen::Vector3f obs01(*point_image.data(y + 1, x + 0, 0), *point_image.data(y + 1, x + 0, 1), *point_image.data(y + 1, x + 0, 2));
			Eigen::Vector3f obs10(*point_image.data(y + 0, x + 1, 0), *point_image.data(y + 0, x + 1, 1), *point_image.data(y + 0, x + 1, 2));
			Eigen::Vector3f obs11(*point_image.data(y + 1, x + 1, 0), *point_image.data(y + 1, x + 1, 1), *point_image.data(y + 1, x + 1, 2));

			Eigen::Vector3i color00(*color_image.data(y + 0, x + 0, 0), *color_image.data(y + 0, x + 0, 1), *color_image.data(y + 0, x + 0, 2));
			Eigen::Vector3i color01(*color_image.data(y + 1, x + 0, 0), *color_image.data(y + 1, x + 0, 1), *color_image.data(y + 1, x + 0, 2));
			Eigen::Vector3i color10(*color_image.data(y + 0, x + 1, 0), *color_image.data(y + 0, x + 1, 1), *color_image.data(y + 0, x + 1, 2));
			Eigen::Vector3i color11(*color_image.data(y + 1, x + 1, 0), *color_image.data(y + 1, x + 1, 1), *color_image.data(y + 1, x + 1, 2));

			// find linear indices
			int idx00 = y * width + x;
			int idx01 = (y + 1) * width + x;
			int idx10 = y * width + (x + 1);
			int idx11 = (y + 1) * width + (x + 1);

			bool valid00 = obs00.z() > 0;
			bool valid01 = obs01.z() > 0;
			bool valid10 = obs10.z() > 0;
			bool valid11 = obs11.z() > 0;

			// region ======= LOWER LEFT TRIANGLE =======
			if (valid00 && valid01 && valid10) {
				float d0 = (obs00 - obs01).norm();
				float d1 = (obs00 - obs10).norm();
				float d2 = (obs01 - obs10).norm();

				if (d0 <= max_triangle_edge_distance && d1 <= max_triangle_edge_distance && d2 <= max_triangle_edge_distance) {
					int vIdx0 = mapPixelToVertexIdx[idx00];
					int vIdx1 = mapPixelToVertexIdx[idx01];
					int vIdx2 = mapPixelToVertexIdx[idx10];

					if (vIdx0 == -1) {
						vIdx0 = vertexIdx;
						mapPixelToVertexIdx[idx00] = vertexIdx;
						vertices.push_back(obs00);
						colors.push_back(color00);
						vertexIdx++;
					}
					if (vIdx1 == -1) {
						vIdx1 = vertexIdx;
						mapPixelToVertexIdx[idx01] = vertexIdx;
						vertices.push_back(obs01);
						colors.push_back(color01);
						vertexIdx++;
					}
					if (vIdx2 == -1) {
						vIdx2 = vertexIdx;
						mapPixelToVertexIdx[idx10] = vertexIdx;
						vertices.push_back(obs10);
						colors.push_back(color10);
						vertexIdx++;
					}

					faces.emplace_back(vIdx0, vIdx1, vIdx2);
				}
			}
			// endregion
			// region ======= UPPER RIGHT TRIANGLE =======
			if (valid01 && valid10 && valid11) {
				float d0 = (obs10 - obs01).norm();
				float d1 = (obs10 - obs11).norm();
				float d2 = (obs01 - obs11).norm();

				if (d0 <= max_triangle_edge_distance && d1 <= max_triangle_edge_distance && d2 <= max_triangle_edge_distance) {
					int vIdx0 = mapPixelToVertexIdx[idx11];
					int vIdx1 = mapPixelToVertexIdx[idx10];
					int vIdx2 = mapPixelToVertexIdx[idx01];

					if (vIdx0 == -1) {
						vIdx0 = vertexIdx;
						mapPixelToVertexIdx[idx11] = vertexIdx;
						vertices.push_back(obs11);
						colors.push_back(color11);
						vertexIdx++;
					}
					if (vIdx1 == -1) {
						vIdx1 = vertexIdx;
						mapPixelToVertexIdx[idx10] = vertexIdx;
						vertices.push_back(obs10);
						colors.push_back(color10);
						vertexIdx++;
					}
					if (vIdx2 == -1) {
						vIdx2 = vertexIdx;
						mapPixelToVertexIdx[idx01] = vertexIdx;
						vertices.push_back(obs01);
						colors.push_back(color01);
						vertexIdx++;
					}

					faces.emplace_back(vIdx0, vIdx1, vIdx2);
				}
			}
			// endregion
		}
	}

	// Convert to numpy array.
	int vertex_count = vertices.size();
	int face_count = faces.size();

	if (vertex_count > 0 && face_count > 0) {
		// Reference check should be set to false otherwise there is a runtime
		// error. Check why that is the case.
		vertex_positions.resize({vertex_count, 3}, false);
		vertex_colors.resize({vertex_count, 3}, false);
		face_indices.resize({face_count, 3}, false);

		for (int i = 0; i < vertex_count; i++) {
			*vertex_positions.mutable_data(i, 0) = vertices[i].x();
			*vertex_positions.mutable_data(i, 1) = vertices[i].y();
			*vertex_positions.mutable_data(i, 2) = vertices[i].z();

			*vertex_colors.mutable_data(i, 0) = colors[i].x();
			*vertex_colors.mutable_data(i, 1) = colors[i].y();
			*vertex_colors.mutable_data(i, 2) = colors[i].z();
		}

		for (int i = 0; i < face_count; i++) {
			*face_indices.mutable_data(i, 0) = faces[i].x();
			*face_indices.mutable_data(i, 1) = faces[i].y();
			*face_indices.mutable_data(i, 2) = faces[i].z();
		}
	}
}


void compute_mesh_from_depth_and_flow(
		const py::array_t<float>& point_image_in, const py::array_t<float>& flow_image_in,
		float max_triangle_edge_distance, py::array_t<float>& vertex_positions_out,
		py::array_t<float>& vertex_flows_out, py::array_t<int>& vertex_pixels_out, py::array_t<int>& face_indices_out
) {
	int width = point_image_in.shape(1);
	int height = point_image_in.shape(0);


	// Compute valid pixel vertices and faces
	// a pixel is considered valid if the corresponding point's z coordinate is greater than zero and x, y, and z of the flow vector are all finite

	// We also need to compute the pixel -> vertex index mapping for computation of faces.
	// TODO: why are there no vertex indices > 400, if the total image has 640x480=307200 pixels?

	// For every 2x2 pixel area, we connect neighboring pixels into two triangles.
	// [ ]-------[ ]
	//  | ╲       |
	//  |    ╲    |
	//  |       ╲ |
	// [ ]-------[ ]

	// We only select valid triangles, i.e. with all valid vertices, and vertices not too far apart.

	// Important: The triangle orientation is set such that the normals point towards the camera.

	std::vector<Eigen::Vector3f> vertices;
	std::vector<Eigen::Vector3f> flows;
	std::vector<Eigen::Vector2i> pixels;
	std::vector<Eigen::Vector3i> faces;

	int vertex_index = 0;
	std::vector<int> pixel_to_vertex_index_map(width * height, -1);

	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			Eigen::Vector3f obs00(*point_image_in.data(y, x, 0), *point_image_in.data(y, x, 1), *point_image_in.data(y, x, 2));
			Eigen::Vector3f obs01(*point_image_in.data(y + 1, x, 0), *point_image_in.data(y + 1, x, 1), *point_image_in.data(y + 1, x, 2));
			Eigen::Vector3f obs10(*point_image_in.data(y, x + 1, 0), *point_image_in.data(y, x + 1, 1), *point_image_in.data(y, x + 1, 2));
			Eigen::Vector3f obs11(*point_image_in.data(y + 1, x + 1, 0), *point_image_in.data(y + 1, x + 1, 1), *point_image_in.data(y + 1, x + 1, 2));

			Eigen::Vector3f flow00(*flow_image_in.data(y, x, 0), *flow_image_in.data(y, x, 1), *flow_image_in.data(y, x, 2));
			Eigen::Vector3f flow01(*flow_image_in.data(y + 1, x, 0), *flow_image_in.data(y + 1, x, 1), *flow_image_in.data(y + 1, x, 2));
			Eigen::Vector3f flow10(*flow_image_in.data(y, x + 1, 0), *flow_image_in.data(y, x + 1, 1), *flow_image_in.data(y, x + 1, 2));
			Eigen::Vector3f flow11(*flow_image_in.data(y + 1, x + 1, 0), *flow_image_in.data(y + 1, x + 1, 1), *flow_image_in.data(y + 1, x + 1, 2));

			// linear indices of the four pixels
			int idx00 = y * width + x;
			int idx01 = (y + 1) * width + x;
			int idx10 = y * width + (x + 1);
			int idx11 = (y + 1) * width + (x + 1);

			// determine pixel validity
			bool valid00 = obs00.z() > 0 && std::isfinite(flow00.x()) && std::isfinite(flow00.y()) && std::isfinite(flow00.z());
			bool valid01 = obs01.z() > 0 && std::isfinite(flow01.x()) && std::isfinite(flow01.y()) && std::isfinite(flow01.z());
			bool valid10 = obs10.z() > 0 && std::isfinite(flow10.x()) && std::isfinite(flow10.y()) && std::isfinite(flow10.z());
			bool valid11 = obs11.z() > 0 && std::isfinite(flow11.x()) && std::isfinite(flow11.y()) && std::isfinite(flow11.z());

			// region ======= LOWER LEFT TRIANGLE =============
			if (valid00 && valid01 && valid10) {
				float d0 = (obs00 - obs01).norm();
				float d1 = (obs00 - obs10).norm(); // hypotenuse when triangle projected to image plane
				float d2 = (obs01 - obs10).norm();

				if (d0 <= max_triangle_edge_distance && d1 <= max_triangle_edge_distance && d2 <= max_triangle_edge_distance) {
					int vertex_index_0 = pixel_to_vertex_index_map[idx00];
					int vertex_index_1 = pixel_to_vertex_index_map[idx01];
					int vertex_index_2 = pixel_to_vertex_index_map[idx10];

					if (vertex_index_0 == -1) {
						vertex_index_0 = vertex_index;
						pixel_to_vertex_index_map[idx00] = vertex_index;
						vertices.push_back(obs00);
						flows.push_back(flow00);
						pixels.emplace_back(x, y);
						vertex_index++;
					}
					if (vertex_index_1 == -1) {
						vertex_index_1 = vertex_index;
						pixel_to_vertex_index_map[idx01] = vertex_index;
						vertices.push_back(obs01);
						flows.push_back(flow01);
						pixels.emplace_back(x, y + 1);
						vertex_index++;
					}
					if (vertex_index_2 == -1) {
						vertex_index_2 = vertex_index;
						pixel_to_vertex_index_map[idx10] = vertex_index;
						vertices.push_back(obs10);
						flows.push_back(flow10);
						pixels.emplace_back(x + 1, y);
						vertex_index++;
					}

					faces.emplace_back(vertex_index_0, vertex_index_1, vertex_index_2);
				}
			}
			// endregion
			// region ======= UPPER RIGHT TRIANGLE ==================
			if (valid01 && valid10 && valid11) {
				float d0 = (obs10 - obs01).norm();
				float d1 = (obs10 - obs11).norm();
				float d2 = (obs01 - obs11).norm();

				if (d0 <= max_triangle_edge_distance && d1 <= max_triangle_edge_distance && d2 <= max_triangle_edge_distance) {
					int vertex_index_0 = pixel_to_vertex_index_map[idx11];
					int vertex_index_1 = pixel_to_vertex_index_map[idx10];
					int vertex_index_2 = pixel_to_vertex_index_map[idx01];

					if (vertex_index_0 == -1) {
						vertex_index_0 = vertex_index;
						pixel_to_vertex_index_map[idx11] = vertex_index;
						vertices.push_back(obs11);
						flows.push_back(flow11);
						pixels.emplace_back(x + 1, y + 1);
						vertex_index++;
					}
					if (vertex_index_1 == -1) {
						vertex_index_1 = vertex_index;
						pixel_to_vertex_index_map[idx10] = vertex_index;
						vertices.push_back(obs10);
						flows.push_back(flow10);
						pixels.emplace_back(x + 1, y);
						vertex_index++;
					}
					if (vertex_index_2 == -1) {
						vertex_index_2 = vertex_index;
						pixel_to_vertex_index_map[idx01] = vertex_index;
						vertices.push_back(obs01);
						flows.push_back(flow01);
						pixels.emplace_back(x, y + 1);
						vertex_index++;
					}

					faces.emplace_back(vertex_index_0, vertex_index_1, vertex_index_2);
				}
			}
			// endregion
		}
	}

	// Convert to numpy array.
	int vertex_count = vertices.size();
	int face_count = faces.size();

	if (vertex_count > 0 && face_count > 0) {
		// Reference check should be set to false otherwise there is a runtime
		// error. Check why that is the case.
		vertex_positions_out.resize({vertex_count, 3}, false);
		vertex_flows_out.resize({vertex_count, 3}, false);
		vertex_pixels_out.resize({vertex_count, 2}, false);
		face_indices_out.resize({face_count, 3}, false);

		for (int i = 0; i < vertex_count; i++) {
			*vertex_positions_out.mutable_data(i, 0) = vertices[i].x();
			*vertex_positions_out.mutable_data(i, 1) = vertices[i].y();
			*vertex_positions_out.mutable_data(i, 2) = vertices[i].z();

			*vertex_flows_out.mutable_data(i, 0) = flows[i].x();
			*vertex_flows_out.mutable_data(i, 1) = flows[i].y();
			*vertex_flows_out.mutable_data(i, 2) = flows[i].z();

			*vertex_pixels_out.mutable_data(i, 0) = pixels[i].x();
			*vertex_pixels_out.mutable_data(i, 1) = pixels[i].y();
		}

		for (int i = 0; i < face_count; i++) {
			*face_indices_out.mutable_data(i, 0) = faces[i].x();
			*face_indices_out.mutable_data(i, 1) = faces[i].y();
			*face_indices_out.mutable_data(i, 2) = faces[i].z();
		}
	}
}

py::tuple compute_mesh_from_depth_and_flow(const py::array_t<float>& point_image_in, const py::array_t<float>& flow_image_in,
                                           float max_triangle_edge_distance) {
	py::array_t<float> vertex_positions_out;
	py::array_t<float> vertex_flows_out;
	py::array_t<int> vertex_pixels_out;
	py::array_t<int> face_indices_out;
	compute_mesh_from_depth_and_flow(point_image_in, flow_image_in, max_triangle_edge_distance, vertex_positions_out, vertex_flows_out, vertex_pixels_out, face_indices_out);
	return py::make_tuple(vertex_positions_out, vertex_flows_out, vertex_pixels_out, face_indices_out);
}

void filter_depth(py::array_t<unsigned short>& depth_image_in, py::array_t<unsigned short>& depth_image_out, int radius) {
	assert(depth_image_in.ndim() == 2);
	assert(depth_image_out.ndim() == 2);
	unsigned kernel_size = 2 * radius + 1;
	unsigned window_size = kernel_size * kernel_size;

	int width = depth_image_in.shape(1);
	int height = depth_image_in.shape(0);
	assert(depth_image_out.shape(0) == height);
	assert(depth_image_out.shape(1) == width);

	// #pragma omp parallel for
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Get all residuals in the median window.
			int x_min = std::max(x - radius, 0);
			int x_max = std::min(x + radius, int(width) - 1);
			int y_min = std::max(y - radius, 0);
			int y_max = std::min(y + radius, int(height) - 1);

			std::vector<unsigned short> window_values;
			window_values.reserve(window_size);

			for (int y_near = y_min; y_near <= y_max; y_near++) {
				for (int x_near = x_min; x_near <= x_max; x_near++) {
					unsigned short depth = *depth_image_in.data(y_near, x_near);
					if (depth > 0) {
						window_values.push_back(depth);
					}
				}
			}

			// Sort the residuals and pick the median as the middle element.
			unsigned element_count = window_values.size();
			std::sort(window_values.begin(), window_values.end());

			unsigned middle_index = std::floor(element_count / 2);
			unsigned short median = window_values[middle_index];

			// Write out the median value.
			*depth_image_out.mutable_data(y, x) = median;
		}
	}
}

py::array_t<unsigned short> filter_depth(py::array_t<unsigned short>& depth_image_in, int radius) {
	py::array_t<unsigned short> depth_image_out({depth_image_in.shape(0), depth_image_in.shape(1)});
	memset(depth_image_out.mutable_data(0, 0), 0, depth_image_out.size() * sizeof(unsigned short));
	filter_depth(depth_image_in, depth_image_out, radius);
	return depth_image_out;
}

py::array_t<float> warp_flow(const py::array_t<float>& image, const py::array_t<float>& flow, const py::array_t<float>& mask) {
	// We assume:
	//      image shape (3, h, w)
	//      flow shape  (2, h, w)
	//      mask shape  (2, h, w)

	int width = image.shape(2);
	int height = image.shape(1);

	py::array_t<float> imageWarped = py::array_t<float>({3, height, width});
	py::array_t<float> weightsWarped = py::array_t<float>({1, height, width});

	// Initialize to zero.
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			*imageWarped.mutable_data(0, v, u) = 0.0;
			*imageWarped.mutable_data(1, v, u) = 0.0;
			*imageWarped.mutable_data(2, v, u) = 0.0;
			*weightsWarped.mutable_data(0, v, u) = 0.0;
		}
	}

	// Compute image residuals and interpolation weights.
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			// Check if pixel is inside the mask.
			if (*mask.data(0, v, u) <= 0 || *mask.data(1, v, u) <= 0) continue;

			// Compute the warped pixel.
			float u_warped = static_cast<float>(u) + *flow.data(0, v, u);
			float v_warped = static_cast<float>(v) + *flow.data(1, v, u);

			int u0 = std::floor(u_warped);
			int u1 = u0 + 1;
			int v0 = std::floor(v_warped);
			int v1 = v0 + 1;

			if (u0 < 0 || u1 >= width || v0 < 0 || v1 >= height) continue;

			// Interpolate the color contributions.
			float du = u_warped - u0;
			float dv = v_warped - v0;

			float w00 = (1 - du) * (1 - dv);
			float w01 = (1 - du) * dv;
			float w10 = du * (1 - dv);
			float w11 = du * dv;

			float c0 = *image.data(0, v, u);
			float c1 = *image.data(1, v, u);
			float c2 = *image.data(2, v, u);

			*imageWarped.mutable_data(0, v0, u0) += w00 * c0;
			*imageWarped.mutable_data(1, v0, u0) += w00 * c1;
			*imageWarped.mutable_data(2, v0, u0) += w00 * c2;
			*imageWarped.mutable_data(0, v1, u0) += w01 * c0;
			*imageWarped.mutable_data(1, v1, u0) += w01 * c1;
			*imageWarped.mutable_data(2, v1, u0) += w01 * c2;
			*imageWarped.mutable_data(0, v0, u1) += w10 * c0;
			*imageWarped.mutable_data(1, v0, u1) += w10 * c1;
			*imageWarped.mutable_data(2, v0, u1) += w10 * c2;
			*imageWarped.mutable_data(0, v1, u1) += w11 * c0;
			*imageWarped.mutable_data(1, v1, u1) += w11 * c1;
			*imageWarped.mutable_data(2, v1, u1) += w11 * c2;

			*weightsWarped.mutable_data(0, v0, u0) += w00;
			*weightsWarped.mutable_data(0, v1, u0) += w01;
			*weightsWarped.mutable_data(0, v0, u1) += w10;
			*weightsWarped.mutable_data(0, v1, u1) += w11;
		}
	}

	// Normalize image.
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			float w = *weightsWarped.data(0, v, u);
			if (w > 0) {
				*imageWarped.mutable_data(0, v, u) /= w;
				*imageWarped.mutable_data(1, v, u) /= w;
				*imageWarped.mutable_data(2, v, u) /= w;
			} else {
				*imageWarped.mutable_data(0, v, u) = 1.0;
				*imageWarped.mutable_data(1, v, u) = 1.0;
				*imageWarped.mutable_data(2, v, u) = 1.0;
			}
		}
	}

	return imageWarped;
}

py::array_t<float> warp_rigid(
		const py::array_t<float>& rgbxyz_image,
		const py::array_t<float>& rotation,
		const py::array_t<float>& translation,
		float fx, float fy, float cx, float cy
) {
	// We assume:
	//      rgbd shape (6, h, w)
	//      rotation shape  (9)
	//      translation shape  (2)

	int width = rgbxyz_image.shape(2);
	int height = rgbxyz_image.shape(1);

	float r00 = *rotation.data(0);
	float r01 = *rotation.data(1);
	float r02 = *rotation.data(2);
	float r10 = *rotation.data(3);
	float r11 = *rotation.data(4);
	float r12 = *rotation.data(5);
	float r20 = *rotation.data(6);
	float r21 = *rotation.data(7);
	float r22 = *rotation.data(8);
	float t0 = *translation.data(0);
	float t1 = *translation.data(1);
	float t2 = *translation.data(2);

	py::array_t<float> image_warped = py::array_t<float>({3, height, width});
	py::array_t<float> weights_warped = py::array_t<float>({1, height, width});

	// Initialize to zero.
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			*image_warped.mutable_data(0, v, u) = 0.0;
			*image_warped.mutable_data(1, v, u) = 0.0;
			*image_warped.mutable_data(2, v, u) = 0.0;
			*weights_warped.mutable_data(0, v, u) = 0.0;
		}
	}

	// Compute image residuals and interpolation weights.
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			// Compute the warped pixel.
			float x = *rgbxyz_image.data(3, v, u);
			float y = *rgbxyz_image.data(4, v, u);
			float z = *rgbxyz_image.data(5, v, u);
			if (z <= 0) continue;

			float x_def = r00 * x + r01 * y + r02 * z + t0;
			float y_def = r10 * x + r11 * y + r12 * z + t1;
			float z_def = r20 * x + r21 * y + r22 * z + t2;
			if (z_def <= 0) continue;

			float u_warped = fx * x_def / z_def + cx;
			float v_warped = fy * y_def / z_def + cy;

			int u0 = std::floor(u_warped);
			int u1 = u0 + 1;
			int v0 = std::floor(v_warped);
			int v1 = v0 + 1;

			if (u0 < 0 || u1 >= width || v0 < 0 || v1 >= height) continue;

			// Interpolate the color contributions.
			float du = u_warped - u0;
			float dv = v_warped - v0;

			float w00 = (1 - du) * (1 - dv);
			float w01 = (1 - du) * dv;
			float w10 = du * (1 - dv);
			float w11 = du * dv;

			float c0 = *rgbxyz_image.data(0, v, u);
			float c1 = *rgbxyz_image.data(1, v, u);
			float c2 = *rgbxyz_image.data(2, v, u);

			*image_warped.mutable_data(0, v0, u0) += w00 * c0;
			*image_warped.mutable_data(1, v0, u0) += w00 * c1;
			*image_warped.mutable_data(2, v0, u0) += w00 * c2;
			*image_warped.mutable_data(0, v1, u0) += w01 * c0;
			*image_warped.mutable_data(1, v1, u0) += w01 * c1;
			*image_warped.mutable_data(2, v1, u0) += w01 * c2;
			*image_warped.mutable_data(0, v0, u1) += w10 * c0;
			*image_warped.mutable_data(1, v0, u1) += w10 * c1;
			*image_warped.mutable_data(2, v0, u1) += w10 * c2;
			*image_warped.mutable_data(0, v1, u1) += w11 * c0;
			*image_warped.mutable_data(1, v1, u1) += w11 * c1;
			*image_warped.mutable_data(2, v1, u1) += w11 * c2;

			*weights_warped.mutable_data(0, v0, u0) += w00;
			*weights_warped.mutable_data(0, v1, u0) += w01;
			*weights_warped.mutable_data(0, v0, u1) += w10;
			*weights_warped.mutable_data(0, v1, u1) += w11;
		}
	}

	// Normalize image.
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			float w = *weights_warped.data(0, v, u);
			if (w > 0) {
				*image_warped.mutable_data(0, v, u) /= w;
				*image_warped.mutable_data(1, v, u) /= w;
				*image_warped.mutable_data(2, v, u) /= w;
			} else {
				*image_warped.mutable_data(0, v, u) = 1.0;
				*image_warped.mutable_data(1, v, u) = 1.0;
				*image_warped.mutable_data(2, v, u) = 1.0;
			}
		}
	}

	return image_warped;
}

py::array_t<float>
warp_3d(const py::array_t<float>& rgbxyz_image, const py::array_t<float>& points,
        const py::array_t<int>& mask, float fx, float fy, float cx, float cy) {

	// We assume:
	//      image shape             (6, h, w)
	//      points shape            (3, h, w)
	//      mask shape    (h, w)

	int width = rgbxyz_image.shape(2);
	int height = rgbxyz_image.shape(1);

	py::array_t<float> image_warped = py::array_t<float>({3, height, width});
	py::array_t<float> weights_warped = py::array_t<float>({1, height, width});

	// Initialize to zero.
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			*image_warped.mutable_data(0, v, u) = 0.0;
			*image_warped.mutable_data(1, v, u) = 0.0;
			*image_warped.mutable_data(2, v, u) = 0.0;
			*weights_warped.mutable_data(0, v, u) = 0.0;
		}
	}

	// Compute image residuals and interpolation weights.
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			// Compute the warped pixel.
			if (*mask.data(v, u) <= 0) continue;

			float z = *rgbxyz_image.data(5, v, u);
			if (z <= 0) continue;

			float x_def = *points.data(0, v, u);
			float y_def = *points.data(1, v, u);
			float z_def = *points.data(2, v, u);
			if (z_def <= 0) continue;

			float u_warped = fx * x_def / z_def + cx;
			float v_warped = fy * y_def / z_def + cy;

			int u0 = std::floor(u_warped);
			int u1 = u0 + 1;
			int v0 = std::floor(v_warped);
			int v1 = v0 + 1;

			if (u0 < 0 || u1 >= width || v0 < 0 || v1 >= height) continue;

			// Interpolate the color contributions.
			float du = u_warped - u0;
			float dv = v_warped - v0;

			float w00 = (1 - du) * (1 - dv);
			float w01 = (1 - du) * dv;
			float w10 = du * (1 - dv);
			float w11 = du * dv;

			float c0 = *rgbxyz_image.data(0, v, u);
			float c1 = *rgbxyz_image.data(1, v, u);
			float c2 = *rgbxyz_image.data(2, v, u);

			*image_warped.mutable_data(0, v0, u0) += w00 * c0;
			*image_warped.mutable_data(1, v0, u0) += w00 * c1;
			*image_warped.mutable_data(2, v0, u0) += w00 * c2;
			*image_warped.mutable_data(0, v1, u0) += w01 * c0;
			*image_warped.mutable_data(1, v1, u0) += w01 * c1;
			*image_warped.mutable_data(2, v1, u0) += w01 * c2;
			*image_warped.mutable_data(0, v0, u1) += w10 * c0;
			*image_warped.mutable_data(1, v0, u1) += w10 * c1;
			*image_warped.mutable_data(2, v0, u1) += w10 * c2;
			*image_warped.mutable_data(0, v1, u1) += w11 * c0;
			*image_warped.mutable_data(1, v1, u1) += w11 * c1;
			*image_warped.mutable_data(2, v1, u1) += w11 * c2;

			*weights_warped.mutable_data(0, v0, u0) += w00;
			*weights_warped.mutable_data(0, v1, u0) += w01;
			*weights_warped.mutable_data(0, v0, u1) += w10;
			*weights_warped.mutable_data(0, v1, u1) += w11;
		}
	}

	// Normalize image.
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			float w = *weights_warped.data(0, v, u);
			if (w > 0) {
				*image_warped.mutable_data(0, v, u) /= w;
				*image_warped.mutable_data(1, v, u) /= w;
				*image_warped.mutable_data(2, v, u) /= w;
			} else {
				*image_warped.mutable_data(0, v, u) = 1.0;
				*image_warped.mutable_data(1, v, u) = 1.0;
				*image_warped.mutable_data(2, v, u) = 1.0;
			}
		}
	}

	return image_warped;
}

} //namespace image_proc