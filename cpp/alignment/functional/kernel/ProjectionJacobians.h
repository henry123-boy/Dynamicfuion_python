//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/6/22.
//  Copyright (c) 2022 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#pragma once
// stdlib includes

// third-party includes
#include <Eigen/Dense>

// local includes
#include "core/PlatformIndependence.h"

namespace nnrt::alignment::functional::kernel {
template<typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline Matrix2x3f CameraToNdcSpaceProjectionJacobian(float ndc_focal_coefficient_x,
                                                     float ndc_focal_coefficient_y,
                                                     TVertex camera_space_vertex) {
	Matrix2x3f jacobian;
	//TODO: try optimizing by avoiding the comma initializer syntax,
	// see https://stackoverflow.com/a/17704129/844728
	jacobian <<
	         ndc_focal_coefficient_x / camera_space_vertex.z(), 0.f, -ndc_focal_coefficient_x * camera_space_vertex.x() / camera_space_vertex.z(),
			0.f, ndc_focal_coefficient_y / camera_space_vertex.z(), -ndc_focal_coefficient_y * camera_space_vertex.y() / camera_space_vertex.z();
	return jacobian;
}
} // namespace nnrt::alignment::functional::kernel