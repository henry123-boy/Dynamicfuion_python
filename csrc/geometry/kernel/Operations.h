//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/14/21.
//  Copyright (c) 2021 Gregory Kramida
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

#include <utility/PlatformIndependence.h>

namespace nnrt{
namespace geometry{
namespace kernel{
template<typename T>
CPU_OR_CUDA T squared_euclidean_distance_3D(T point_a_x, T point_a_y, T point_a_z, T point_b_x, T point_b_y, T point_b_z){
	T dist_x = point_b_x - point_a_x;
	T dist_y = point_b_y - point_a_y;
	T dist_z = point_b_z - point_a_z;
	return dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;
}
} // namespace kernel
} // namespace geometry
} // namespace nnrt
