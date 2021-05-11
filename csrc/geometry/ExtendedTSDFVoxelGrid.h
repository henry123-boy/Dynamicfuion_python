//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/6/21.
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
#include <open3d/t/geometry/TSDFVoxelGrid.h>

namespace nnrt {
namespace geometry {

class ExtendedTSDFVoxelGrid : public open3d::t::geometry::TSDFVoxelGrid {
	using open3d::t::geometry::TSDFVoxelGrid::TSDFVoxelGrid;
public:
	/// Extract all indexed voxel centers.
	open3d::core::Tensor ExtractVoxelCenters();

	/// Extract all TSDF values in the same order as the voxel centers in the output
	/// of the ExtractVoxelCenters function
	open3d::core::Tensor ExtractTSDFValuesAndWeights();

	/// Extract all SDF values in the specified spatial extent
	/// All undefined SDF values will be kept as -2.0
	open3d::core::Tensor ExtractValuesInExtent(int min_x, int min_y, int min_z, int max_x, int max_y, int max_z);

};

}// namespace nnrt
}// namespace geometry





