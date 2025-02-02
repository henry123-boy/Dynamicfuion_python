//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/4/22.
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
// local
#include "core/functional/Sorting.h"
#include "core/functional/kernel/Sorting.h"
#include "core/functional/kernel/Tile.h"

namespace o3c = open3d::core;

namespace nnrt::core::functional {

open3d::core::Tensor SortTensorAlongLastDimension(const open3d::core::Tensor& unsorted, bool non_negative_first, SortOrder order) {
	if (unsorted.NumDims() == 0 || unsorted.NumElements() == 0) {
		return unsorted;
	}
	o3c::Tensor sorted;
	kernel::SortTensorAlongLastDimension(sorted, unsorted.Contiguous(), non_negative_first, order);
	return sorted;
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
SortTensorAlongLastDimensionByKey(const open3d::core::Tensor& values, const open3d::core::Tensor& keys, bool non_negative_first, SortOrder order) {
	if (values.NumDims() == 0 || values.NumElements() == 0) {
		return {};
	}
	o3c::AssertTensorShape(values, keys.GetShape());
	o3c::AssertTensorDevice(values, keys.GetDevice());

	o3c::Tensor sorted_values, sorted_keys;
	kernel::SortTensorAlongLastDimensionByKey(sorted_values, sorted_keys, values.Contiguous(), keys.Contiguous(), non_negative_first, order);
	return std::make_tuple(sorted_values, sorted_keys);
}


open3d::core::Tensor ArgSortTensorAlongLastDimension(const open3d::core::Tensor& unsorted, bool non_negative_first, SortOrder order) {
	o3c::Tensor unsorted_index_untiled =
			o3c::Tensor::Arange(0, unsorted.GetShape(-1), 1, o3c::Int64, unsorted.GetDevice())
			.Reshape({-1, unsorted.GetShape(-1)});
	o3c::Tensor unsorted_index_tiled;
	int64_t tile_count = 1;
	size_t last_dimension = unsorted.GetShape().size() - 1;
	size_t current_dimension = 0;
	for(auto extent : unsorted.GetShape()){
		if(current_dimension < last_dimension){
			tile_count *= extent;
		}
		current_dimension++;
	}
	functional::kernel::Tile(unsorted_index_tiled, unsorted_index_untiled, static_cast<int>(tile_count), 1);
	unsorted_index_tiled = unsorted_index_tiled.Reshape(unsorted.GetShape());

	o3c::Tensor sorted_index, sorted;
	std::tie(sorted_index, sorted) = SortTensorAlongLastDimensionByKey(unsorted_index_tiled, unsorted, non_negative_first, order);
	return sorted_index;
}


open3d::core::Tensor SortTensorByColumn(const open3d::core::Tensor& unsorted, int column) {
	if (unsorted.NumDims() == 0 || unsorted.NumElements() == 0) {
		return unsorted;
	}
	o3c::Tensor sorted;
	kernel::SortTensorByColumn(sorted, unsorted, column, false);
	return sorted;
}

open3d::core::Tensor SortTensorByColumns(const open3d::core::Tensor& unsorted, const o3c::SizeVector& columns) {
	o3c::Tensor sorted = unsorted.Clone();
	for(const int64_t& column : columns){
		kernel::SortTensorByColumn(sorted, sorted, static_cast<int32_t>(column), true);
	}
	return sorted;
}

open3d::core::Tensor ArgSortByColumn(const open3d::core::Tensor& unsorted, int column) {
	if (unsorted.NumDims() == 0 || unsorted.NumElements() == 0) {
		return {};
	}
	o3c::Tensor index;
	kernel::ArgSortTensorByColumn(index, unsorted, column);
	return index;
}



} // nnrt::core::functional