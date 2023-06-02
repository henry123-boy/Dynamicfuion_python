//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/2/23.
//  Copyright (c) 2023 Gregory Kramida
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
// stdlib includes

// third-party includes

// local includes
#include "ArapHessian.h"
#include "alignment/functional/kernel/ArapHessian.h"
#include "core/linalg/FactorizeBlocksCholesky.h"

namespace o3c = open3d::core;

namespace nnrt::alignment::functional {

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
ComputeArapBlockSparseHessianApproximation(
		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& condensed_edge_jacobians,
		int64_t first_layer_node_count,
		int64_t node_count
) {
	o3c::Tensor arap_hessian_blocks_upper, arap_hessian_upper_block_coordinates, arap_hessian_block_breadboard, arap_hessian_blocks_diagonal;

	kernel::ArapSparseHessianApproximation(
			arap_hessian_blocks_upper,
			arap_hessian_upper_block_coordinates,
			arap_hessian_block_breadboard,
			arap_hessian_blocks_diagonal,

			edges,
			condensed_edge_jacobians,
			first_layer_node_count,
			node_count
	);

	return std::make_tuple(arap_hessian_blocks_upper, arap_hessian_upper_block_coordinates,
						   arap_hessian_block_breadboard, arap_hessian_blocks_diagonal);
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor> FactorArapBlockSparseHessianApproximation(
		const open3d::core::Tensor& arap_hessian_blocks_upper,
		const open3d::core::Tensor& arap_hessian_upper_block_coordinates,
		const open3d::core::Tensor& arap_hessian_block_breadboard,
		const open3d::core::Tensor& arap_hessian_blocks_diagonal,
		int64_t first_layer_node_count
) {
	o3c::Tensor L_diag_upper_left;
	core::linalg::FactorizeBlocksCholesky(L_diag_upper_left, arap_hessian_blocks_diagonal.Slice(0,0, first_layer_node_count));


}


} // namespace nnrt::alignment::functional