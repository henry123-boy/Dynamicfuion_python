//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/9/23.
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
// local includes
#include "core/linalg/MatmulBlockSparseImpl.h"
namespace nnrt::core::linalg::internal {

template
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MatmulBlockSparseRowWise<open3d::core::Device::DeviceType::CUDA>(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates,
		bool padded
);

} // namespace nnrt::core::linalg::internal