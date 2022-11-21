//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/21/22.
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
// stdlib includes
#include <variant>

// third-party includes
#include <open3d/core/Dispatch.h>

// local includes
#include "core/functional/Masking.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;


namespace nnrt::core {
template<typename TElement>
open3d::core::Tensor SetMaskedToValue(open3d::core::Tensor& tensor, const open3d::core::Tensor& mask, TElement element) {
	auto dtype = tensor.GetDtype();
	auto device = tensor.GetDevice();
	o3c::AssertTensorDtype(mask, o3c::Bool);
	o3c::AssertTensorDevice(mask, device);

	DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
		if (std::is_same_v<scalar_t, TElement>) {
			auto valueTensor = o3c::Tensor(std::vector<TElement>{element}, {1}, dtype, device);
			tensor.SetItem(o3c::TensorKey::IndexTensor(mask), valueTensor);
		} else {
			utility::LogError("Attempting to set values of tensor with dtype {} to value {} (type mismatch).", dtype, element);
		}
	});
	return tensor;
}
} // namespace nnrt::core
