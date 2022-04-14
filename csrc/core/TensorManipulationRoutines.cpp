//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/25/21.
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
#include "core/TensorManipulationRoutines.h"
#include "core/linalg/Matmul3D.h"

namespace o3u = open3d::utility;
namespace o3c = open3d::core;

namespace nnrt::core{
	open3d::core::Tensor CombineAlongAxis0(const open3d::core::Tensor& tensor1, const open3d::core::Tensor& tensor2){
		o3c::AssertTensorDtype(tensor2, tensor1.GetDtype());
		o3c::AssertTensorDevice(tensor2, tensor1.GetDevice());

		auto tensor1_length = tensor1.GetLength();

		// Check shape compatibility.
		auto tensor1_shape = tensor1.GetShape();
		auto tensor2_shape = tensor2.GetShape();
		int64_t combined_length = tensor2_shape[0] + tensor1_shape[0];
		tensor2_shape[0] = combined_length;
		tensor1_shape[0] = combined_length;
		if (tensor2_shape != tensor1_shape) {
			o3u::LogError(
					"Shape mismatch. Tensor of shape {} is not "
					"compatible with tensor of shape {} for vertical combination.",
					tensor2.GetShape(), tensor1.GetShape());
		}

		o3c::Tensor concatenated_tensor =
				o3c::Tensor::Empty(tensor2_shape, tensor1.GetDtype(),tensor1.GetDevice());

		concatenated_tensor.SetItem(o3c::TensorKey::Slice(0, tensor1_length, 1), tensor1);
		concatenated_tensor.SetItem(o3c::TensorKey::Slice(tensor1_length, combined_length, 1), tensor2);
		return concatenated_tensor;
	}

	open3d::core::Tensor Matmul3D(const open3d::core::Tensor& tensor1, const open3d::core::Tensor& tensor2){

		o3c::Tensor output;
		core::linalg::Matmul3D(output, tensor1, tensor2);
		return output;

	}
}