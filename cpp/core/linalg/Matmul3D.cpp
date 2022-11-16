//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/12/22.
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
#include "Matmul3D.h"
#include "core/DeviceSelection.h"
namespace o3c = open3d::core;

namespace nnrt::core::linalg {
namespace o3c = open3d::core;
namespace o3u = open3d::utility;

void Matmul3D(open3d::core::Tensor& output, const open3d::core::Tensor& array_of_matrices_A, const open3d::core::Tensor& array_of_matrices_B){
	o3c::Device device = array_of_matrices_A.GetDevice();
	o3c::Dtype dtype_original = array_of_matrices_A.GetDtype();
	o3c::Dtype dtype;

	o3c::AssertTensorDtype(array_of_matrices_B, dtype_original);
	o3c::AssertTensorDevice(array_of_matrices_B, device);

	if (dtype_original != o3c::Float32 && dtype_original != o3c::Float64) {
		o3u::LogDebug("Converting to Float32 dtype to from {}.",
		              dtype_original.ToString());
		dtype = o3c::Float32;
	} else {
		dtype = dtype_original;
	}

	o3c::SizeVector A_shape = array_of_matrices_A.GetShape();
	o3c::SizeVector B_shape = array_of_matrices_B.GetShape();


	if (A_shape.size() != 3) {
		o3u::LogError("Tensor A must be 3D (array of matrices), but got {}D.", A_shape.size());
	}
	if (B_shape.size() != 2 && B_shape.size() != 3) {
		o3u::LogError("Tensor B must be 2D (array of vectors) or 3D (array of matrices), but got {}D.", B_shape.size());
	}
	if (A_shape[2] != B_shape[1]) {
		o3u::LogError("Tensor A columns {} mismatch with Tensor B rows {}.", A_shape[2], B_shape[1]);
	}
	if(A_shape[0] != B_shape[0]){
		o3u::LogError("Tensors A and B should have matching first dimension. Got: {} vs. {}",A_shape[0], B_shape[0]);
	}

	int64_t m = A_shape[1];
	int64_t k = A_shape[2];
	int64_t n = B_shape.size() == 3 ? B_shape[2] : 1;

	const int64_t batch_size = A_shape[0];

	o3c::Tensor A_contiguous = array_of_matrices_A.Contiguous().To(dtype);
	o3c::Tensor B_contiguous = array_of_matrices_B.Contiguous().To(dtype);

	const void* A_data = A_contiguous.GetDataPtr();
	const void* B_data = B_contiguous.GetDataPtr();

	output = o3c::Tensor({batch_size, m, n}, dtype, device);
	void* C_data = output.GetDataPtr();

	core::ExecuteOnDevice(
			device,
			[&] { Matmul3D<o3c::Device::DeviceType::CPU>(A_data, B_data, C_data, m, k, n, batch_size, dtype); },
			[&] { NNRT_IF_CUDA(Matmul3D<o3c::Device::DeviceType::CUDA>(A_data, B_data, C_data, m, k, n, batch_size, dtype);); }
	);
	output = output.To(dtype_original);

}

} // nnrt::core::linalg