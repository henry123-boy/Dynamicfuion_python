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
#pragma once
// third-party includes
#include <open3d/core/ParallelFor.h>
// local includes
#include "core/functional/kernel/Tile.h"
#include <open3d/utility/Parallel.h>
#include "core/platform_independence/Qualifiers.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::functional::kernel {

template<typename open3d::core::Device::DeviceType TDeviceType>
void Tile(open3d::core::Tensor& tiled, const open3d::core::Tensor& source_tensor, int rows, int columns) {
	int tensor_width = source_tensor.GetShape(1);
	int tensor_height = source_tensor.GetShape(2);
	o3c::AssertTensorShape(source_tensor, { tensor_height, tensor_width });
	o3c::Device device = source_tensor.GetDevice();
	tiled = o3c::Tensor({tensor_height * rows, tensor_width * columns}, source_tensor.GetDtype(), device);
	auto tiled_data = reinterpret_cast<uint8_t*>(tiled.GetDataPtr());
	auto source_data = reinterpret_cast<const uint8_t*>(source_tensor.GetDataPtr());


#ifdef __CUDACC__
	//TODO: not sure if this works faster than just a bunch of cudaMemcpy calls. Test.
	auto element_byte_size = source_tensor.GetDtype().ByteSize();
	int64_t repeat_count = rows * columns;
	int64_t source_element_count = source_tensor.NumElements();
	int64_t tiled_element_count = repeat_count * source_element_count;
	o3c::ParallelFor(
		device,
		tiled_element_count,
		NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_output_element){
			int64_t i_input_element = i_output_element % source_element_count;
			memcpy(tiled_data + i_output_element * element_byte_size, source_data + i_input_element * element_byte_size, element_byte_size);
		}
	);
#endif
	auto source_tensor_byte_size = source_tensor.GetDtype().ByteSize() * source_tensor.NumElements();

#pragma omp parallel for schedule(static) num_threads(open3d::utility::EstimateMaxThreads()) \
    default(none) \
    firstprivate(source_tensor_byte_size, A_block_stride, B_block_stride, C_block_stride) \
    shared(A_data, B_data, C_data, A_array, B_array, C_array)
	for (int i_matrix = 0; i_matrix < batch_size; i_matrix++) {
	}

} // namespace nnrt::core::functional::kernel