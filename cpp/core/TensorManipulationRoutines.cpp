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
// 3rd-party
#include <open3d/core/Dispatch.h>
// local
#include "core/TensorManipulationRoutines.h"
#include "core/linalg/Matmul3D.h"
#include "core/GetDType.h"

namespace utility = open3d::utility;
namespace o3c = open3d::core;


namespace nnrt::core {
//TODO: move to core::functional
open3d::core::Tensor VStack(const open3d::core::Tensor& tensor1, const open3d::core::Tensor& tensor2) {
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
        utility::LogError(
                "Shape mismatch. Tensor of shape {} is not "
                "compatible with tensor of shape {} for vertical combination.",
                tensor2.GetShape(), tensor1.GetShape());
    }

    o3c::Tensor concatenated_tensor =
            o3c::Tensor::Empty(tensor2_shape, tensor1.GetDtype(), tensor1.GetDevice());

    concatenated_tensor.SetItem(o3c::TensorKey::Slice(0, tensor1_length, 1), tensor1);
    concatenated_tensor.SetItem(o3c::TensorKey::Slice(tensor1_length, combined_length, 1), tensor2);
    return concatenated_tensor;
}

//TODO: move to core::linalg
open3d::core::Tensor Matmul3D(const open3d::core::Tensor& tensor1, const open3d::core::Tensor& tensor2) {

    o3c::Tensor output;
    core::linalg::Matmul3D(output, tensor1, tensor2);
    return output;

}



template<typename TElement>
open3d::core::Tensor SingleValueTensor(TElement element, const open3d::core::Device& device) {
    return o3c::Tensor(std::vector<TElement>({element}), {1}, GetDType<TElement>(), device);
}

template open3d::core::Tensor SingleValueTensor<double>(double element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<float>(float element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<int8_t>(int8_t element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<int16_t>(int16_t element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<int32_t>(int32_t element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<int64_t>(int64_t element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<uint8_t>(uint8_t element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<uint16_t>(uint16_t element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<uint32_t>(uint32_t element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<uint64_t>(uint64_t element, const open3d::core::Device& device);

template open3d::core::Tensor SingleValueTensor<bool>(bool element, const open3d::core::Device& device);

template<typename TElementOut, typename TElement>
TElementOut At_Dispatched(const open3d::core::Tensor& tensor, const std::vector<int64_t>& coordinates) {
    if (!std::is_same<TElementOut, TElement>::value) {
        open3d::utility::LogError(
                "Trying to access an element of type {} from a tensor with elements of type {}. Types must match.",
                typeid(TElementOut).name(), typeid(TElement).name()
        );
    }
    const o3c::Tensor key(coordinates, {static_cast<int64_t>(coordinates.size())}, o3c::Int64, tensor.GetDevice());
    return tensor.GetItem(o3c::TensorKey::IndexTensor(key)).ToFlatVector<TElementOut>()[0];
}

template<typename TElement>
TElement At(const open3d::core::Tensor& tensor, int64_t first_coord...) {
    auto dimension_count = tensor.NumDims();
    std::vector<int64_t> coordinates;

    va_list args;
    va_start(args, first_coord);
    coordinates.push_back(first_coord);
    for (int i_dimension = 1; i_dimension < dimension_count; i_dimension++) {
        int64_t coordinate = va_arg(args, int64_t);
        coordinates.push_back(coordinate);
    }
    va_end(args);

    TElement out;
    DISPATCH_DTYPE_TO_TEMPLATE(tensor.GetDtype(), [&]() {
        out = At_Dispatched<TElement, scalar_t>(tensor, coordinates);
    });
    return out;
}

template double At<double>(const open3d::core::Tensor& tensor, int64_t first_coord...);
template float At<float>(const open3d::core::Tensor& tensor, int64_t first_coord...);

}