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
#pragma once

#include <open3d/core/Tensor.h>

// TODO: move all this to nnrt::core::functional (and update file structure accordingly)
namespace nnrt::core{
	// TODO: open3d/core/TensorFunction.h has `Concatenate`, which should deprecate this
	open3d::core::Tensor VStack(const open3d::core::Tensor& tensor1, const open3d::core::Tensor& tensor2);
	open3d::core::Tensor Matmul3D(const open3d::core::Tensor& tensor1, const open3d::core::Tensor& tensor2);
    template<typename TElement>
    open3d::core::Tensor SingleValueTensor(TElement element, const open3d::core::Device& device);
    template<typename TElement>
    TElement At(const open3d::core::Tensor& tensor, int64_t first_coord...);
} // nnrt::core