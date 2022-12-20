//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/19/22.
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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/core/Dispatch.h>
#include <Eigen/Dense>

// local includes
#include "core/GetDType.h"

namespace utility = open3d::utility;

namespace nnrt::core {




template<typename TEigenMatrix>
open3d::core::Tensor EigenMatrixToTensor(const TEigenMatrix& matrix, const open3d::core::Device& device) {
    const auto row_count = static_cast<int64_t>(matrix.rows());
    const auto col_count = static_cast<int64_t>(matrix.cols());
    if (TEigenMatrix::IsRowMajor) {
        return {matrix.data(), {row_count, col_count}, GetDType<typename TEigenMatrix::Scalar>()};
    } else {
        open3d::core::Tensor tensor(matrix.data(), {col_count, row_count}, GetDType<typename TEigenMatrix::Scalar>());
        return tensor.T();
    }
}

template<typename TEigenMatrix, typename TElement>
TEigenMatrix TensorToEigenMatrix_Dispatched(const open3d::core::Tensor& tensor) {
    auto host_tensor = tensor.To(open3d::core::Device("CPU:0"));
    if (!std::is_same<typename TEigenMatrix::Scalar, TElement>::value) {
        open3d::utility::LogError("Trying to convert a tensor of incompatible type {} to a matrix of type {}.",
                                  typeid(TElement).name(), typeid(typename TEigenMatrix::Scalar).name());
    }

    if (TEigenMatrix::IsRowMajor) {
        TEigenMatrix matrix = Eigen::Map<TEigenMatrix>(host_tensor.GetDataPtr<typename TEigenMatrix::Scalar>());
        return matrix;
    } else {
        typedef Eigen::Matrix<
                typename TEigenMatrix::Scalar,
                TEigenMatrix::ColsAtCompileTime,
                TEigenMatrix::RowsAtCompileTime, Eigen::ColMajor
        > TEigenTransposed;
        TEigenTransposed transposed;
        transposed = Eigen::Map<TEigenTransposed>(host_tensor.GetDataPtr<typename TEigenMatrix::Scalar>());
        return transposed.transpose();
    }
}

template<typename TEigenMatrix>
TEigenMatrix TensorToEigenMatrix(const open3d::core::Tensor& tensor) {
    if (TEigenMatrix::RowsAtCompileTime != Eigen::Dynamic) {
        open3d::core::AssertTensorShape(tensor, { TEigenMatrix::RowsAtCompileTime, open3d::utility::nullopt });
    }
    if (TEigenMatrix::ColsAtCompileTime != Eigen::Dynamic) {
        open3d::core::AssertTensorShape(tensor, { open3d::utility::nullopt, TEigenMatrix::ColsAtCompileTime });
    }

    TEigenMatrix matrix;
    DISPATCH_DTYPE_TO_TEMPLATE(tensor.GetDtype(), [&]() {
        matrix = TensorToEigenMatrix_Dispatched<TEigenMatrix, scalar_t>(tensor);
    });
    return matrix;
}
} // namespace nnrt::core