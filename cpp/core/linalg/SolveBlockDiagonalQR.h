//  ================================================================
//  Created by Gregory Kramida on 4/4/23.
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
// stdlib includes
// third-party includes
#include <open3d/core/Tensor.h>
// local includes
#include "core/platform_independence/Macros.h"

namespace nnrt::core::linalg {
void SolveQRBlockDiagonal(open3d::core::Tensor &X, const open3d::core::Tensor &A_blocks, const open3d::core::Tensor &B);

namespace internal {
void SolveQRBlockDiagonalCPU(
        void *A_blocks_data,
        void *B_data,
        int64_t A_and_B_block_row_count,
        int64_t B_column_count,
        int64_t block_count,
        open3d::core::Dtype data_type,
        const open3d::core::Device &device
);

void SolveQRBlockDiagonalCUDA(
        void *A_blocks_data,
        void *B_data,
        int64_t A_and_B_block_row_count,
        int64_t B_column_count,
        int64_t block_count,
        open3d::core::Dtype data_type,
        const open3d::core::Device &device
);

} // namespace internal
} // namespace nnrt::core::linalg