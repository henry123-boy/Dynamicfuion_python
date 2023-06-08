//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/4/23.
//  Copyright (c) Gregory Kramida. All rights reserved.
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
#include "core/linalg/SolveBlockDiagonalQR.h"

namespace nnrt::core::linalg::internal {

namespace utility = open3d::utility;

void SolveBlockDiagonalQR_CUDA_FullRank(
        void *A_blocks_data,
        void *B_data,
        int64_t A_and_B_block_row_count,
        int64_t B_column_count,
        int64_t block_count,
        open3d::core::Dtype data_type,
        const open3d::core::Device &device
) {
    //TODO
    utility::LogError("Not implemented.");
}

void SolveBlockDiagonalQR_CUDA_General(
        void *A_blocks_data,
        void *B_data,
        int64_t A_and_B_block_row_count,
        int64_t B_column_count,
        int64_t block_count,
        open3d::core::Dtype data_type,
        const open3d::core::Device &device
) {
    //TODO
    utility::LogError("Not implemented.");
}

} // namespace nnrt::core::linalg::internal