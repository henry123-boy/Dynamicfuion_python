//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/8/23.
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

// local includes

namespace nnrt::core{

#define OPTIMAL_CUDA_BLOCK_THREAD_COUNT 256
#define ASSUMED_SHARED_BLOCK_MEMORY_SIZE 49152 //48 Kb

__host__ __device__
static inline int ceildiv( int x, int y )
{
	return (x + y - 1)/y;
}

} // nnrt::core