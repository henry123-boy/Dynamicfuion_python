//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/5/23.
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

# ifdef BUILD_CUDA_MODULE
// third-party includes
#include <magma_v2.h>

namespace nnrt::core::linalg {

// local includes
class MagmaManager {
public:

	MagmaManager(MagmaManager &other) = delete;
	void operator=(const MagmaManager &) = delete;

	static MagmaManager& GetInstance();
public:
	void SetDevice(int device);
	magma_queue_t  GetDefaultQueue() const;

protected:
	MagmaManager();
	virtual ~MagmaManager();

private:
	int current_device = -1;
	magma_queue_t default_magma_queue;
};

} // nnrt::core::linalg
#endif