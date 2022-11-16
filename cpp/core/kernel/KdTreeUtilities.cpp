//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/12/22.
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
#include "core/kernel/KdTreeUtilities.h"
namespace o3c = open3d::core;

namespace nnrt::core::kernel::kdtree {

open3d::core::Blob BlobToDevice(const open3d::core::Blob& index_data, int64_t byte_count, const o3c::Device& device) {
	o3c::Blob target_blob(byte_count, device);
	o3c::MemoryManager::Memcpy(target_blob.GetDataPtr(), device, index_data.GetDataPtr(), index_data.GetDevice(), byte_count);
	return target_blob;
}

} // namespace nnrt::core::kernel::kdtree