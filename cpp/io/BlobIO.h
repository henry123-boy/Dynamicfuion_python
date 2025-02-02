//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 8/8/22.
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
// stdlib
#include <istream>
#include <ostream>
// 3rd party
#include <open3d/core/Blob.h>

namespace nnrt::io {

std::ostream& operator<<(std::ostream& ostream, const std::pair<const std::shared_ptr<open3d::core::Blob>&, int64_t>& blob_and_bytesize);
std::istream& operator>>(std::istream& istream, std::pair<std::shared_ptr<open3d::core::Blob>, int64_t>& blob_and_bytesize);

} // namespace nnrt::io