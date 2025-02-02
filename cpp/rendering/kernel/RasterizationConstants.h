//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/6/22.
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
namespace nnrt::rendering::kernel {
// The maximum number of points per pixel that we can return. Since we use
// thread-local arrays to hold and sort points, the maximum size of the array
// needs to be known at compile time. There might be some fancy template magic
// we could use to make this more dynamic, but for now just fix a constant.
#define MAX_POINTS_PER_PIXEL 8

#define MIN_NEAR_CLIPPING_DISTANCE 0.0

// TODO: move to core
#define K_EPSILON 1e-8f

#define MAX_BINS_ALONG_IMAGE_DIMENSION 22

}