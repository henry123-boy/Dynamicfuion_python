//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/11/21.
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

#include <atomic>

#include <open3d/core/Dispatch.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/t/geometry/kernel/GeometryMacros.h>
#include <open3d/t/geometry/kernel/TSDFVoxel.h>

using namespace open3d::t::geometry::kernel::tsdf;

#define DISPATCH_BYTESIZE_TO_VOXEL(BYTESIZE, ...)            \
    [&] {                                                    \
        if (BYTESIZE == sizeof(WarpableColoredVoxel32f)) {   \
            using voxel_t = ColoredVoxel32f;                 \
            return __VA_ARGS__();                            \
        } else if (BYTESIZE == sizeof(WarpableColoredVoxel16i)) {    \
            using voxel_t = ColoredVoxel16i;                 \
            return __VA_ARGS__();                            \
        } else if (BYTESIZE == sizeof(WarpableVoxel32f)) {   \
            using voxel_t = Voxel32f;                        \
            return __VA_ARGS__();                            \
        } else if (BYTESIZE == sizeof(WarpableAnchoredVoxel32f)) {   \
            using voxel_t = Voxel32f;                        \
            return __VA_ARGS__();                            \
        } else {                                             \
            utility::LogError("Unsupported voxel bytesize"); \
        }                                                    \
    }()

namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {

/// 8-byte voxel structure.
/// Smallest struct we can get. float tsdf + uint16_t weight also requires
/// 8-bytes for alignement, so not implemented anyway.
struct WarpableVoxel32f : public struct Voxel32f{
	static bool HasAnchors() { return false; }
	static bool HasColor() { return Voxel32f::HasColor(); }
	OPEN3D_HOST_DEVICE void UpdateAnchors(int* anchor_indices, float* anchor_weights, int anchor_count){}
};

/// 12-byte voxel structure.
/// uint16_t for colors and weights, sacrifices minor accuracy but saves memory.
/// Basically, kColorFactor=255.0 extends the range of the uint8_t input color
/// to the range of uint16_t where weight average is computed. In practice, it
/// preserves most of the color details.
struct WarpableColoredVoxel16i : public struct ColoredVoxel16i{
	static bool HasAnchors() { return false; }
	static bool HasColor() { return ColoredVoxel16i::HasColor(); }
	OPEN3D_HOST_DEVICE void UpdateAnchors(int* anchor_indices, float* anchor_weights, int anchor_count){}
};

/// 20-byte voxel structure.
/// Float for colors and weights, accurate but memory-consuming.
struct WarpableColoredVoxel32f : public struct ColoredVoxel32f{
	static bool HasAnchors() { return false; }
	static bool HasColor() { return ColoredVoxel32f::HasColor(); }
	OPEN3D_HOST_DEVICE void UpdateAnchors(int* anchor_indices, float* anchor_weights, int anchor_count){}
};

template<int anchor_count>
struct WarpableAnchoredVoxel32f : public struct Voxel32f{
	static bool HasAnchors() { return false; }
	static bool HasColor() { return Voxel32f::HasColor(); }
	OPEN3D_HOST_DEVICE void UpdateAnchors(int* anchor_indices, float* anchor_weights, int anchor_count){}
};


} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt