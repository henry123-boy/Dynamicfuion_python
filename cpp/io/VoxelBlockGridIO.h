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

namespace nnrt::geometry {
class VoxelBlockGrid;
class NonRigidSurfaceVoxelBlockGrid;
} // nnrt::geometry
namespace nnrt::io {
std::ostream& operator<<(std::ostream& ostream, const nnrt::geometry::VoxelBlockGrid& voxel_block_grid);
std::istream& operator>>(std::istream& istream, nnrt::geometry::VoxelBlockGrid& voxel_block_grid);
void WriteVoxelBlockGrid(const std::string& path, const nnrt::geometry::VoxelBlockGrid& voxel_block_grid, bool compressed = true);
void ReadVoxelBlockGrid(const std::string& path, nnrt::geometry::VoxelBlockGrid& voxel_block_grid, bool compressed = true);
nnrt::geometry::VoxelBlockGrid ReadVoxelBlockGrid(const std::string& path, bool compressed = true);
std::ostream& operator<<(std::ostream& ostream, const nnrt::geometry::NonRigidSurfaceVoxelBlockGrid& voxel_block_grid);
std::istream& operator>>(std::istream& istream, nnrt::geometry::NonRigidSurfaceVoxelBlockGrid& voxel_block_grid);
void WriteNonRigidSurfaceVoxelBlockGrid(const std::string& path, const nnrt::geometry::NonRigidSurfaceVoxelBlockGrid& voxel_block_grid,
										bool compressed = true);
void ReadNonRigidSurfaceVoxelBlockGrid(const std::string& path, nnrt::geometry::NonRigidSurfaceVoxelBlockGrid& voxel_block_grid,
									   bool compressed = true);
nnrt::geometry::NonRigidSurfaceVoxelBlockGrid ReadNonRigidSurfaceVoxelBlockGrid(const std::string& path, bool compressed = true);
} // nnrt::io