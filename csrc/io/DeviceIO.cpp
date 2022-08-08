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
#include "DeviceIO.h"
namespace nnrt::io {

std::ostream& operator<<(std::ostream& ostream, const open3d::core::Device& device) {
	open3d::core::Device::DeviceType device_type = device.GetType();
	ostream.write(reinterpret_cast<const char*>(&device_type), sizeof(open3d::core::Device::DeviceType));
	int device_id = device.GetID();
	ostream.write(reinterpret_cast<const char*>(&device_id), sizeof(int));
	return ostream;
}

std::istream& operator>>(std::istream& istream, open3d::core::Device& device) {
	open3d::core::Device::DeviceType device_type;
	istream.read(reinterpret_cast<char*>(&device_type), sizeof(open3d::core::Device::DeviceType));
	int device_id;
	istream.read(reinterpret_cast<char*>(&device_id), sizeof(int));
	device = open3d::core::Device(device_type, device_id);
	return istream;
}
} // namespace nnrt::io