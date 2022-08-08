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
#include <open3d/utility/Logging.h>
#include "BlobIO.h"
#include "io/DeviceIO.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace nnrt::io {

std::ostream& operator<<(std::ostream& ostream, const std::pair<const std::shared_ptr<open3d::core::Blob>&, int64_t>& blob_and_bytesize) {
	auto blob = blob_and_bytesize.first;
	o3c::Device host("CPU:0");
	if(blob->GetDevice() != host){
		o3u::LogError("Blob ostream output for blobs with device {} not supported (only CPU:0 device is supported).", blob->GetDevice().ToString());
	}
	auto byte_size = blob_and_bytesize.second;
	ostream.write(reinterpret_cast<const char*>(&byte_size), sizeof(int64_t));
	ostream << blob->GetDevice();
	ostream.write(reinterpret_cast<const char*>(blob->GetDataPtr()), byte_size);
	return ostream;
}

std::istream& operator>>(std::istream& istream, std::pair<std::shared_ptr<open3d::core::Blob>, int64_t>& blob_and_bytesize) {

	istream.read(reinterpret_cast<char*>(&blob_and_bytesize.second), sizeof(int64_t));
	auto byte_size = blob_and_bytesize.second;
	o3c::Device device;
	istream >> device;
	o3c::Device host("CPU:0");
	if(device != host){
		o3u::LogError("Blob istream input for blobs with device {} not supported (only CPU:0 device is supported).", device.ToString());
	}
	blob_and_bytesize.first = std::make_shared<o3c::Blob>(byte_size, device);
	istream.read(reinterpret_cast<char*>(blob_and_bytesize.first->GetDataPtr()), byte_size);
	istream >> device;
	return istream;
}
} // namespace nnrt::io