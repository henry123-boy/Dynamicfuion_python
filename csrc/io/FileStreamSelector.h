//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 8/11/22.
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

#include <string>
#include <zstr.hpp>
#include <open3d/utility/Logging.h>

namespace nnrt::io {

template<typename TObjectToWrite>
void WriteObject(const std::string& path, const TObjectToWrite& object, bool compressed) {
	if (compressed) {
		zstr::ofstream ofstream(path, std::ios::binary | std::ios::out);
		if(!ofstream.is_open()){
			open3d::utility::LogError("Failed to open file at \"{}\" for writing.", path);
		}
		ofstream << object;
	} else {
		std::ofstream ofstream(path, std::ios::binary | std::ios::out);
		if(!ofstream.is_open()){
			open3d::utility::LogError("Failed to open file at \"{}\" for writing.", path);
		}
		ofstream << object;
	}
}

template<typename TObjectToRead>
void ReadObject(const std::string& path, TObjectToRead& object, bool compressed) {
	if (compressed) {
		zstr::ifstream ifstream(path, std::ios::binary | std::ios::out);
		if(!ifstream.is_open()){
			open3d::utility::LogError("Failed to open file at \"{}\" for reading.", path);
		}
		ifstream >> object;
	} else {
		std::ifstream ifstream(path, std::ios::binary | std::ios::out);
		if(!ifstream.is_open()){
			open3d::utility::LogError("Failed to open file at \"{}\" for reading.", path);
		}
		ifstream >> object;
	}
}

template<typename TObjectToRead>
TObjectToRead ReadObject(const std::string& path, bool compressed){
	TObjectToRead object;
	ReadObject(path, object, compressed);
	return object;
}

} // namespace nnrt::io
