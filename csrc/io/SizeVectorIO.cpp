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
// stdlib
#include <ostream>
#include <istream>
// local
#include "SizeVectorIO.h"
#include <open3d/utility/Logging.h>

namespace utility = open3d::utility;
namespace nnrt::io {

std::ostream& operator<<(std::ostream& ostream, const open3d::core::SizeVector& size_vector) {
	auto dimension_count = static_cast<uint32_t>(size_vector.size());
	ostream.write(reinterpret_cast<const char*>(&dimension_count), sizeof(uint32_t));
	for (const int64_t& element: size_vector) {
		ostream.write(reinterpret_cast<const char*>(&element), sizeof(int64_t));
	}
	if(ostream.bad()){
		utility::LogError("Could not write to output stream.");
	}
	return ostream;
}

std::istream& operator>>(std::istream& istream, open3d::core::SizeVector& size_vector) {
	uint32_t length;
	istream.read(reinterpret_cast<char*>(&length), sizeof(uint32_t));
	size_vector.clear();
	for (uint32_t i_element = 0; i_element < length; i_element++) {
		int64_t element;
		istream.read(reinterpret_cast<char*>(&element), sizeof(int64_t));
		size_vector.push_back(element);
	}
	return istream;
}

} // namespace nnrt::io