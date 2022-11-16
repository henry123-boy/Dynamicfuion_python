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
#include "DtypeIO.h"

namespace utility = open3d::utility;
namespace nnrt::io {

std::ostream& operator<<(std::ostream& ostream, const open3d::core::Dtype& dtype) {
	open3d::core::Dtype::DtypeCode code = dtype.GetDtypeCode();
	ostream.write(reinterpret_cast<const char*>(&code), sizeof(open3d::core::Dtype::DtypeCode));
	int64_t byte_size = dtype.ByteSize();
	ostream.write(reinterpret_cast<const char*>(&byte_size), sizeof(int64_t));
	std::string dtype_name = dtype.ToString();
	auto name_size = static_cast<uint8_t>(dtype_name.size());
	ostream.write(reinterpret_cast<const char*>(&name_size), sizeof(uint8_t));
	ostream.write(reinterpret_cast<const char*>(dtype_name.c_str()), name_size);
	return ostream;
}

std::istream& operator>>(std::istream& istream, open3d::core::Dtype& dtype) {
	open3d::core::Dtype::DtypeCode code;
	istream.read(reinterpret_cast<char*>(&code), sizeof(open3d::core::Dtype::DtypeCode));
	int64_t byte_size;
	istream.read(reinterpret_cast<char*>(&byte_size), sizeof(int64_t));
	uint8_t name_size;
	istream.read(reinterpret_cast<char*>(&name_size), sizeof(uint8_t));
	std::string dtype_name;
	dtype_name.resize(name_size);
	istream.read(reinterpret_cast<char*>(&dtype_name[0]), name_size);
	if(istream.bad()){
		utility::LogError("Failure reading from istream.");
	}
	dtype = open3d::core::Dtype(code, byte_size, dtype_name);
	return istream;
}
} // namespace nnrt::io