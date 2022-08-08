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
#include "DtypeIO.h"

namespace nnrt::io {

std::ostream& operator<<(std::ostream& ostream, const open3d::core::Dtype& dtype) {
	open3d::core::Dtype::DtypeCode code = dtype.GetDtypeCode();
	ostream.write(reinterpret_cast<const char*>(&code), sizeof(open3d::core::Dtype::DtypeCode));
	int64_t byte_size = dtype.ByteSize();
	ostream.write(reinterpret_cast<const char*>(&byte_size), sizeof(int64_t));
	ostream << dtype.ToString();
	return ostream;
}

std::istream& operator>>(std::istream& istream, open3d::core::Dtype& dtype) {
	open3d::core::Dtype::DtypeCode code;
	istream.read(reinterpret_cast<char*>(&code), sizeof(open3d::core::Dtype::DtypeCode));
	int64_t byte_size;
	istream.read(reinterpret_cast<char*>(&byte_size), sizeof(int64_t));
	std::string name;
	istream >> name;
	dtype = open3d::core::Dtype(code, byte_size, name);
	return istream;
}
} // namespace nnrt::io