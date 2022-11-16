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
// local
#include <open3d/utility/Logging.h>
#include "TensorIO.h"
#include "io/SizeVectorIO.h"
#include "io/DtypeIO.h"
#include "io/BlobIO.h"
#include "io/FileStreamSelector.h"

#include <zstr.hpp>

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::io {

std::ostream& operator<<(std::ostream& ostream, const open3d::core::Tensor& tensor) {
	ostream << tensor.GetShape();
	ostream << tensor.GetStrides();
	ostream << tensor.GetDtype();

	o3c::Device host("CPU:0");
	o3c::Tensor prepped_tensor = tensor.Contiguous().To(host);
	int64_t byte_size = prepped_tensor.GetDtype().ByteSize() * prepped_tensor.GetShape().NumElements();
	ostream << std::make_pair(prepped_tensor.GetBlob(), byte_size);
	return ostream;
}

std::istream& operator>>(std::istream& istream, open3d::core::Tensor& tensor) {
	o3c::SizeVector shape;
	istream >> shape;
	o3c::SizeVector strides;
	istream >> strides;
	o3c::Dtype dtype;
	istream >> dtype;

	if(istream.bad()){
		utility::LogError("Failure reading from istream.");
	}

	auto blob_and_bytesize = std::make_pair(std::shared_ptr<o3c::Blob>(nullptr), (int64_t)0L);
	istream >> blob_and_bytesize;
	// in the contiguous tensor case, the tensor pointer coincides with the blob pointer.
	tensor = o3c::Tensor(shape, strides, blob_and_bytesize.first->GetDataPtr(), dtype, blob_and_bytesize.first);
	return istream;
}

void WriteTensor(const std::string& path, const open3d::core::Tensor& tensor, bool compressed) {
	WriteObject(path, tensor, compressed);
}

void ReadTensor(const std::string& path, open3d::core::Tensor& tensor, bool compressed) {
	ReadObject(path, tensor, compressed);
}

open3d::core::Tensor ReadTensor(const std::string& path, bool compressed) {
	return ReadObject<o3c::Tensor>(path, compressed);
}


} // namespace nnrt::io



