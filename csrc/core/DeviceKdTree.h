//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/24/21.
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

namespace nnrt::core {

template<typename TCoordinate, int TDimensionCount>
class DeviceKdTree{
	struct Node{
		NNRT_DEVICE_WHEN_CUDACC Node(int64_t index) : index(index), right(nullptr), left(nullptr) {};
		int64_t index;
		Node* right;
		Node* left;
		float distance;
	};
	virtual private
public:
	virtual  ~DeviceKdTree();
	virtual  bool Insert(TCoordinate*, int count);
	virtual NNRT_DEVICE_WHEN_CUDACC void FindKNearestTo(TCoordinate*, int k);
	TCoordinate* indexed_data;
};



} // namespace nnrt::core