//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/28/21.
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
template<typename TKey, typename TValue>
struct KeyValuePair {
	TKey key;
	TValue value;
};

template<typename TKey, typename TValue>
struct DistanceIndexPair {
	union{
		TKey key;
		TKey distance;
	};
	union{
		TValue value;
		TValue index;
	};
};

template<typename TKey, typename TValue>
struct DistancePointPair {
	union{
		TKey key;
		TKey distance;
	};
	union{
		TValue value;
		TValue point;
	};
};

template<typename TKey, typename TValue>
struct DistanceNodePair {
	union{
		TKey key;
		TKey distance;
	};
	union{
		TValue value;
		TValue node;
	};
};
} // namespace nnrt::core
