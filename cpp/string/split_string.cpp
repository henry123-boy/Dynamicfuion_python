//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/22/21.
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

#include <algorithm>
#include <iostream>
#include <string_view>
#include <vector>

#include "split_string.h"

namespace nnrt::string{

std::vector<std::string_view> split(std::string_view buffer,
                                    const std::string_view delimiter) {
	std::vector<std::string_view> ret{};
	std::decay_t<decltype(std::string_view::npos)> pos{};
	while ((pos = buffer.find(delimiter)) != std::string_view::npos) {
		const auto match = buffer.substr(0, pos);
		if (!match.empty()) ret.push_back(match);
		buffer = buffer.substr(pos + delimiter.size());
	}
	if (!buffer.empty()) ret.push_back(buffer);
	return ret;
}

} // namespace nnrt