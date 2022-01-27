//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/4/22.
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
#include <vector>
#include <string>
#include <sstream>
#include "join_string_separator.h"

namespace nnrt::string {
template <typename Iterator>
std::string join(Iterator begin, Iterator end, const char* separator){
	std::ostringstream o;
	if(begin != end){
		o << *begin++;
		for(;begin != end; ++begin){
			o  << separator << *begin;
		}
	}
	return o.str();
}
template <typename Container>
std::string join(Container const& c, const char* separator) // can pass array as reference, too
{
	using std::begin;
	using std::end;
	return join(begin(c), end(c), separator);
	// not using std::... directly:
	// there might be a non-std overload that wouldn't be found if we did
}

std::string join(const std::vector<std::string>& strings, const char* separator) {
	return join<std::vector<std::string>>(strings, separator);
}
} // namespace nnrt::string
