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
#pragma once

#include <pybind11/pybind11.h>
#include "3rd_party/magic_enum.hpp"
#include "string/split_string.h"

namespace nnrt {


// credit: based on https://github.com/pybind/pybind11/issues/1759#issuecomment-691553696 by YannikJadoul & bstaletic
template<typename EnumType, typename... Extra>
pybind11::enum_<EnumType> export_enum(const pybind11::handle& scope, Extra&& ... extra) {
	std::string_view unqualified_enum_name = nnrt::string::split(magic_enum::enum_type_name<EnumType>(), "::").back();
	pybind11::enum_<EnumType> enum_type(scope, unqualified_enum_name.data(), std::forward<Extra>(extra)...);
	for (const auto &[value, name]: magic_enum::enum_entries<EnumType>()) {
		enum_type.value(name.data(), value);
	}
	return enum_type;
}


} // namespace nnrt