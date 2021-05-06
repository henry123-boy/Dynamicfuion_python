//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/23/21.
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

#include "backward.hpp"


#define print_stack_trace(stack_depth)\
    backward::StackTrace st;\
    st.load_here(stack_depth);\
    backward::TraceResolver tr;\
    tr.load_stacktrace(st);\
    for (size_t i = 2; i < st.size(); ++i) {\
        backward::ResolvedTrace trace = tr.resolve(st[i]);\
        std::cout << "#" << i-2\
        << " " << trace.object_filename\
        << " " << trace.object_function\
        << " [" << trace.addr << "]"\
        << std::endl << "file: " << trace.source.filename\
        << " line: " << trace.source.line\
        << " column: " << trace.source.col\
        << std::endl;\
    }\
    static_assert(true, "")