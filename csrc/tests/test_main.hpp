//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/1/21.
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

#define CATCH_CONFIG_RUNNER

#include <catch2/catch_session.hpp>
#include <pybind11/pybind11.h>


// Necessary to start running the test(s) as a python program
int main(int argc, char* argv[]) {
#ifdef NNRT_TEST_USE_PYTHON
	wchar_t* program = Py_DecodeLocale(argv[0], nullptr);
	if (program == nullptr) {
		fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
		exit(1);
	}
	Py_SetProgramName(program);  /* optional but recommended */
	Py_Initialize();
#endif
	int result = Catch::Session().run(argc, argv);
#ifdef NNRT_TEST_USE_PYTHON
	if (Py_FinalizeEx() < 0) {
		exit(120);
	}
	PyMem_RawFree(program);
#endif
	return (result < 0xff ? result : 0xff);
}