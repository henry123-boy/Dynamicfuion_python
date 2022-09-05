#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 9/5/22.
#  Copyright (c) 2022 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
from pathlib import Path
from collections import namedtuple

import pytest

TestDataPaths = namedtuple("TestDataPaths", "images intrinsics meshes tensors")


@pytest.fixture(scope="package")
def path() -> TestDataPaths:
    test_path = Path(__file__).parent.resolve()

    test_data_path = test_path / "test_data"
    image_test_data_path = test_data_path / "images"
    intrinsics_test_data_path = test_data_path / "intrinsics"
    mesh_test_data_path = test_data_path / "meshes"
    tensor_test_data_path = test_data_path / "tensors"
    return TestDataPaths(image_test_data_path, intrinsics_test_data_path, mesh_test_data_path, tensor_test_data_path)
