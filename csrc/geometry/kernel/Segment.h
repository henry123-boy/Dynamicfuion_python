//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/8/22.
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
#pragma once

#include <Eigen/Dense>
#include "core/PlatformIndependence.h"


namespace nnrt::geometry::kernel {

class Segment {
public:
	NNRT_DEVICE_WHEN_CUDACC
	Segment() :
			origin(Eigen::Vector3f::Zero()), vector_to_destination(Eigen::Vector3f::Zero()),
			inverseDirection(Eigen::Vector3f::Zero()),
			sign{(inverseDirection.x() < 0), (inverseDirection.y() < 0), (inverseDirection.z() < 0)} {}

	NNRT_DEVICE_WHEN_CUDACC
	Segment(const Eigen::Vector3f& startPoint, const Eigen::Vector3f& endPoint) :
			origin(startPoint), vector_to_destination(endPoint - startPoint),
			inverseDirection(Eigen::Vector3f(Eigen::Vector3f::Constant(1.f)).cwiseQuotient(vector_to_destination)),
			sign{(inverseDirection.x() < 0), (inverseDirection.y() < 0), (inverseDirection.z() < 0)} {}

	template <typename TVector3f>
	NNRT_DEVICE_WHEN_CUDACC
	bool IntersectsAxisAlignedBox(TVector3f box_min, TVector3f box_max) {
		float t_min, t_max, ty_min, ty_max, tz_min, tz_max;
		Eigen::Vector3f bounds[] = {box_min, box_max};

		t_min = (bounds[sign[0]].x() - origin.x()) * inverseDirection.x();
		t_max = (bounds[1 - sign[0]].x() - origin.x()) * inverseDirection.x();
		ty_min = (bounds[sign[1]].y() - origin.y()) * inverseDirection.y();
		ty_max = (bounds[1 - sign[1]].y() - origin.y()) * inverseDirection.y();

		if ((t_min > ty_max) || (ty_min > t_max)) {
			return false;
		}
		if (ty_min > t_min) {
			t_min = ty_min;
		}
		if (ty_max < t_max) {
			t_max = ty_max;
		}

		tz_min = (bounds[sign[2]].z() - origin.z()) * inverseDirection.z();
		tz_max = (bounds[1 - sign[2]].z() - origin.z()) * inverseDirection.z();

		if ((t_min > tz_max) || (tz_min > t_max)) {
			return false;
		}
		if (tz_min > t_min) {
			t_min = tz_min;
		}
		if (tz_max < t_max) {
			t_max = tz_max;
		}
		return !(t_max < 0.0f || t_min > 1.0f);
	}

	NNRT_DEVICE_WHEN_CUDACC
	float length() const {
		return vector_to_destination.norm();
	}

	NNRT_DEVICE_WHEN_CUDACC
	Eigen::Vector3f destination() const {
		return origin + vector_to_destination;
	}

	Eigen::Vector3f origin, vector_to_destination;
	Eigen::Vector3f inverseDirection;
	int sign[3];
};

} // nnrt::geometry::kernel
