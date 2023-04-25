// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
#pragma once

// third-party includes
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>

#include <open3d/utility/Logging.h>


// clang-format off
#if TBB_INTERFACE_VERSION >= 10000
#ifdef OPEN3D_USE_ONEAPI_PACKAGES
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#else
// Check if the C++ standard library implements parallel algorithms
// and use this over parallelstl to avoid conflicts.
// Clang does not implement it so far, so checking for C++17 is not sufficient.
#ifdef __cpp_lib_parallel_algorithm

#include <execution>
#include <numeric>

#else
#include <pstl/execution>
#include <pstl/numeric>
// parallelstl incorrectly assumes MSVC to unconditionally implement
// parallel algorithms even if __cpp_lib_parallel_algorithm is not
// defined. So manually include the header which pulls all
// "pstl::execution" definitions into the "std" namespace.
#if __PSTL_CPP17_EXECUTION_POLICIES_PRESENT
#include <pstl/internal/glue_execution_defs.h>
#endif
#endif
#endif
#endif

//__DEBUG
#define BUILD_CUDA_MODULE

#ifdef BUILD_CUDA_MODULE
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <open3d/core/Device.h>

#endif

//NOTE: Code adapted from open3d/utility/ParallelScan.h (Open3D release 0.16, https://github.com/isl-org/Open3D)

// clang-format on
namespace nnrt::core::kernel {

namespace {
template<class Tin, class Tout>
class ExclusiveScanSumBody {
	Tout sum;
	const Tin* in;
	Tout* const out;

public:
	ExclusiveScanSumBody(Tout* out_, const Tin* in_) : sum(0), in(in_), out(out_) {}

	Tout get_sum() const { return sum; }

	template<class Tag>
	void operator()(const tbb::blocked_range<size_t>& range, Tag) {
		Tout temp = sum;
		if (Tag::is_final_scan()) {
			for (size_t i = range.begin(); i < range.end() - 1; ++i) {
				out[i] = temp;
			}
		} else {
			for (size_t i = range.begin(); i < range.end(); ++i) {
				temp = temp + in[i];
			}
		}

		sum = temp;
	}

	ExclusiveScanSumBody(ExclusiveScanSumBody& b, tbb::split) : sum(0), in(b.in), out(b.out + 1) {}

	void reverse_join(ExclusiveScanSumBody& a) { sum = a.sum + sum; }

	void assign(ExclusiveScanSumBody& b) { sum = b.sum; }
};
} // namespace

template<class Tin, class Tout>
void ExclusivePrefixSumCPU(const Tin* first, const Tin* last, Tout* out) {
	out[0] = 0;
#if TBB_INTERFACE_VERSION >= 10000
	// use parallelstl if we have TBB 2018 or later
#ifdef NNRT_USE_ONEAPI_PACKAGES
	std::exclusive_scan(oneapi::dpl::execution::par_unseq, first, last, out);
#else
	std::exclusive_scan(std::execution::par_unseq, first, last, out);
#endif
#else
	ScanSumBody<Tin, Tout> body(out, first);
	size_t n = std::distance(first, last);
	tbb::parallel_scan(tbb::blocked_range<size_t>(0, n), body);
#endif
}

#ifdef BUILD_CUDA_MODULE
template<class Tin, class Tout>
void ExclusivePrefixSumCUDA(const Tin* first, const Tin* last, Tout* out, const open3d::core::Device& device){
	cudaSetDevice(device.GetID());
	thrust::exclusive_scan(thrust::device, first, last, out);
}
#else
template<class Tin, class Tout>
void ExclusivePrefixSumCUDA(const Tin* first, const Tin* last, Tout* out){
	open3d::utility::LogError("CUDA BUILD is OFF, there must be an error in the code using ExclusivePrefixSumCUDA.")
}
#endif


}  // namespace nnrt::core::kernel