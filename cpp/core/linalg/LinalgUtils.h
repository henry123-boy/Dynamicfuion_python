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

#include <memory>
#include <string>

#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "core/linalg/LinalgHeadersCPU.h"
#include "core/linalg/LinalgHeadersCUDA.h"
#include "open3d/utility/Logging.h"

namespace nnrt::core {

#define DISPATCH_LINALG_DTYPE_TO_TEMPLATE(DTYPE, ...)            \
    [&] {                                                        \
        if (DTYPE == open3d::core::Float32) {                    \
            using scalar_t = float;                              \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Float64) {             \
            using scalar_t = double;                             \
            return __VA_ARGS__();                                \
        } else {                                                 \
            open3d::utility::LogError("Unsupported data type."); \
        }                                                        \
    }()

inline void NNRT_LAPACK_CHECK(NNRT_CPU_LINALG_INT info,
                                const std::string& msg) {
    if (info < 0) {
        open3d::utility::LogError("{}: {}-th parameter is invalid.", msg, -info);
    } else if (info > 0) {
	    open3d::utility::LogError("{}: singular condition detected.", msg);
    }
}

#ifdef BUILD_CUDA_MODULE
inline void NNRT_CUBLAS_CHECK(cublasStatus_t status, const std::string& msg) {
    if (CUBLAS_STATUS_SUCCESS != status) {
	    open3d::utility::LogError("{}", msg);
    }
}

inline void NNRT_CUSOLVER_CHECK(cusolverStatus_t status,
                                  const std::string& msg) {
    if (CUSOLVER_STATUS_SUCCESS != status) {
	    open3d::utility::LogError("{}", msg);
    }
}

inline void NNRT_CUSOLVER_CHECK_WITH_DINFO(cusolverStatus_t status,
                                             const std::string& msg,
                                             int* dinfo,
                                             const open3d::core::Device& device) {
    int hinfo;
    open3d::core::MemoryManager::MemcpyToHost(&hinfo, dinfo, device, sizeof(int));
    if (status != CUSOLVER_STATUS_SUCCESS || hinfo != 0) {
        if (hinfo < 0) {
            open3d::utility::LogError("{}: {}-th parameter is invalid.", msg, -hinfo);
        } else if (hinfo > 0) {
	        open3d::utility::LogError("{}: singular condition detected.", msg);
        } else {
	        open3d::utility::LogError("{}: status error code = {}.", msg, status);
        }
    }
}

inline void NNRT_CUSOLVER_CHECK_WITH_MULTIPLE_DINFO(cusolverStatus_t status,
                                           const std::string& msg,
                                           int* dinfo,
										   int info_count,
                                           const open3d::core::Device& device) {
	std::vector<int> hinfo(info_count);
	open3d::core::MemoryManager::MemcpyToHost(hinfo.data(), dinfo, device, info_count * sizeof(int));
	int i_info = 0;
	for (auto hinfo_instance : hinfo) {
		if (status != CUSOLVER_STATUS_SUCCESS || hinfo_instance != 0) {
			if (hinfo_instance < 0) {
				open3d::utility::LogError("{}: {}-th parameter is invalid.", msg, i_info);
			} else if (hinfo_instance > 0) {
				open3d::utility::LogError("{}: leading submatrix of order {} of {}-th matrix is not positive definite.", msg, hinfo, i_info);
			} else {
				open3d::utility::LogError("{}: status error code = {}.", msg, status);
			}
		}
		i_info++;
	}
}

inline void NNRT_MAGMA_CHECK(magma_int_t status,  const std::string& msg) {
	if (MAGMA_SUCCESS != status) {
		open3d::utility::LogError("{}: MAGMA error {}", msg, magma_strerror(status));
	}
}

class CuSolverContext {
public:
    static std::shared_ptr<CuSolverContext> GetInstance();
    CuSolverContext();
    ~CuSolverContext();

    cusolverDnHandle_t& GetHandle() { return handle_; }

private:
    cusolverDnHandle_t handle_;

    static std::shared_ptr<CuSolverContext> instance_;
};

class CuBLASContext {
public:
    static std::shared_ptr<CuBLASContext> GetInstance();

    CuBLASContext();
    ~CuBLASContext();

    cublasHandle_t& GetHandle() { return handle_; }

private:
    cublasHandle_t handle_;

    static std::shared_ptr<CuBLASContext> instance_;
};
#endif

template <typename T>
struct cusolver_traits;
template <>
struct cusolver_traits <float>
{
	static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
};
template <>
struct cusolver_traits <double>
{
	static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
};
}  // namespace nnrt::core
