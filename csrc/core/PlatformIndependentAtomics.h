//  ================================================================
//  Created by Gregory Kramida on 10/10/19.
//  Copyright (c) 2019 Gregory Kramida
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
#if defined(__CUDACC__)
__device__ __forceinline__ float atomicMin (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
}

__device__ __forceinline__ float atomicMax (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

#else
#include <atomic>

template<typename T>
inline
T atomicMax_CPU(std::atomic<T>& variable, T value) {
	auto current = variable.load();
	while (current < value && !variable.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed));
	return current;
}

template<typename T>
inline
T atomicMin_CPU(std::atomic<T>& variable, T value) {
	auto current = variable.load();
	while (current > value && !variable.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed));
	return current;
}


template<typename T>
inline
T atomicAdd_CPU(std::atomic<T>& variable, T addend) {
	auto current = variable.load();
	while (!variable.compare_exchange_weak(current, current + addend, std::memory_order_relaxed, std::memory_order_relaxed));
	return current;
}



template<>
inline
int atomicAdd_CPU<int>(std::atomic<int>& variable, int addend){
	return variable.fetch_add(addend, std::memory_order_relaxed);
}

template<>
inline
int64_t atomicAdd_CPU<int64_t>(std::atomic<int64_t>& variable, int64_t addend){
	return variable.fetch_add(addend, std::memory_order_relaxed);
}

template<>
inline
unsigned int atomicAdd_CPU<unsigned int>(std::atomic<unsigned int>& variable, unsigned int addend){
	return variable.fetch_add(addend, std::memory_order_relaxed);
}

template<typename T>
inline
T atomicSub_CPU(std::atomic<T>& variable, T subtracted) {
	auto current = variable.load();
	while (!variable.compare_exchange_weak(current, current - subtracted, std::memory_order_relaxed, std::memory_order_relaxed));
	return current;
}

template<>
inline
int atomicSub_CPU<int>(std::atomic<int>& variable, int subtracted){
	return variable.fetch_sub(subtracted, std::memory_order_relaxed);
}

template <typename T>
inline void initializeAtomic_CPU(std::atomic<T>& var, T value){
	var.store(value);
}



template <typename T>
inline bool CompareExchange_CPU(std::atomic<T>& variable, T expected, T desired){
	return variable.compare_exchange_weak(expected, desired);
}

#endif

#if defined(__CUDACC__)
#include <open3d/core/CUDAUtils.h>
template <typename T>
inline T GetDataCPU(T* var){
	T var_val;
	OPEN3D_CUDA_CHECK(cudaMemcpy(&var_val, var, sizeof(T), cudaMemcpyDeviceToHost));
	return var_val;
}

template <typename T>
inline void SetDataCPU(T* var, T var_val){
	OPEN3D_CUDA_CHECK(cudaMemcpy(var, &var_val, sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
__device__
inline bool CompareExchange_CUDA(T* var, T expected, T desired){
	return expected == atomicCAS(var, expected, desired);
}

#endif

#if defined(__CUDACC__)
// for CUDA device code
#define NNRT_DECLARE_ATOMIC(type, name) type* name = nullptr
#define NNRT_DECLARE_ATOMIC_INT(name)  int* name = nullptr
#define NNRT_DECLARE_ATOMIC_UINT(name) unsigned int* name = nullptr
#define NNRT_DECLARE_ATOMIC_FLOAT(name) float* name = nullptr
#define NNRT_DECLARE_ATOMIC_DOUBLE(name) double* name = nullptr
#define NNRT_CLEAN_UP_ATOMIC(name) OPEN3D_CUDA_CHECK (cudaFree(name))
#define NNRT_SET_ATOMIC_VALUE_CPU(name, value) SetDataCPU(name, value)
#define NNRT_GET_ATOMIC_VALUE(name) (* name)
#define NNRT_GET_ATOMIC_VALUE_CPU(name) GetDataCPU( name )
#define NNRT_ATOMIC_ARGUMENT(type) type *
#define NNRT_ATOMIC_FLOAT_ARGUMENT_TYPE float*
#define NNRT_ATOMIC_INT_ARGUMENT_TYPE int*

#define NNRT_INITIALIZE_ATOMIC(type, var, value) \
{ \
type val = value; \
OPEN3D_CUDA_CHECK(cudaMalloc((void**)&var, sizeof( type ))); \
OPEN3D_CUDA_CHECK(cudaMemcpy(var, &val, sizeof( type ), cudaMemcpyHostToDevice)); \
}
#else
#define NNRT_DECLARE_ATOMIC(type, name) std::atomic< type > name = {0}
#define NNRT_DECLARE_ATOMIC_INT(name)  std::atomic<int> name = {0}
#define NNRT_DECLARE_ATOMIC_UINT(name)  std::atomic<unsigned int> name = {0}
#define NNRT_DECLARE_ATOMIC_FLOAT(name) std::atomic<float> name = {0}
#define NNRT_DECLARE_ATOMIC_DOUBLE(name) std::atomic<double> name = {0}
#define NNRT_CLEAN_UP_ATOMIC(name) ;
#define NNRT_SET_ATOMIC_VALUE_CPU(name, value) (name .store(value))
#define NNRT_GET_ATOMIC_VALUE(name) (name .load())
#define NNRT_GET_ATOMIC_VALUE_CPU(name) NNRT_GET_ATOMIC_VALUE(name)
#define NNRT_ATOMIC_ARGUMENT(type) std::atomic< type >&
#define NNRT_ATOMIC_FLOAT_ARGUMENT_TYPE std::atomic<float>&
#define NNRT_ATOMIC_INT_ARGUMENT_TYPE std::atomic<int>&

#define NNRT_INITIALIZE_ATOMIC(type, var, value) \
initializeAtomic_CPU< type > (var, value)
#endif


//region CAS-type operations on atomics
#if defined(__CUDACC__)
#define NNRT_ATOMIC_ADD(name, value) atomicAdd( name, value )
#define NNRT_ATOMIC_SUB(name, value) atomicSub( name, value )
#define NNRT_ATOMIC_MAX(name, value) atomicMax( name, value )
#define NNRT_ATOMIC_MIN(name, value) atomicMin( name, value )
#define NNRT_ATOMIC_CE(name, expected, desired) CompareExchange_CUDA( name, expected, desired )
#else
#define NNRT_ATOMIC_ADD(name, value) atomicAdd_CPU( name, value )
#define NNRT_ATOMIC_SUB(name, value) atomicSub_CPU( name, value )
#define NNRT_ATOMIC_MAX(name, value) atomicMax_CPU( name, value )
#define NNRT_ATOMIC_MIN(name, value) atomicMin_CPU( name, value )
#define NNRT_ATOMIC_CE(name, expected, desired) CompareExchange_CPU( name, expected, desired )
#endif
//endregion


