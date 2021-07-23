//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/23/21.
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

#include <stdexcept>
#include <pybind11/numpy.h>

#include "image_view.h"
#include "pixel.h"

namespace py = pybind11;

namespace test {

// region ==================================== assert_is_image =============================================

/*!
	ensures
		- returns true if and only if the given python numpy array can reasonably be
		  interpreted as an image containing pixel_type pixels.
!*/
template <typename pixel_type>
bool is_image (const py::array& img){
	using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
	const size_t expected_channels = pixel_traits<pixel_type>::num;

	const bool has_correct_number_of_dims = (img.ndim()==2 && expected_channels==1) ||
	                                        (img.ndim()==3 && img.shape(2)==expected_channels);

	return img.dtype().kind() == py::dtype::of<basic_pixel_type>().kind() &&
	       img.itemsize() == sizeof(basic_pixel_type) &&
	       has_correct_number_of_dims;
}

template <typename pixel_type>
void assert_correct_num_channels_in_image (const py::array& img){
	const size_t expected_channels = pixel_traits<pixel_type>::num;
	if (expected_channels == 1){
		if (!(img.ndim() == 2 || (img.ndim()==3&&img.shape(2)==1))){
			throw std::runtime_error("Expected a 2D numpy array, but instead got one with " + std::to_string(img.ndim()) + " dimensions.");
		}
	}else{
		if (img.ndim() != 3){
			throw std::runtime_error("Expected a numpy array with 3 dimensions, but instead got one with " + std::to_string(img.ndim()) + " dimensions.");
		} else if (img.shape(2) != expected_channels) {
			if (pixel_traits<pixel_type>::rgb){
				throw std::runtime_error("Expected a RGB image with " + std::to_string(expected_channels) + " channels but got an image with " + std::to_string(img.shape(2)) + " channels.");
			} else {
				throw std::runtime_error("Expected an image with " + std::to_string(expected_channels) + " channels but got an image with " + std::to_string(img.shape(2)) + " channels.");
			}
		}
	}
}

template <typename pixel_type>
void assert_is_image ( const py::array& obj) {
	if (!is_image<pixel_type>(obj)) {
		assert_correct_num_channels_in_image<pixel_type>(obj);

		using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
		const char expected_type = py::dtype::of<basic_pixel_type>().kind();
		const char got_type = obj.dtype().kind();

		const size_t expected_size = sizeof(basic_pixel_type);
		const size_t got_size = obj.itemsize();

		auto toname = [](char type, size_t size) {
			if (type == 'i' && size == 1) return "int8";
			else if (type == 'i' && size == 2) return "int16";
			else if (type == 'i' && size == 4) return "int32";
			else if (type == 'i' && size == 8) return "int64";
			else if (type == 'u' && size == 1) return "uint8";
			else if (type == 'u' && size == 2) return "uint16";
			else if (type == 'u' && size == 4) return "uint32";
			else if (type == 'u' && size == 8) return "uint64";
			else if (type == 'f' && size == 4) return "float32";
			else if (type == 'd' && size == 8) return "float64";
			else throw std::runtime_error("unknown type");
		};

		throw std::runtime_error("Expected numpy array with elements of type " +
		                         std::string(toname(expected_type, expected_size)) + " but got " + toname(got_type, got_size) + ".");
	}
}

// endregion

// region ==================================== numpy_image class ============================================
template<typename pixel_type>
class numpy_image : public py::array_t<typename pixel_traits<pixel_type>::basic_pixel_type, py::array::c_style> {
	/*!
		REQUIREMENTS ON pixel_type
			- is a dlib pixel type, this just means that dlib::pixel_traits<pixel_type>
			  is defined.

		WHAT THIS OBJECT REPRESENTS
			This is an image object that implements dlib's generic image interface and
			is backed by a numpy array.  It therefore is easily interchanged with
			python since there is no copying.  It is functionally just a pybind11
			array_t object with the additional routines needed to conform to dlib's
			generic image API.  It also includes appropriate runtime checks to make
			sure that the numpy array is always typed and sized appropriately relative
			to the supplied pixel_type.
	!*/
public:

	numpy_image() = default;

	numpy_image(const py::array& img) : py::array_t<typename pixel_traits<pixel_type>::basic_pixel_type, py::array::c_style>(img) {
		assert_is_image<pixel_type>(img);
	}

	numpy_image(long rows, long cols) {
		set_size(rows, cols);
	}

	numpy_image(
			const py::object& img
	) : numpy_image(img.cast<py::array>()) {}

	numpy_image(
			const numpy_image& img
	) = default;

	numpy_image& operator=(
			const py::object& rhs
	) {
		*this = numpy_image(rhs);
		return *this;
	}

	numpy_image& operator=(
			const py::array_t<typename pixel_traits<pixel_type>::basic_pixel_type, py::array::c_style>& rhs
	) {
		*this = numpy_image(rhs);
		return *this;
	}

	numpy_image& operator=(
			const numpy_image& rhs
	) = default;

	// template<long NR, long NC>
	// numpy_image(
	// 		matrix <pixel_type, NR, NC>&& rhs
	// ) : numpy_image(convert_to_numpy(std::move(rhs))) {}

	// template<long NR, long NC>
	// numpy_image& operator=(
	// 		matrix <pixel_type, NR, NC>&& rhs
	// ) {
	// 	*this = numpy_image(rhs);
	// 	return *this;
	// }

	void set_size(size_t rows, size_t cols) {
		using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
		constexpr size_t channels = pixel_traits<pixel_type>::num;
		if (channels != 1)
			*this = py::array_t<basic_pixel_type, py::array::c_style>({rows, cols, channels});
		else
			*this = py::array_t<basic_pixel_type, py::array::c_style>({rows, cols});
	}

private:
	// static py::array_t<typename pixel_traits<pixel_type>::basic_pixel_type, py::array::c_style> convert_to_numpy(matrix <pixel_type>&& img) {
	// 	using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
	// 	const size_t dtype_size = sizeof(basic_pixel_type);
	// 	const auto rows = static_cast<const size_t>(num_rows(img));
	// 	const auto cols = static_cast<const size_t>(num_columns(img));
	// 	const size_t channels = pixel_traits<pixel_type>::num;
	// 	const size_t image_size = dtype_size * rows * cols * channels;
	//
	// 	std::unique_ptr < pixel_type[] > arr_ptr = img.steal_memory();
	// 	basic_pixel_type* arr = (basic_pixel_type*) arr_ptr.release();
	//
	// 	if (channels == 1) {
	// 		return pybind11::template array_t<basic_pixel_type, py::array::c_style>(
	// 				{rows, cols},                                                       // shape
	// 				{dtype_size * cols, dtype_size},                                      // strides
	// 				arr,                                                                // pointer
	// 				pybind11::capsule{arr, [](void* arr_p) { delete[] reinterpret_cast<basic_pixel_type*>(arr_p); }}
	// 		);
	// 	} else {
	// 		return pybind11::template array_t<basic_pixel_type, py::array::c_style>(
	// 				{rows, cols, channels},                                                     // shape
	// 				{dtype_size * cols * channels, dtype_size * channels, dtype_size},          // strides
	// 				arr,                                                                        // pointer
	// 				pybind11::capsule{arr, [](void* arr_p) { delete[] reinterpret_cast<basic_pixel_type*>(arr_p); }}
	// 		);
	// 	}
	// }

};
// endregion

// region ================================================= generic image functions ==================================================================

template <typename T>
struct image_traits<numpy_image<T>>
{
	typedef T pixel_type;
};

template <typename pixel_type>
void* image_data(numpy_image<pixel_type>& img)
{
	if (img.size()==0)
		return 0;

	assert_is_image<pixel_type>(img);
	return img.mutable_data(0);
}

template <typename pixel_type>
const void* image_data (const numpy_image<pixel_type>& img)
{
	if (img.size()==0)
		return 0;

	assert_is_image<pixel_type>(img);
	return img.data(0);
}

template <typename pixel_type>
long width_step (const numpy_image<pixel_type>& img)
{
	if (img.size()==0)
		return 0;

	assert_correct_num_channels_in_image<pixel_type>(img);
	using basic_pixel_type = typename pixel_traits<pixel_type>::basic_pixel_type;
	if (img.ndim()==3 && img.strides(2) != sizeof(basic_pixel_type))
		throw std::runtime_error("The stride of the 3rd dimension (the channel dimension) of the numpy array must be " + std::to_string(sizeof(basic_pixel_type)));
	if (img.strides(1) != sizeof(pixel_type))
		throw std::runtime_error("The stride of the 2nd dimension (the columns dimension) of the numpy array must be " + std::to_string(sizeof(pixel_type)));

	return img.strides(0);
}

template <typename pixel_type>
long num_rows(const numpy_image<pixel_type>& img)
{
	if (img.size()==0)
		return 0;

	assert_correct_num_channels_in_image<pixel_type>(img);
	return img.shape(0);
}

template <typename pixel_type>
long num_columns(const numpy_image<pixel_type>& img)
{
	if (img.size()==0)
		return 0;

	assert_correct_num_channels_in_image<pixel_type>(img);
	return img.shape(1);
}

template <typename pixel_type>
void set_image_size(numpy_image<pixel_type>& img, size_t rows, size_t cols)
{
	img.set_size(rows, cols);
}

// endregion
} // namespace test