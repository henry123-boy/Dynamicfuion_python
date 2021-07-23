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


#include <limits>
#include <complex>

namespace test {

//region =================================== primitive datatype typedefs ========================


#ifdef __GNUC__
typedef unsigned long long uint64;
typedef long long int64;
#elif defined(__BORLANDC__)
typedef unsigned __int64 uint64;
	typedef __int64 int64;
#elif defined(_MSC_VER)
	typedef unsigned __int64 uint64;
	typedef __int64 int64;
#else
	typedef unsigned long long uint64;
	typedef long long int64;
#endif

typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned char uint8;

typedef short int16;
typedef int int32;
typedef char int8;


// make sure these types have the right sizes on this platform
static_assert(sizeof(uint8) == 1, "Failed assertion");
static_assert(sizeof(uint16) == 2, "Failed assertion");
static_assert(sizeof(uint32) == 4, "Failed assertion");
static_assert(sizeof(uint64) == 8, "Failed assertion");

static_assert(sizeof(int8) == 1, "Failed assertion");
static_assert(sizeof(int16) == 2, "Failed assertion");
static_assert(sizeof(int32) == 4, "Failed assertion");
static_assert(sizeof(int64) == 8, "Failed assertion");

template <typename T, size_t s = sizeof(T)>
struct unsigned_type;
template <typename T>
struct unsigned_type<T,1> { typedef uint8 type; };
template <typename T>
struct unsigned_type<T,2> { typedef uint16 type; };
template <typename T>
struct unsigned_type<T,4> { typedef uint32 type; };
template <typename T>
struct unsigned_type<T,8> { typedef uint64 type; };

//endregion =====================================================================================


template<typename T>
struct pixel_traits;


struct rgb_alpha_pixel {
	/*!
		WHAT THIS OBJECT REPRESENTS
			This is a simple struct that represents an RGB colored graphical pixel
			with an alpha channel.
	!*/

	rgb_alpha_pixel() {}

	rgb_alpha_pixel(
			unsigned char red_,
			unsigned char green_,
			unsigned char blue_,
			unsigned char alpha_
	) : red(red_), green(green_), blue(blue_), alpha(alpha_) {}

	unsigned char red;
	unsigned char green;
	unsigned char blue;
	unsigned char alpha;

	bool operator==(const rgb_alpha_pixel& that) const {
		return this->red == that.red
		       && this->green == that.green
		       && this->blue == that.blue
		       && this->alpha == that.alpha;
	}

	bool operator!=(const rgb_alpha_pixel& that) const {
		return !(*this == that);
	}

};


template<>
struct pixel_traits<rgb_alpha_pixel> {
	constexpr static bool rgb = false;
	constexpr static bool rgb_alpha = true;
	constexpr static bool grayscale = false;
	constexpr static bool hsi = false;
	constexpr static bool lab = false;
	constexpr static long num = 4;
	typedef unsigned char basic_pixel_type;

	static basic_pixel_type min() { return 0; }

	static basic_pixel_type max() { return 255; }

	constexpr static bool is_unsigned = true;
	constexpr static bool has_alpha = true;
};

struct rgb_pixel {
	/*!
		WHAT THIS OBJECT REPRESENTS
			This is a simple struct that represents an RGB colored graphical pixel.
	!*/

	rgb_pixel(
	) {}

	rgb_pixel(
			unsigned char red_,
			unsigned char green_,
			unsigned char blue_
	) : red(red_), green(green_), blue(blue_) {}

	unsigned char red;
	unsigned char green;
	unsigned char blue;

	bool operator==(const rgb_pixel& that) const {
		return this->red == that.red
		       && this->green == that.green
		       && this->blue == that.blue;
	}

	bool operator!=(const rgb_pixel& that) const {
		return !(*this == that);
	}

};
// ----------------------------------------------------------------------------------------

template<>
struct pixel_traits<rgb_pixel> {
	constexpr static bool rgb = true;
	constexpr static bool rgb_alpha = false;
	constexpr static bool grayscale = false;
	constexpr static bool hsi = false;
	constexpr static bool lab = false;
	enum {
		num = 3
	};
	typedef unsigned char basic_pixel_type;

	static basic_pixel_type min() { return 0; }

	static basic_pixel_type max() { return 255; }

	constexpr static bool is_unsigned = true;
	constexpr static bool has_alpha = false;
};

// ----------------------------------------------------------------------------------------

struct bgr_pixel {
	/*!
		WHAT THIS OBJECT REPRESENTS
			This is a simple struct that represents an BGR colored graphical pixel.
			(the reason it exists in addition to the rgb_pixel is so you can lay
			it down on top of a memory region that organizes its color data in the
			BGR format and still be able to read it)
	!*/

	bgr_pixel(
	) {}

	bgr_pixel(
			unsigned char blue_,
			unsigned char green_,
			unsigned char red_
	) : blue(blue_), green(green_), red(red_) {}

	unsigned char blue;
	unsigned char green;
	unsigned char red;

	bool operator==(const bgr_pixel& that) const {
		return this->blue == that.blue
		       && this->green == that.green
		       && this->red == that.red;
	}

	bool operator!=(const bgr_pixel& that) const {
		return !(*this == that);
	}

};
// ----------------------------------------------------------------------------------------

template<>
struct pixel_traits<bgr_pixel> {
	constexpr static bool rgb = true;
	constexpr static bool rgb_alpha = false;
	constexpr static bool grayscale = false;
	constexpr static bool hsi = false;
	constexpr static bool lab = false;
	constexpr static long num = 3;
	typedef unsigned char basic_pixel_type;

	static basic_pixel_type min() { return 0; }

	static basic_pixel_type max() { return 255; }

	constexpr static bool is_unsigned = true;
	constexpr static bool has_alpha = false;
};

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

/*!A is_unsigned_type

	This is a template where is_unsigned_type<T>::value == true when T is an unsigned
	scalar type and false when T is a signed scalar type.
!*/
template<
		typename T
>
struct is_unsigned_type {
	static const bool value = static_cast<T>((static_cast<T>(0) - static_cast<T>(1))) > 0;
};
template<>
struct is_unsigned_type<long double> {
	static const bool value = false;
};
template<>
struct is_unsigned_type<double> {
	static const bool value = false;
};
template<>
struct is_unsigned_type<float> {
	static const bool value = false;
};

template<typename T>
struct grayscale_pixel_traits {
	constexpr static bool rgb = false;
	constexpr static bool rgb_alpha = false;
	constexpr static bool grayscale = true;
	constexpr static bool hsi = false;
	constexpr static bool lab = false;
	constexpr static long num = 1;
	constexpr static bool has_alpha = false;
	typedef T basic_pixel_type;

	static basic_pixel_type min() { return std::numeric_limits<T>::min(); }

	static basic_pixel_type max() { return std::numeric_limits<T>::max(); }

	constexpr static bool is_unsigned = is_unsigned_type<T>::value;
};

template<>
struct pixel_traits<unsigned char> : public grayscale_pixel_traits<unsigned char> {
};
template<>
struct pixel_traits<unsigned short> : public grayscale_pixel_traits<unsigned short> {
};
template<>
struct pixel_traits<unsigned int> : public grayscale_pixel_traits<unsigned int> {
};
template<>
struct pixel_traits<unsigned long> : public grayscale_pixel_traits<unsigned long> {
};

template<>
struct pixel_traits<char> : public grayscale_pixel_traits<char> {
};
template<>
struct pixel_traits<signed char> : public grayscale_pixel_traits<signed char> {
};
template<>
struct pixel_traits<short> : public grayscale_pixel_traits<short> {
};
template<>
struct pixel_traits<int> : public grayscale_pixel_traits<int> {
};
template<>
struct pixel_traits<long> : public grayscale_pixel_traits<long> {
};

template<>
struct pixel_traits<int64> : public grayscale_pixel_traits<int64> {
};
template<>
struct pixel_traits<uint64> : public grayscale_pixel_traits<uint64> {
};

// ----------------------------------------------------------------------------------------

template<typename T>
struct float_grayscale_pixel_traits {
	constexpr static bool rgb = false;
	constexpr static bool rgb_alpha = false;
	constexpr static bool grayscale = true;
	constexpr static bool hsi = false;
	constexpr static bool lab = false;
	constexpr static long num = 1;
	constexpr static bool has_alpha = false;
	typedef T basic_pixel_type;

	static basic_pixel_type min() { return -std::numeric_limits<T>::max(); }

	static basic_pixel_type max() { return std::numeric_limits<T>::max(); }

	constexpr static bool is_unsigned = false;
};

template<>
struct pixel_traits<float> : public float_grayscale_pixel_traits<float> {
};
template<>
struct pixel_traits<double> : public float_grayscale_pixel_traits<double> {
};
template<>
struct pixel_traits<long double> : public float_grayscale_pixel_traits<long double> {
};

// These are here mainly so you can easily copy images into complex arrays.  This is
// useful when you want to do a FFT on an image or some similar operation.
template<>
struct pixel_traits<std::complex<float> > : public float_grayscale_pixel_traits<float> {
};
template<>
struct pixel_traits<std::complex<double> > : public float_grayscale_pixel_traits<double> {
};
template<>
struct pixel_traits<std::complex<long double> > : public float_grayscale_pixel_traits<long double> {
};
// region ================================ enable if =====================================================
template<bool B, class T = void>
struct enable_if_c {
	typedef T type;
};

template<class T>
struct enable_if_c<false, T> {
};

template<class Cond, class T = void>
struct enable_if : public enable_if_c<Cond::value, T> {
};

// endregion ===========================================================================================
// region ================================ assign pixel ==============================================

namespace assign_pixel_helpers {

// -----------------------------
// all the same kind

template<typename P>
typename enable_if_c<pixel_traits<P>::grayscale>::type
assign(P& dest, const P& src) {
	dest = src;
}

template <typename T>
typename unsigned_type<T>::type make_unsigned (
		const T& val
) { return static_cast<typename unsigned_type<T>::type>(val); }

inline float make_unsigned(const float& val) { return val; }
inline double make_unsigned(const double& val) { return val; }
inline long double make_unsigned(const long double& val) { return val; }

/*!
	ensures
		- returns true if p is <= max value of T
!*/
template<typename T, typename P>
typename enable_if_c<pixel_traits<T>::is_unsigned == pixel_traits<P>::is_unsigned, bool>::type
less_or_equal_to_max(const P& p) {
	return p <= pixel_traits<T>::max();
}

template <typename T, typename P>
typename enable_if_c<pixel_traits<T>::is_unsigned && !pixel_traits<P>::is_unsigned, bool>::type
less_or_equal_to_max (const P& p){
	if (p <= 0 || make_unsigned(p) <= pixel_traits<T>::max())
		return true;
	else
		return false;
}
template <typename T, typename P>
typename enable_if_c<!pixel_traits<T>::is_unsigned && pixel_traits<P>::is_unsigned, bool>::type \
less_or_equal_to_max (const P& p){
	return p <= make_unsigned(pixel_traits<T>::max());
}

/*!
	ensures
		- returns true if p is >= min value of T
!*/
template<typename T, typename P>
typename enable_if_c<pixel_traits<P>::is_unsigned, bool>::type
greater_or_equal_to_min(const P&) {
	return true;
}

template<typename T, typename P>
typename enable_if_c<!pixel_traits<P>::is_unsigned && pixel_traits<T>::is_unsigned, bool>::type
greater_or_equal_to_min(const P& p) {
	return p >= 0;
}

template<typename T, typename P>
typename enable_if_c<!pixel_traits<P>::is_unsigned && !pixel_traits<T>::is_unsigned, bool>::type greater_or_equal_to_min(const P& p) {
	return p >= pixel_traits<T>::min();
}


template<typename P1, typename P2>
typename enable_if_c<pixel_traits<P1>::grayscale && pixel_traits<P2>::grayscale>::type
assign(P1& dest, const P2& src) {
	/*
		The reason for these weird comparison functions is to avoid getting compiler
		warnings about comparing signed types to unsigned and stuff like that.
	*/

	if (less_or_equal_to_max<P1>(src))
		if (greater_or_equal_to_min<P1>(src))
			dest = static_cast<P1>(src);
		else
			dest = pixel_traits<P1>::min();
	else
		dest = pixel_traits<P1>::max();
}

template<typename P1, typename P2>
typename enable_if_c<pixel_traits<P1>::rgb && pixel_traits<P2>::rgb>::type
assign(P1& dest, const P2& src) {
	dest.red = src.red;
	dest.green = src.green;
	dest.blue = src.blue;
}

template < typename P1, typename P2 >
typename enable_if_c<pixel_traits<P1>::rgb_alpha && pixel_traits<P2>::rgb_alpha>::type
assign(P1& dest, const P2& src)
{
	dest.red = src.red;
	dest.green = src.green;
	dest.blue = src.blue;
	dest.alpha = src.alpha;
}

template < typename P1, typename P2 >
typename enable_if_c<pixel_traits<P1>::grayscale && pixel_traits<P2>::rgb>::type
assign(P1& dest, const P2& src)
{
	const unsigned int temp = ((static_cast<unsigned int>(src.red) +
	                            static_cast<unsigned int>(src.green) +
	                            static_cast<unsigned int>(src.blue))/3);
	assign(dest, temp);
}

template < typename P1, typename P2 >
typename enable_if_c<pixel_traits<P1>::grayscale && pixel_traits<P2>::rgb_alpha>::type
assign(P1& dest, const P2& src)
{

	const unsigned char avg = static_cast<unsigned char>((static_cast<unsigned int>(src.red) +
	                                                      static_cast<unsigned int>(src.green) +
	                                                      static_cast<unsigned int>(src.blue))/3);

	if (src.alpha == 255)
	{
		assign(dest, avg);
	}
	else
	{
		// perform this assignment using fixed point arithmetic:
		// dest = src*(alpha/255) + dest*(1 - alpha/255);
		// dest = src*(alpha/255) + dest*1 - dest*(alpha/255);
		// dest = dest*1 + src*(alpha/255) - dest*(alpha/255);
		// dest = dest*1 + (src - dest)*(alpha/255);
		// dest += (src - dest)*(alpha/255);

		int temp = avg;
		// copy dest into dest_copy using assign_pixel to avoid potential
		// warnings about implicit float to int warnings.
		int dest_copy;
		assign(dest_copy, dest);

		temp -= dest_copy;

		temp *= src.alpha;

		temp /= 255;

		assign(dest, temp+dest_copy);
	}
}

template < typename P1 >
typename enable_if_c<pixel_traits<P1>::rgb>::type
assign(P1& dest, const unsigned char& src)
{
	dest.red = src;
	dest.green = src;
	dest.blue = src;
}

template < typename P1, typename P2 >
typename enable_if_c<pixel_traits<P1>::rgb && pixel_traits<P2>::grayscale>::type
assign(P1& dest, const P2& src)
{
	unsigned char p;
	assign_pixel(p, src);
	dest.red = p;
	dest.green = p;
	dest.blue = p;
}

template < typename P1, typename P2 >
typename enable_if_c<pixel_traits<P1>::rgb && pixel_traits<P2>::rgb_alpha>::type
assign(P1& dest, const P2& src)
{
	if (src.alpha == 255)
	{
		dest.red = src.red;
		dest.green = src.green;
		dest.blue = src.blue;
	}
	else
	{
		// perform this assignment using fixed point arithmetic:
		// dest = src*(alpha/255) + dest*(1 - alpha/255);
		// dest = src*(alpha/255) + dest*1 - dest*(alpha/255);
		// dest = dest*1 + src*(alpha/255) - dest*(alpha/255);
		// dest = dest*1 + (src - dest)*(alpha/255);
		// dest += (src - dest)*(alpha/255);

		unsigned int temp_r = src.red;
		unsigned int temp_g = src.green;
		unsigned int temp_b = src.blue;

		temp_r -= dest.red;
		temp_g -= dest.green;
		temp_b -= dest.blue;

		temp_r *= src.alpha;
		temp_g *= src.alpha;
		temp_b *= src.alpha;

		temp_r >>= 8;
		temp_g >>= 8;
		temp_b >>= 8;

		dest.red += static_cast<unsigned char>(temp_r&0xFF);
		dest.green += static_cast<unsigned char>(temp_g&0xFF);
		dest.blue += static_cast<unsigned char>(temp_b&0xFF);
	}
}

template < typename P1 >
typename enable_if_c<pixel_traits<P1>::rgb_alpha>::type
assign(P1& dest, const unsigned char& src)
{
	dest.red = src;
	dest.green = src;
	dest.blue = src;
	dest.alpha = 255;
}


template < typename P1, typename P2 >
typename enable_if_c<pixel_traits<P1>::rgb_alpha && pixel_traits<P2>::grayscale>::type
assign(P1& dest, const P2& src)
{
	unsigned char p;
	assign(p, src);

	dest.red = p;
	dest.green = p;
	dest.blue = p;
	dest.alpha = 255;
}

template < typename P1, typename P2 >
typename enable_if_c<pixel_traits<P1>::rgb_alpha && pixel_traits<P2>::rgb>::type
assign(P1& dest, const P2& src)
{
	dest.red = src.red;
	dest.green = src.green;
	dest.blue = src.blue;
	dest.alpha = 255;
}

} // namespace assign_pixel_helpers

template<typename P1, typename P2>
inline void assign_pixel(P1& dest, const P2& src) {
	assign_pixel_helpers::assign(dest, src);
}


// endregion


} // namespace test