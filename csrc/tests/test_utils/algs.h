//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/26/21.
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

namespace test {
// region ============================================== tmax / tmin =================================================================================
/*!A tmax

	This is a template to compute the max of two values at compile time

	For example,
		abs<4,7>::value == 7
!*/

template<long x, long y, typename enabled=void>
struct tmax {
	const static long value = x;
};
template<long x, long y>
struct tmax<x, y, typename enable_if_c<(y > x)>::type> {
	const static long value = y;
};
/*!A tmin

	This is a template to compute the min of two values at compile time

	For example,
		abs<4,7>::value == 4
!*/
template<long x, long y, typename enabled=void>
struct tmin {
	const static long value = x;
};
template<long x, long y>
struct tmin<x, y, typename enable_if_c<(y < x)>::type> {
	const static long value = y;
};

// endregion

// ----------------------------------------------------------------------------------------

/*!A is_same_type

	This is a template where is_same_type<T,U>::value == true when T and U are the
	same type and false otherwise.
!*/

template<typename T, typename U>
class is_same_type {
public:
	enum {
		value = false
	};
private:
	is_same_type();
};

template<typename T>
class is_same_type<T, T> {
public:
	enum {
		value = true
	};
private:
	is_same_type();
};
// ----------------------------------------------------------------------------------------

/*!A is_convertible

	This is a template that can be used to determine if one type is convertible
	into another type.

	For example:
		is_convertible<int,float>::value == true    // because ints are convertible to floats
		is_convertible<int*,float>::value == false  // because int pointers are NOT convertible to floats
!*/

template<typename from, typename to>
struct is_convertible {
	struct yes_type {
		char a;
	};
	struct no_type {
		yes_type a[2];
	};
	static const from& from_helper();
	static yes_type test(to);
	static no_type test(...);
	const static bool value = sizeof(test(from_helper())) == sizeof(yes_type);
};

// ----------------------------------------------------------------------------------------


/*!A is_same_object

	This is a templated function which checks if both of its arguments are actually
	references to the same object.  It returns true if they are and false otherwise.

!*/

// handle the case where T and U are unrelated types.
template<typename T, typename U>
typename disable_if_c<is_convertible<T*, U*>::value || is_convertible<U*, T*>::value, bool>::type
is_same_object(const T& a, const U& b) {
	return ((void*) &a == (void*) &b);
}

// handle the case where T and U are related types because their pointers can be
// implicitly converted into one or the other.  E.g. a derived class and its base class.
// Or where both T and U are just the same type.  This way we make sure that if there is a
// valid way to convert between these two pointer types then we will take that route rather
// than the void* approach used otherwise.
template<typename T, typename U>
typename enable_if_c<is_convertible<T*, U*>::value || is_convertible<U*, T*>::value, bool>::type
is_same_object(const T& a, const U& b) {
	return (&a == &b);
}

// ----------------------------------------------------------------------------------------
/*!A is_float_type

	This is a template that can be used to determine if a type is one of the built
	int floating point types (i.e. float, double, or long double).
!*/
template < typename T > struct is_float_type  { const static bool value = false; };
template <> struct is_float_type<float>       { const static bool value = true; };
template <> struct is_float_type<double>      { const static bool value = true; };
template <> struct is_float_type<long double> { const static bool value = true; };

// ----------------------------------------------------------------------------------------
} // namespace test