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

#include "algs.h"

namespace test {
struct float_details {
	/*!
		WHAT THIS OBJECT REPRESENTS
			This object is a tool for converting floating point numbers into an
			explicit integer representation and then also converting back.  In
			particular, a float_details object represents a floating point number with
			a 64 bit mantissa and 16 bit exponent.  These are stored in the public
			fields of the same names.

			The main use of this object is to convert floating point residuals into a
			known uniform representation so they can be serialized to an output stream.
			This allows dlib serialization code to work on any system, regardless of
			the floating point representation used by the hardware.  It also means
			that, for example, a double can be serialized and then deserialized into a
			float and it will perform the appropriate conversion.


			In more detail, this object represents a floating point value equal to
			mantissa*pow(2,exponent), except when exponent takes on any of the
			following special residuals:
				- is_inf
				- is_ninf
				- is_nan
			These residuals are used to indicate that the floating point value should be
			either infinity, negative infinity, or not-a-number respectively.
	!*/

	float_details(
			int64 man,
			int16 exp
	) : mantissa(man), exponent(exp) {}
	/*!
		ensures
			- #mantissa == man
			- #exponent == exp
	!*/

	float_details() :
			mantissa(0), exponent(0)
	{}
	/*!
		ensures
			- this object represents a floating point value of 0
	!*/

	float_details ( const double&      val) { *this = val; }
	float_details ( const float&       val) { *this = val; }
	float_details ( const long double& val) { *this = val; }
	/*!
		ensures
			- converts the given value into a float_details representation.  This
			  means that converting #*this back into a floating point number should
			  recover the input val.
	!*/

	float_details& operator= ( const double&      val) { convert_from_T(val); return *this; }
	float_details& operator= ( const float&       val) { convert_from_T(val); return *this; }
	float_details& operator= ( const long double& val) { convert_from_T(val); return *this; }
	/*!
		ensures
			- converts the given value into a float_details representation.  This
			  means that converting #*this back into a floating point number should
			  recover the input val.
	!*/

	operator double      () const { return convert_to_T<double>(); }
	operator float       () const { return convert_to_T<float>(); }
	operator long double () const { return convert_to_T<long double>(); }
	/*!
		ensures
			- converts the contents of this float_details object into a floating point number.
	!*/

	const static int16 is_inf  = 32000;
	const static int16 is_ninf = 32001;
	const static int16 is_nan  = 32002;

	int64 mantissa;
	int16 exponent;


private:


// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                                  IMPLEMENTATION DETAILS
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

	template <typename T>
	void convert_from_T (
			const T& val
	)
	{
		mantissa = 0;

		const int digits = tmin<std::numeric_limits<T>::digits, 63>::value;

		if (val == std::numeric_limits<T>::infinity())
		{
			exponent = is_inf;
		}
		else if (val == -std::numeric_limits<T>::infinity())
		{
			exponent = is_ninf;
		}
		else if (val < std::numeric_limits<T>::infinity())
		{
			int exp;
			mantissa = static_cast<int64>(std::frexp(val, &exp)*(((uint64)1)<<digits));
			exponent = exp - digits;

			// Compact the representation a bit by shifting off any low order bytes
			// which are zero in the mantissa.  This makes the numbers in mantissa and
			// exponent generally smaller which can make serialization and other things
			// more efficient in some cases.
			for (int i = 0; i < 8 && ((mantissa&0xFF)==0); ++i)
			{
				mantissa >>= 8;
				exponent += 8;
			}
		}
		else
		{
			exponent = is_nan;
		}
	}

	template <typename T>
	T convert_to_T (
	) const
	{
		if (exponent < is_inf)
			return std::ldexp((T)mantissa, exponent);
		else if (exponent == is_inf)
			return std::numeric_limits<T>::infinity();
		else if (exponent == is_ninf)
			return -std::numeric_limits<T>::infinity();
		else
			return std::numeric_limits<T>::quiet_NaN();
	}

};
} // namespace test