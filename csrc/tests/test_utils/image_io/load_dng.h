// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See https://github.com/davisking/dlib/blob/master/LICENSE.txt for the full license.
#pragma once

#include <iostream>
#include <exception>
#include <vector>
#include <fstream>

#include "tests/test_utils/image_view.h"
#include "dng_shared.h"
#include "tests/test_utils/enable_if.h"
#include "tests/test_utils/entropy_range_decoder.h"
#include "tests/test_utils/entropy_decoder_model.h"
#include "tests/test_utils/float_details.h"
#include "tests/test_utils/vectorstream.h"

namespace test {

// region ============================================== deserialize =================================================================================
/*!
	requires
		- T is a signed integral type
	ensures
		- if (there are no problems deserializing item) then
			- returns false
			- #item == the value stored in in
		- else
			- returns true

!*/
template<typename T>
typename enable_if_c<std::numeric_limits<T>::is_signed, bool>::type unpack_int(T& item, std::istream& in) {
	static_assert(sizeof(T) <= 8, "Failed assertion");

	unsigned char buf[8];
	unsigned char size;
	bool is_negative;

	std::streambuf* sbuf = in.rdbuf();

	item = 0;
	int ch = sbuf->sbumpc();
	if (ch != EOF) {
		size = static_cast<unsigned char>(ch);
	} else {
		in.setstate(std::ios::badbit);
		return true;
	}

	if (size & 0x80) {
		is_negative = true;
	} else {
		is_negative = false;
	}
	size &= 0x0F;

	// check if the serialized object is too big
	if (size > (unsigned long) tmin<sizeof(T), 8>::value || size == 0) {
		return true;
	}

	if (sbuf->sgetn(reinterpret_cast<char*>(&buf), size) != size) {
		in.setstate(std::ios::badbit);
		return true;
	}


	for (unsigned char i = size - 1; true; --i) {
		item <<= 8;
		item |= buf[i];
		if (i == 0)
			break;
	}

	if (is_negative)
		item *= -1;


	return false;
}

/*!
	requires
		- T is an unsigned integral type
	ensures
		- if (there are no problems deserializing item) then
			- returns false
			- #item == the value stored in in
		- else
			- returns true

!*/
template<typename T>
typename disable_if_c<std::numeric_limits<T>::is_signed, bool>::type unpack_int(T& item, std::istream& in) {
	static_assert(sizeof(T) <= 8, "Failed assertion.");

	unsigned char buf[8];
	unsigned char size;

	item = 0;

	std::streambuf* sbuf = in.rdbuf();
	int ch = sbuf->sbumpc();
	if (ch != EOF) {
		size = static_cast<unsigned char>(ch);
	} else {
		in.setstate(std::ios::badbit);
		return true;
	}

	// mask out the 3 reserved bits
	size &= 0x8F;

	// check if an error occurred
	if (size > (unsigned long) tmin<sizeof(T), 8>::value || size == 0) {
		return true;
	}

	if (sbuf->sgetn(reinterpret_cast<char*>(&buf), size) != size) {
		in.setstate(std::ios::badbit);
		return true;
	}

	for (unsigned char i = size - 1; true; --i) {
		item <<= 8;
		item |= buf[i];
		if (i == 0)
			break;
	}

	return false;
}

template <typename alloc>
void deserialize (std::vector<char, alloc>& item, std::istream& in){
	try {
		unsigned long size;
		unpack_int(size,in);
		item.resize(size);
		if (item.size() != 0){
			in.read(&item[0], item.size());
		}
	} catch (std::runtime_error& e) {
		throw std::runtime_error(std::string(e.what()) + "\n   while deserializing object of type std::vector");
	}
}

// endregion
template<typename image_type>
void load_dng(image_type& image_, std::istream& in) {
	image_view<image_type> image(image_);
	using namespace dng_helpers_namespace;
	try {
		if (in.get() != 'D' || in.get() != 'N' || in.get() != 'G') {
			throw std::runtime_error("the stream does not contain a dng image file");
		}

		unsigned long version;
		unpack_int(version, in);
		if (version != 1) {
			throw std::runtime_error("You need the new version of the dlib library to read this dng file");
		}

		unsigned long type;
		unpack_int(type, in);

		long width, height;
		unpack_int(width, in);
		unpack_int(height, in);

		if (width > 0 && height > 0) {
			image.set_size(height, width);
		} else {
			image.clear();
		}

		if (type != grayscale_float) {
			typedef entropy_range_decoder decoder_type;
			decoder_type decoder;
			decoder.set_stream(in);

			entropy_decoder_model_mantissa<256, decoder_type, 200000, 4> edm(decoder);
			unsigned long symbol;
			rgb_pixel p_rgb;
			rgb_alpha_pixel p_rgba;
			hsi_pixel p_hsi;
			switch (type) {
				case rgb_alpha_paeth:

					for (long r = 0; r < image.nr(); ++r) {
						for (long c = 0; c < image.nc(); ++c) {
							p_rgba = predictor_rgb_alpha_paeth(image, r, c);
							edm.decode(symbol);
							p_rgba.red += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgba.green += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgba.blue += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgba.alpha += static_cast<unsigned char>(symbol);

							assign_pixel(image[r][c], p_rgba);
						}
					}
					break;

				case rgb_alpha:

					for (long r = 0; r < image.nr(); ++r) {
						for (long c = 0; c < image.nc(); ++c) {
							p_rgba = predictor_rgb_alpha(image, r, c);
							edm.decode(symbol);
							p_rgba.red += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgba.green += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgba.blue += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgba.alpha += static_cast<unsigned char>(symbol);

							assign_pixel(image[r][c], p_rgba);
						}
					}
					break;

				case rgb_paeth:

					for (long r = 0; r < image.nr(); ++r) {
						for (long c = 0; c < image.nc(); ++c) {
							p_rgb = predictor_rgb_paeth(image, r, c);
							edm.decode(symbol);
							p_rgb.red += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgb.green += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgb.blue += static_cast<unsigned char>(symbol);

							assign_pixel(image[r][c], p_rgb);
						}
					}
					break;

				case rgb:

					for (long r = 0; r < image.nr(); ++r) {
						for (long c = 0; c < image.nc(); ++c) {
							p_rgb = predictor_rgb(image, r, c);
							edm.decode(symbol);
							p_rgb.red += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgb.green += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_rgb.blue += static_cast<unsigned char>(symbol);

							assign_pixel(image[r][c], p_rgb);
						}
					}
					break;

				case hsi:

					for (long r = 0; r < image.nr(); ++r) {
						for (long c = 0; c < image.nc(); ++c) {
							p_hsi = predictor_hsi(image, r, c);
							edm.decode(symbol);
							p_hsi.h += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_hsi.s += static_cast<unsigned char>(symbol);

							edm.decode(symbol);
							p_hsi.i += static_cast<unsigned char>(symbol);

							assign_pixel(image[r][c], p_hsi);
						}
					}
					break;

				case grayscale: {
					unsigned char p;
					for (long r = 0; r < image.nr(); ++r) {
						for (long c = 0; c < image.nc(); ++c) {
							edm.decode(symbol);
							p = static_cast<unsigned char>(symbol);
							p += predictor_grayscale(image, r, c);
							assign_pixel(image[r][c], p);
						}
					}
				}
					break;

				case grayscale_16bit: {
					uint16 p;
					for (long r = 0; r < image.nr(); ++r) {
						for (long c = 0; c < image.nc(); ++c) {
							edm.decode(symbol);
							p = static_cast<uint16>(symbol);
							p <<= 8;
							edm.decode(symbol);
							p |= static_cast<uint16>(symbol);

							p += predictor_grayscale_16(image, r, c);
							assign_pixel(image[r][c], p);
						}
					}
				}
					break;

				default:
					throw std::runtime_error("corruption detected in the dng file");
			} // switch (type)

			edm.decode(symbol);
			if (symbol != dng_magic_byte)
				throw std::runtime_error("corruption detected in the dng file");
			edm.decode(symbol);
			if (symbol != dng_magic_byte)
				throw std::runtime_error("corruption detected in the dng file");
			edm.decode(symbol);
			if (symbol != dng_magic_byte)
				throw std::runtime_error("corruption detected in the dng file");
			edm.decode(symbol);
			if (symbol != dng_magic_byte)
				throw std::runtime_error("corruption detected in the dng file");
		} else { // if this is a grayscale_float type image
			std::vector<int64> mantissa(image.size());
			std::vector<char> expbuf;
			// get the mantissa data
			for (long long &j : mantissa){
				unpack_int(j, in);
			}
			// get the compressed exponent data
			deserialize(expbuf, in);
			typedef entropy_range_decoder decoder_type;
			typedef entropy_decoder_model_exponent<256, decoder_type, 200000, 4> edm_exp_type;
			vectorstream inexp(expbuf);
			decoder_type decoder;
			decoder.set_stream(inexp);

			edm_exp_type edm_exp(decoder);
			float_details prev;
			unsigned long i = 0;
			// fill out the image
			for (long r = 0; r < image.nr(); ++r) {
				for (long c = 0; c < image.nc(); ++c) {
					unsigned long exp1, exp2;
					edm_exp.decode(exp1);
					edm_exp.decode(exp2);

					float_details cur(mantissa[i++], (exp2 << 8) | exp1);
					cur.exponent += prev.exponent;
					cur.mantissa += prev.mantissa;
					prev = cur;

					// Only use long double precision if the target image contains long
					// doubles because it's slower to use those.
					if (!is_same_type<typename image_traits<image_type>::pixel_type, long double>::value) {
						double temp = cur;
						assign_pixel(image[r][c], temp);
					} else {
						long double temp = cur;
						assign_pixel(image[r][c], temp);
					}
				}
			}
			unsigned long symbol;
			edm_exp.decode(symbol);
			if (symbol != dng_magic_byte){
				throw std::runtime_error("corruption detected in the dng file");
			}
			edm_exp.decode(symbol);
			if (symbol != dng_magic_byte)
				throw std::runtime_error("corruption detected in the dng file");
			edm_exp.decode(symbol);
			if (symbol != dng_magic_byte)
				throw std::runtime_error("corruption detected in the dng file");
			edm_exp.decode(symbol);
			if (symbol != dng_magic_byte)
				throw std::runtime_error("corruption detected in the dng file");
		}
	}
	catch (...) {
		image.clear();
		throw;
	}

}

template <typename image_type>
void load_dng ( image_type& image, const std::string& file_name) {
	std::ifstream fin(file_name.c_str(), std::ios::binary);
	if (!fin){
		throw std::runtime_error("Unable to open " + file_name + " for reading.");
	}
	load_dng(image, fin);
}

} // namespace test