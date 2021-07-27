// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See https://github.com/davisking/dlib/blob/master/LICENSE.txt for the full license.
#pragma once

#include <iostream>
#include <fstream>
#include <vector>

#include "dng_shared.h"

#include "../algs.h"
#include "../image_view.h"
#include "../vectorstream.h"
#include "../float_details.h"
#include "../entropy_range_encoder.h"
#include "../entropy_encoder_model.h"

namespace test {

// region ======================================= serialize ===============================================================================

/*!
	requires
		- T is a signed integral type
	ensures
		- if (no problems occur serializing item) then
			- writes item to out
			- returns false
		- else
			- returns true
!*/
template<typename T>
typename enable_if_c<std::numeric_limits<T>::is_signed, bool>::type pack_int(T item, std::ostream& out) {
	static_assert(sizeof(T) <= 8, "Failed assertion.");
	unsigned char buf[9];
	unsigned char size = sizeof(T);
	unsigned char neg;
	if (item < 0) {
		neg = 0x80;
		item *= -1;
	} else {
		neg = 0;
	}

	for (unsigned char i = 1; i <= sizeof(T); ++i) {
		buf[i] = static_cast<unsigned char>(item & 0xFF);
		item >>= 8;
		if (item == 0) {
			size = i;
			break;
		}
	}

	std::streambuf* sbuf = out.rdbuf();
	buf[0] = size | neg;
	if (sbuf->sputn(reinterpret_cast<char*>(buf), size + 1) != size + 1) {
		out.setstate(std::ios::eofbit | std::ios::badbit);
		return true;
	}

	return false;
}
// ------------------------------------------------------------------------------------

/*!
	requires
		- T is an unsigned integral type
	ensures
		- if (no problems occur serializing item) then
			- writes item to out
			- returns false
		- else
			- returns true
!*/
template<typename T>
typename disable_if_c<std::numeric_limits<T>::is_signed, bool>::type pack_int(T item, std::ostream& out) {
	static_assert(sizeof(T) <= 8, "Failed assertion.");
	unsigned char buf[9];
	unsigned char size = sizeof(T);

	for (unsigned char i = 1; i <= sizeof(T); ++i) {
		buf[i] = static_cast<unsigned char>(item & 0xFF);
		item >>= 8;
		if (item == 0) {
			size = i;
			break;
		}
	}

	std::streambuf* sbuf = out.rdbuf();
	buf[0] = size;
	if (sbuf->sputn(reinterpret_cast<char*>(buf), size + 1) != size + 1) {
		out.setstate(std::ios::eofbit | std::ios::badbit);
		return true;
	}

	return false;
}

template<typename alloc>
void serialize(const std::vector<char, alloc>& item, std::ostream& out) {
	try {
		const auto size = static_cast<const unsigned long>(item.size());
		pack_int(size, out);
		if (!item.empty()) {
			out.write(&item[0], item.size());
		}
	} catch (std::runtime_error& e) { throw std::runtime_error(std::string(e.what()) + "\n   while serializing object of type std::vector"); }
}
// ------------------------------------------------------------------------------------


// endregion
namespace dng_helpers_namespace {


template<typename image_type, typename enabled = void>
struct save_dng_helper;

typedef entropy_range_encoder encoder_type;
typedef entropy_encoder_model_mantissa<256, encoder_type, 200000, 4> eem_type;

typedef entropy_encoder_model_exponent<256, encoder_type, 200000, 4> eem_exp_type;

template<typename image_type>
struct save_dng_helper<image_type, typename enable_if<is_float_type<typename image_traits<image_type>::pixel_type> >::type> {
	static void save_dng(const image_type& image_, std::ostream& out) {
		const_image_view<image_type> image(image_);
		out.write("DNG", 3);
		unsigned long version = 1;
		pack_int(version, out);
		unsigned long type = grayscale_float;
		pack_int(type, out);
		pack_int(image.nc(), out);
		pack_int(image.nr(), out);


		// Write the compressed exponent data into expbuf.  We will append it
		// to the stream at the end of the loops.
		std::vector<char> expbuf;
		expbuf.reserve(image.size() * 2);
		vectorstream outexp(expbuf);
		encoder_type encoder;
		encoder.set_stream(outexp);

		eem_exp_type eem_exp(encoder);
		float_details prev;
		for (long r = 0; r < image.nr(); ++r) {
			for (long c = 0; c < image.nc(); ++c) {
				float_details cur = image[r][c];
				int16 exp = cur.exponent - prev.exponent;
				int64 man = cur.mantissa - prev.mantissa;
				prev = cur;

				unsigned char ebyte1 = exp & 0xFF;
				unsigned char ebyte2 = exp >> 8;
				eem_exp.encode(ebyte1);
				eem_exp.encode(ebyte2);

				pack_int(man, out);
			}
		}
		// write out the magic byte to mark the end of the compressed data.
		eem_exp.encode(dng_magic_byte);
		eem_exp.encode(dng_magic_byte);
		eem_exp.encode(dng_magic_byte);
		eem_exp.encode(dng_magic_byte);

		encoder.clear();
		serialize(expbuf, out);
	}
};

template<typename image_type>
struct is_non_float_non8bit_grayscale {
	typedef typename image_traits<image_type>::pixel_type pixel_type;
	const static bool value = pixel_traits<pixel_type>::grayscale &&
	                          sizeof(pixel_type) != 1 &&
	                          !is_float_type<pixel_type>::value;
};

template<typename image_type>
struct save_dng_helper<image_type, typename enable_if<is_non_float_non8bit_grayscale<image_type> >::type> {
	static void save_dng(const image_type& image_, std::ostream& out) {
		const_image_view<image_type> image(image_);
		out.write("DNG", 3);
		unsigned long version = 1;
		pack_int(version, out);
		unsigned long type = grayscale_16bit;
		pack_int(type, out);
		pack_int(image.nc(), out);
		pack_int(image.nr(), out);

		encoder_type encoder;
		encoder.set_stream(out);

		eem_type eem(encoder);
		for (long r = 0; r < image.nr(); ++r) {
			for (long c = 0; c < image.nc(); ++c) {
				uint16 cur;
				assign_pixel(cur, image[r][c]);
				cur -= predictor_grayscale_16(image, r, c);
				unsigned char byte1 = cur & 0xFF;
				unsigned char byte2 = cur >> 8;
				eem.encode(byte2);
				eem.encode(byte1);
			}
		}
		// write out the magic byte to mark the end of the data
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
	}
};

template<typename image_type>
struct is_8bit_grayscale {
	typedef typename image_traits<image_type>::pixel_type pixel_type;
	const static bool value = pixel_traits<pixel_type>::grayscale && sizeof(pixel_type) == 1;
};

template<typename image_type>
struct save_dng_helper<image_type, typename enable_if<is_8bit_grayscale<image_type> >::type> {
	static void save_dng(
			const image_type& image_,
			std::ostream& out
	) {
		const_image_view<image_type> image(image_);
		out.write("DNG", 3);
		unsigned long version = 1;
		pack_int(version, out);
		unsigned long type = grayscale;
		pack_int(type, out);
		pack_int(image.nc(), out);
		pack_int(image.nr(), out);

		encoder_type encoder;
		encoder.set_stream(out);

		eem_type eem(encoder);
		for (long r = 0; r < image.nr(); ++r) {
			for (long c = 0; c < image.nc(); ++c) {
				unsigned char cur;
				assign_pixel(cur, image[r][c]);
				cur -= predictor_grayscale(image, r, c);
				eem.encode(cur);
			}
		}
		// write out the magic byte to mark the end of the data
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
	}
};

template <typename image_type>
struct is_rgb_image { const static bool value = pixel_traits<typename image_traits<image_type>::pixel_type>::rgb; };

template<typename image_type>
struct save_dng_helper<image_type, typename enable_if<is_rgb_image < image_type> >::type> {
static void save_dng(
		const image_type& image_,
		std::ostream& out
) {
	const_image_view<image_type> image(image_);
	out.write("DNG", 3);
	unsigned long version = 1;
	pack_int(version, out);

	unsigned long type = rgb;
	// if this is a small image then we will use a different predictor
	if (image.size() < 4000)
		type = rgb_paeth;

	pack_int(type, out);
	pack_int(image.nc(), out);
	pack_int(image.nr(), out);

	encoder_type encoder;
	encoder.set_stream(out);

	rgb_pixel pre, cur;
	eem_type eem(encoder);

	if (type == rgb) {
		for (long r = 0; r < image.nr(); ++r) {
			for (long c = 0; c < image.nc(); ++c) {
				pre = predictor_rgb(image, r, c);
				assign_pixel(cur, image[r][c]);

				eem.encode((unsigned char) (cur.red - pre.red));
				eem.encode((unsigned char) (cur.green - pre.green));
				eem.encode((unsigned char) (cur.blue - pre.blue));
			}
		}
	} else {
		for (long r = 0; r < image.nr(); ++r) {
			for (long c = 0; c < image.nc(); ++c) {
				pre = predictor_rgb_paeth(image, r, c);
				assign_pixel(cur, image[r][c]);

				eem.encode((unsigned char) (cur.red - pre.red));
				eem.encode((unsigned char) (cur.green - pre.green));
				eem.encode((unsigned char) (cur.blue - pre.blue));
			}
		}
	}
	// write out the magic byte to mark the end of the data
	eem.encode(dng_magic_byte);
	eem.encode(dng_magic_byte);
	eem.encode(dng_magic_byte);
	eem.encode(dng_magic_byte);
}
};

template<typename image_type>
struct is_rgb_alpha_image {
	typedef typename image_traits<image_type>::pixel_type pixel_type;
	const static bool value = pixel_traits<pixel_type>::rgb_alpha;
};

template<typename image_type>
struct save_dng_helper<image_type, typename enable_if<is_rgb_alpha_image<image_type> >::type> {
	static void save_dng(
			const image_type& image_,
			std::ostream& out
	) {
		const_image_view<image_type> image(image_);
		out.write("DNG", 3);
		unsigned long version = 1;
		pack_int(version, out);

		unsigned long type = rgb_alpha;
		// if this is a small image then we will use a different predictor
		if (image.size() < 4000)
			type = rgb_alpha_paeth;

		pack_int(type, out);
		pack_int(image.nc(), out);
		pack_int(image.nr(), out);

		encoder_type encoder;
		encoder.set_stream(out);

		rgb_alpha_pixel pre, cur;
		eem_type eem(encoder);

		if (type == rgb_alpha) {
			for (long r = 0; r < image.nr(); ++r) {
				for (long c = 0; c < image.nc(); ++c) {
					pre = predictor_rgb_alpha(image, r, c);
					assign_pixel(cur, image[r][c]);

					eem.encode((unsigned char) (cur.red - pre.red));
					eem.encode((unsigned char) (cur.green - pre.green));
					eem.encode((unsigned char) (cur.blue - pre.blue));
					eem.encode((unsigned char) (cur.alpha - pre.alpha));
				}
			}
		} else {
			for (long r = 0; r < image.nr(); ++r) {
				for (long c = 0; c < image.nc(); ++c) {
					pre = predictor_rgb_alpha_paeth(image, r, c);
					assign_pixel(cur, image[r][c]);

					eem.encode((unsigned char) (cur.red - pre.red));
					eem.encode((unsigned char) (cur.green - pre.green));
					eem.encode((unsigned char) (cur.blue - pre.blue));
					eem.encode((unsigned char) (cur.alpha - pre.alpha));
				}
			}
		}
		// write out the magic byte to mark the end of the data
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
	}
};

template<typename image_type>
struct is_hsi_image {
	typedef typename image_traits<image_type>::pixel_type pixel_type;
	const static bool value = pixel_traits<pixel_type>::hsi;
};

template<typename image_type>
struct save_dng_helper<image_type, typename enable_if<is_hsi_image<image_type> >::type> {
	static void save_dng(
			const image_type& image_,
			std::ostream& out
	) {
		const_image_view<image_type> image(image_);
		out.write("DNG", 3);
		unsigned long version = 1;
		pack_int(version, out);
		unsigned long type = hsi;
		pack_int(type, out);
		serialize(image.nc(), out);
		serialize(image.nr(), out);

		encoder_type encoder;
		encoder.set_stream(out);

		hsi_pixel pre, cur;
		eem_type eem(encoder);
		for (long r = 0; r < image.nr(); ++r) {
			for (long c = 0; c < image.nc(); ++c) {
				pre = predictor_hsi(image, r, c);
				assign_pixel(cur, image[r][c]);

				eem.encode((unsigned char) (cur.h - pre.h));
				eem.encode((unsigned char) (cur.s - pre.s));
				eem.encode((unsigned char) (cur.i - pre.i));
			}
		}
		// write out the magic byte to mark the end of the data
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
		eem.encode(dng_magic_byte);
	}
};


} // namespace dng_helpers_namespace

// comment is legacy dlib code that stood in place of the following 'void',
// which had matrix-type images that had to be saved via a completely different implementation of the 'save_png' function.
template<typename image_type>
inline /*typename disable_if<is_matrix<image_type> >::type*/ void save_dng(const image_type& image, std::ostream& out) {
	using namespace dng_helpers_namespace;
	save_dng_helper<image_type>::save_dng(image, out);
}

// ----------------------------------------------------------------------------------------
template<typename image_type>
void save_dng(const image_type& image, const std::string& file_name) {
	std::ofstream fout(file_name.c_str(), std::ios::binary);
	if (!fout) {
		throw std::runtime_error("Unable to open " + file_name + " for writing.");
	}
	save_dng(image, fout);
}

// ----------------------------------------------------------------------------------------
} // namespace test