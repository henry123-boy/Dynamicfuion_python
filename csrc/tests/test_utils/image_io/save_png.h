// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See https://github.com/davisking/dlib/blob/master/LICENSE.txt for the full license.
#pragma once

#include <string>
#include <vector>
#include "../image_view.h"
#include "../algs.h"
#include "../array2d.h"

namespace test {
// region ====================================== assign_image ========================================================================================

template<typename dest_image_type, typename src_image_type>
void impl_assign_image(image_view<dest_image_type>& dest, const const_image_view<src_image_type>& src) {
	dest.set_size(src.nr(), src.nc());
	for (long r = 0; r < src.nr(); ++r) {
		for (long c = 0; c < src.nc(); ++c) {
			assign_pixel(dest[r][c], src[r][c]);
		}
	}
}

template<typename TDestImage, typename TSourceImage>
void impl_assign_image(TDestImage& dest_, const TSourceImage& src_) {
	image_view<TDestImage> dest(dest_);
	const_image_view<TSourceImage> src(src_);
	impl_assign_image(dest, src);
}

template<typename dest_image_type, typename src_image_type>
void assign_image(dest_image_type& dest, const src_image_type& src) {
	// check for the case where dest is the same object as src
	if (is_same_object(dest, src))
		return;
	impl_assign_image(dest, src);
}
// endregion =========================================================================================================================================
// region ====================================== save_png & impl namespace ===========================================================================

// ----------------------------------------------------------------------------------------

namespace impl {

enum png_type {
	png_type_rgb,
	png_type_rgb_alpha,
	png_type_gray,
};

void impl_save_png(const std::string& file_name, std::vector<unsigned char*>& row_pointers,
				   long width, png_type type,int bit_depth);

}

// ----------------------------------------------------------------------------------------


// comment is legacy dlib code that stood in place of the following 'void',
// which had matrix-type images that had to be saved via a completely different implementation of the 'save_png' function.
template<typename image_type>
/* typename disable_if<is_matrix < image_type> >::type*/ void save_png(
		const image_type& img_,
		const std::string& file_name) {
	const_image_view<image_type> img(img_);

	// make sure requires clause is not broken
	if (img.size() != 0) {
		std::stringstream ss;
		ss << "\t save_png(): " << "\n\t You can't save an empty image as a PNG" << std::endl;
		throw std::runtime_error(ss.str());
	}

	std::vector<unsigned char*> row_pointers(img.nr());
	typedef typename image_traits<image_type>::pixel_type pixel_type;

	if (is_same_type<rgb_pixel, pixel_type>::value) {
		for (unsigned long i = 0; i < row_pointers.size(); ++i)
			row_pointers[i] = (unsigned char*) (&img[i][0]);

		impl::impl_save_png(file_name, row_pointers, img.nc(), impl::png_type_rgb, 8);
	} else if (is_same_type<rgb_alpha_pixel, pixel_type>::value) {
		for (unsigned long i = 0; i < row_pointers.size(); ++i)
			row_pointers[i] = (unsigned char*) (&img[i][0]);

		impl::impl_save_png(file_name, row_pointers, img.nc(), impl::png_type_rgb_alpha, 8);
	} else if (pixel_traits<pixel_type>::lab || pixel_traits<pixel_type>::hsi || pixel_traits<pixel_type>::rgb) {
		// convert from Lab or HSI to RGB (Or potentially RGB pixels that aren't laid out as R G B)
		array2d<rgb_pixel> temp_img;
		assign_image(temp_img, img_);
		for (unsigned long i = 0; i < row_pointers.size(); ++i)
			row_pointers[i] = (unsigned char*) (&temp_img[i][0]);

		impl::impl_save_png(file_name, row_pointers, img.nc(), impl::png_type_rgb, 8);
	} else if (pixel_traits<pixel_type>::rgb_alpha) {
		// convert from RGBA pixels that aren't laid out as R G B A
		array2d<rgb_alpha_pixel> temp_img;
		assign_image(temp_img, img_);
		for (unsigned long i = 0; i < row_pointers.size(); ++i)
			row_pointers[i] = (unsigned char*) (&temp_img[i][0]);

		impl::impl_save_png(file_name, row_pointers, img.nc(), impl::png_type_rgb_alpha, 8);
	} else // this is supposed to be grayscale
	{
		static_assert(pixel_traits<pixel_type>::grayscale, "Failed assertion.");

		if (pixel_traits<pixel_type>::is_unsigned && sizeof(pixel_type) == 1) {
			for (unsigned long i = 0; i < row_pointers.size(); ++i)
				row_pointers[i] = (unsigned char*) (&img[i][0]);

			impl::impl_save_png(file_name, row_pointers, img.nc(), impl::png_type_gray, 8);
		} else if (pixel_traits<pixel_type>::is_unsigned && sizeof(pixel_type) == 2) {
			for (unsigned long i = 0; i < row_pointers.size(); ++i)
				row_pointers[i] = (unsigned char*) (&img[i][0]);

			impl::impl_save_png(file_name, row_pointers, img.nc(), impl::png_type_gray, 16);
		} else {
			// convert from whatever this is to 16bit grayscale
			array2d<test::uint16> temp_img;
			assign_image(temp_img, img_);
			for (unsigned long i = 0; i < row_pointers.size(); ++i)
				row_pointers[i] = (unsigned char*) (&temp_img[i][0]);

			impl::impl_save_png(file_name, row_pointers, img.nc(), impl::png_type_gray, 16);
		}
	}
}

// ----------------------------------------------------------------------------------------

// template <typename EXP>
// void save_png( const matrix_exp<EXP>& img, const std::string& file_name){
// 	array2d<typename EXP::type> temp;
// 	assign_image(temp, img);
// 	save_png(temp, file_name);
// }

// ----------------------------------------------------------------------------------------

// endregion =========================================================================================================================================
} // namespace test
