// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See https://github.com/davisking/dlib/blob/master/LICENSE.txt for the full license.
#pragma once

#include "pixel.h"

namespace test{

template <typename image_type>
struct image_traits;

// region ============================================ image view ==============================================
template <typename image_type>
class image_view
{
	/*!
		REQUIREMENTS ON image_type
			image_type must be an image object as defined at the top of this file.

		WHAT THIS OBJECT REPRESENTS
			This object takes an image object and wraps it with an interface that makes
			it look like a dlib::array2d.  That is, it makes it look similar to a
			regular 2-dimensional C style array, making code which operates on the
			pixels simple to read.

			Note that an image_view instance is valid until the image given to its
			constructor is modified through an interface other than the image_view
			instance.  This is because, for example, someone might cause the underlying
			image object to reallocate its memory, thus invalidating the pointer to its
			pixel data stored in the image_view.

			As an side, the reason why this object stores a pointer to the image
			object's data and uses that pointer instead of calling image_data() each
			time a pixel is accessed is to allow for image objects to implement
			complex, and possibly slow, image_data() functions.  For example, an image
			object might perform some kind of synchronization between a GPU and the
			host memory during a call to image_data().  Therefore, we call image_data()
			only in image_view's constructor to avoid the performance penalty of
			calling it for each pixel access.
	!*/

public:
	typedef typename image_traits<image_type>::pixel_type pixel_type;

	image_view(image_type& img) :
			_data(reinterpret_cast<char*>(image_data(img))),
			_width_step(width_step(img)),
			_nr(num_rows(img)),
			_nc(num_columns(img)),
			_img(&img)
	{}

	long nr() const { return _nr; }
	/*!
		ensures
			- returns the number of rows in this image.
	!*/

	long nc() const { return _nc; }
	/*!
		ensures
			- returns the number of columns in this image.
	!*/

	unsigned long size() const { return static_cast<unsigned long>(nr()*nc()); }
	/*!
		ensures
			- returns the number of pixels in this image.
	!*/

#ifndef ENABLE_ASSERTS
	pixel_type* operator[] (long row) { return (pixel_type*)(_data+_width_step*row); }
	/*!
		requires
			- 0 <= row < nr()
		ensures
			- returns a pointer to the first pixel in the row-th row.  Therefore, the
			  pixel at row and column position r,c can be accessed via (*this)[r][c].
	!*/

	const pixel_type* operator[] (long row) const { return (const pixel_type*)(_data+_width_step*row); }
	/*!
		requires
			- 0 <= row < nr()
		ensures
			- returns a const pointer to the first pixel in the row-th row.  Therefore,
			  the pixel at row and column position r,c can be accessed via
			  (*this)[r][c].
	!*/
#else
	// If asserts are enabled then we need to return a proxy class so we can make sure
        // the column accesses don't go out of bounds.
        struct pix_row
        {
            pix_row(pixel_type* data_, long nc_) : data(data_),_nc(nc_) {}
            const pixel_type& operator[] (long col) const
            {
                DLIB_ASSERT(0 <= col && col < _nc,
                    "\t The given column index is out of range."
                    << "\n\t col: " << col
                    << "\n\t _nc: " << _nc);
                return data[col];
            }
            pixel_type& operator[] (long col)
            {
                DLIB_ASSERT(0 <= col && col < _nc,
                    "\t The given column index is out of range."
                    << "\n\t col: " << col
                    << "\n\t _nc: " << _nc);
                return data[col];
            }
        private:
            pixel_type* const data;
            const long _nc;
        };
        pix_row operator[] (long row)
        {
            DLIB_ASSERT(0 <= row && row < _nr,
                "\t The given row index is out of range."
                << "\n\t row: " << row
                << "\n\t _nr: " << _nr);
            return pix_row((pixel_type*)(_data+_width_step*row), _nc);
        }
        const pix_row operator[] (long row) const
        {
            DLIB_ASSERT(0 <= row && row < _nr,
                "\t The given row index is out of range."
                << "\n\t row: " << row
                << "\n\t _nr: " << _nr);
            return pix_row((pixel_type*)(_data+_width_step*row), _nc);
        }
#endif

	void set_size(long rows, long cols)
	/*!
		requires
			- rows >= 0 && cols >= 0
		ensures
			- Tells the underlying image to resize itself to have the given number of
			  rows and columns.
			- #nr() == rows
			- #nc() == cols
	!*/
	{
		if(cols < 0 || rows < 0){

			std::stringstream ss;
		            ss << "\t image_view::set_size(long rows, long cols)"
				            << "\n\t The images can't have negative rows or columns."
				            << "\n\t cols: " << cols
				            << "\n\t rows: " << rows;
            throw std::runtime_error(ss.str());
		}
		set_image_size(*_img, rows, cols); *this = *_img;
	}

	void clear() { set_size(0,0); }
	/*!
		ensures
			- sets the image to have 0 pixels in it.
	!*/

	long get_width_step() const { return _width_step; }

private:

	char* _data;
	long _width_step;
	long _nr;
	long _nc;
	image_type* _img;
};

// endregion ===================================================================================================
// region ========================================= generic image functions ====================================
template <typename T>
inline void* image_data( image_view<T>& img)
{
	if (img.size() != 0)
		return &img[0][0];
	else
		return nullptr;
}

template <typename T>
inline const void* image_data(
		const image_view<T>& img
)
{
	if (img.size() != 0)
		return &img[0][0];
	else
		return nullptr;
}


// endregion ===================================================================================================
// region ========================================= assign pixels ==============================================
// ----------------------------------------------------------------------------------------

template <
		typename dest_image_type,
		typename src_pixel_type
>
void assign_all_pixels (
		image_view<dest_image_type>& dest_img,
		const src_pixel_type& src_pixel
)
{
	for (long r = 0; r < dest_img.nr(); ++r)
	{
		for (long c = 0; c < dest_img.nc(); ++c)
		{
			assign_pixel(dest_img[r][c], src_pixel);
		}
	}
}

// ----------------------------------------------------------------------------------------

template <
		typename dest_image_type,
		typename src_pixel_type
>
void assign_all_pixels (
		dest_image_type& dest_img_,
		const src_pixel_type& src_pixel
)
{
	image_view<dest_image_type> dest_img(dest_img_);
	assign_all_pixels(dest_img, src_pixel);
}


// endregion ===================================================================================================
// region ================================= const image view ===================================================

template <typename image_type>
class const_image_view{
	/*!
		REQUIREMENTS ON image_type
			image_type must be an image object as defined at the top of this file.

		WHAT THIS OBJECT REPRESENTS
			This object is just like the image_view except that it provides a "const"
			view into an image.  That is, it has the same interface as image_view
			except that you can't modify the image through a const_image_view.
	!*/

public:
	typedef typename image_traits<image_type>::pixel_type pixel_type;

	const_image_view(
			const image_type& img
	) :
			_data(reinterpret_cast<const char*>(image_data(img))),
			_width_step(width_step(img)),
			_nr(num_rows(img)),
			_nc(num_columns(img))
	{}

	long nr() const { return _nr; }
	long nc() const { return _nc; }
	unsigned long size() const { return static_cast<unsigned long>(nr()*nc()); }
#ifndef ENABLE_ASSERTS
	const pixel_type* operator[] (long row) const { return (const pixel_type*)(_data+_width_step*row); }
#else
	// If asserts are enabled then we need to return a proxy class so we can make sure
        // the column accesses don't go out of bounds.
        struct pix_row
        {
            pix_row(pixel_type* data_, long nc_) : data(data_),_nc(nc_) {}
            const pixel_type& operator[] (long col) const
            {
                DLIB_ASSERT(0 <= col && col < _nc,
                    "\t The given column index is out of range."
                    << "\n\t col: " << col
                    << "\n\t _nc: " << _nc);
                return data[col];
            }
        private:
            pixel_type* const data;
            const long _nc;
        };
        const pix_row operator[] (long row) const
        {
            DLIB_ASSERT(0 <= row && row < _nr,
                "\t The given row index is out of range."
                << "\n\t row: " << row
                << "\n\t _nr: " << _nr);
            return pix_row((pixel_type*)(_data+_width_step*row), _nc);
        }
#endif

	long get_width_step() const { return _width_step; }

private:
	const char* _data;
	long _width_step;
	long _nr;
	long _nc;
};

// endregion

} // namespace test