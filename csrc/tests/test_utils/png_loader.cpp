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

#include "png_loader.h"
#include <png.h>
#include <memory>
#include <cstring>
#include <cassert>

namespace test
{

// ----------------------------------------------------------------------------------------

struct LibpngData{
	png_bytep* row_pointers_;
	png_structp png_ptr_;
	png_infop info_ptr_;
	png_infop end_info_;
};

struct PngBufferReaderState
{
	const unsigned char* buffer_;
	size_t buffer_size_;
	size_t current_pos_;
};

struct FileInfo
{
	FileInfo( FILE *fp, const char* filename ) : fp_( fp ), filename_( filename )
	{
	}

	FileInfo( const unsigned char* buffer, size_t buffer_size ) : buffer_( buffer ), buffer_size_( buffer_size )
	{
	}

	// no copying this object.
	FileInfo(const FileInfo&) = delete;
	FileInfo& operator=(const FileInfo&) = delete;

	~FileInfo()
	{
		if ( fp_ != nullptr ) fclose( fp_ );
	}

	FILE* fp_{nullptr};
	const char* filename_{nullptr};
	const unsigned char* buffer_{nullptr};
	size_t buffer_size_{0};
};

// ----------------------------------------------------------------------------------------

png_loader::
png_loader( const char* filename ) : height_( 0 ), width_( 0 )
{
	read_image( check_file( filename ) );
}

// ----------------------------------------------------------------------------------------

png_loader::
png_loader( const std::string& filename ) : height_( 0 ), width_( 0 ){
	read_image( check_file( filename.c_str() ) );
}

// ----------------------------------------------------------------------------------------

// png_loader::
// png_loader( const dlib::file& f ) : height_( 0 ), width_( 0 ){
// 	read_image( check_file( f.full_name().c_str() ) );
// }

// ----------------------------------------------------------------------------------------

png_loader::
png_loader( const unsigned char* image_buffer, size_t buffer_size ) : height_( 0 ), width_( 0 )
{
	read_image( std::unique_ptr<FileInfo>( new FileInfo( image_buffer, buffer_size ) ) );
}

// ----------------------------------------------------------------------------------------

const unsigned char* png_loader::get_row( unsigned i ) const
{
	return ld_->row_pointers_[i];
}

// ----------------------------------------------------------------------------------------

png_loader::~png_loader()
{
	if ( ld_ && ld_->row_pointers_ != nullptr )
		png_destroy_read_struct( &( ld_->png_ptr_ ), &( ld_->info_ptr_ ), &( ld_->end_info_ ) );
}

// ----------------------------------------------------------------------------------------

bool png_loader::is_gray() const
{
	return ( color_type_ == PNG_COLOR_TYPE_GRAY );
}

// ----------------------------------------------------------------------------------------

bool png_loader::is_graya() const
{
	return ( color_type_ == PNG_COLOR_TYPE_GRAY_ALPHA );
}

// ----------------------------------------------------------------------------------------

bool png_loader::is_rgb() const
{
	return ( color_type_ == PNG_COLOR_TYPE_RGB );
}

// ----------------------------------------------------------------------------------------

bool png_loader::is_rgba() const
{
	return ( color_type_ == PNG_COLOR_TYPE_RGB_ALPHA );
}

// ----------------------------------------------------------------------------------------

std::unique_ptr<FileInfo> png_loader::check_file( const char* filename ){
	if ( filename == nullptr ){
		throw std::runtime_error("png_loader: invalid filename, it is nullptr");
	}
	FILE *fp = fopen( filename, "rb" );
	if ( !fp ){
		throw std::runtime_error(std::string("png_loader: unable to open file ") + filename);
	}

	return std::unique_ptr<FileInfo>( new FileInfo( fp, filename ) );
}

// ----------------------------------------------------------------------------------------

// Don't do anything when libpng calls us to tell us about an error.  Just return to
// our own code and throw an exception (at the long jump target).
void png_loader_user_error_fn_silent(png_structp  png_struct, png_const_charp )
{
	longjmp(png_jmpbuf(png_struct),1);
}
void png_loader_user_warning_fn_silent(png_structp , png_const_charp )
{
}

void png_buffer_reader(png_structp png_ptr, png_bytep data, size_t length)
{
	PngBufferReaderState* state = static_cast<PngBufferReaderState*>( png_get_io_ptr( png_ptr ) );
	if ( length > ( state->buffer_size_ - state->current_pos_ ) )
	{
		png_error(png_ptr, "png_loader: read error in png_buffer_reader");
	}
	memcpy( data, state->buffer_ + state->current_pos_, length );
	state->current_pos_ += length;
}

template <
		typename T
>
const std::string cast_to_string (
		const T& item
)
{
	std::ostringstream sout;
	sout << item;
	if (!sout){
		throw std::runtime_error("String cast error.");
	}
	return sout.str();
}

void png_loader::read_image( std::unique_ptr<FileInfo> file_info )
{
	assert(file_info);

	ld_ = std::make_unique<LibpngData>();

	constexpr png_size_t png_header_size = 8;
	std::string load_error_info;

	if ( file_info->fp_ != nullptr ){
		png_byte sig[png_header_size];
		if (fread( sig, 1, png_header_size, file_info->fp_ ) != png_header_size){
			throw std::runtime_error(std::string("png_loader: error reading file ") + file_info->filename_);
		}
		load_error_info = std::string(" in file ") + file_info->filename_;
		if ( png_sig_cmp( sig, 0, png_header_size ) != 0 ){
			throw std::runtime_error(std::string("png_loader: format error") + load_error_info);
		}
	} else {
		if ( file_info->buffer_ == nullptr ) {
			throw std::runtime_error(std::string("png_loader: invalid image buffer, it is nullptr"));
		}
		if ( file_info->buffer_size_ == 0 ) {
			throw std::runtime_error(std::string("png_loader: invalid image buffer size, it is 0"));
		}
		if ( file_info->buffer_size_ < png_header_size ||
		     png_sig_cmp( (png_bytep)file_info->buffer_, 0, png_header_size ) != 0 ) {
			throw std::runtime_error(std::string("png_loader: format error in image buffer"));
		}
		buffer_reader_state_ = std::make_unique<PngBufferReaderState>();
	}

	ld_->png_ptr_ = png_create_read_struct( PNG_LIBPNG_VER_STRING, nullptr, &png_loader_user_error_fn_silent, &png_loader_user_warning_fn_silent );
	if ( ld_->png_ptr_ == nullptr )
	{
		std::ostringstream sout;
		sout << "Error, unable to allocate png structure" << std::endl;
		const char* runtime_version = png_get_header_ver(nullptr);
		if (runtime_version && std::strcmp(PNG_LIBPNG_VER_STRING, runtime_version) != 0)
		{
			sout << "This is happening because you compiled against one version of libpng, but then linked to another." << std::endl;
			sout << "Compiled against libpng version:   " << PNG_LIBPNG_VER_STRING << std::endl;
			sout << "Linking to this version of libpng: " << runtime_version << std::endl;
		}
		throw std::runtime_error(sout.str());
	}
	ld_->info_ptr_ = png_create_info_struct( ld_->png_ptr_ );
	if ( ld_->info_ptr_ == nullptr )
	{
		png_destroy_read_struct( &( ld_->png_ptr_ ), ( png_infopp )nullptr, ( png_infopp )nullptr );
		throw std::runtime_error(std::string("png_loader: unable to allocate png info structure") + load_error_info);
	}
	ld_->end_info_ = png_create_info_struct( ld_->png_ptr_ );
	if ( ld_->end_info_ == nullptr )
	{
		png_destroy_read_struct( &( ld_->png_ptr_ ), &( ld_->info_ptr_ ), ( png_infopp )nullptr );
		throw std::runtime_error(std::string("png_loader: unable to allocate png info structure") + load_error_info);
	}

	if (setjmp(png_jmpbuf(ld_->png_ptr_)))
	{
		// If we get here, we had a problem writing the file
		png_destroy_read_struct( &( ld_->png_ptr_ ), &( ld_->info_ptr_ ), &( ld_->end_info_ ) );
		throw std::runtime_error(std::string("png_loader: parse error") + load_error_info);
	}

	png_set_palette_to_rgb(ld_->png_ptr_);

	if ( file_info->fp_ != nullptr )
	{
		png_init_io( ld_->png_ptr_, file_info->fp_ );
	}
	else
	{
		buffer_reader_state_->buffer_ = file_info->buffer_;
		buffer_reader_state_->buffer_size_ = file_info->buffer_size_;
		// skipping header
		buffer_reader_state_->current_pos_ = png_header_size;
		png_set_read_fn( ld_->png_ptr_, buffer_reader_state_.get(), png_buffer_reader );
	}

	png_set_sig_bytes( ld_->png_ptr_, png_header_size );
	// flags force one byte per channel output
	// byte_orderer bo;
	int png_transforms = PNG_TRANSFORM_PACKING;
	// if (bo.host_is_little_endian())
	// WARNING TO YE BIG-ENDIAN PLATFORM USERS: ASSUMES LITTLE-ENDIAN
	png_transforms |= PNG_TRANSFORM_SWAP_ENDIAN;
	png_read_png( ld_->png_ptr_, ld_->info_ptr_, png_transforms, nullptr );
	height_ = png_get_image_height( ld_->png_ptr_, ld_->info_ptr_ );
	width_ = png_get_image_width( ld_->png_ptr_, ld_->info_ptr_ );
	bit_depth_ = png_get_bit_depth( ld_->png_ptr_, ld_->info_ptr_ );
	color_type_ = png_get_color_type( ld_->png_ptr_, ld_-> info_ptr_ );


	if (color_type_ != PNG_COLOR_TYPE_GRAY &&
	    color_type_ != PNG_COLOR_TYPE_RGB &&
	    color_type_ != PNG_COLOR_TYPE_RGB_ALPHA &&
	    color_type_ != PNG_COLOR_TYPE_GRAY_ALPHA)
	{
		png_destroy_read_struct( &( ld_->png_ptr_ ), &( ld_->info_ptr_ ), &( ld_->end_info_ ) );
		throw std::runtime_error(std::string("png_loader: unsupported color type") + load_error_info);
	}

	if (bit_depth_ != 8 && bit_depth_ != 16)
	{
		png_destroy_read_struct( &( ld_->png_ptr_ ), &( ld_->info_ptr_ ), &( ld_->end_info_ ) );
		throw std::runtime_error("png_loader: unsupported bit depth of " + cast_to_string(bit_depth_) + load_error_info);
	}

	ld_->row_pointers_ = png_get_rows( ld_->png_ptr_, ld_->info_ptr_ );

	if ( ld_->row_pointers_ == nullptr )
	{
		png_destroy_read_struct( &( ld_->png_ptr_ ), &( ld_->info_ptr_ ), &( ld_->end_info_ ) );
		throw std::runtime_error(std::string("png_loader: parse error") + load_error_info);
	}
}

// ----------------------------------------------------------------------------------------

} // namespace test