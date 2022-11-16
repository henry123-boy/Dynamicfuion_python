// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See https://github.com/davisking/dlib/blob/master/LICENSE.txt for the full license.
#include <iostream>

#include "entropy_range_encoder.h"

namespace test {
// ----------------------------------------------------------------------------------------

entropy_range_encoder::entropy_range_encoder() :
		initial_low(0x00000001),
		initial_high(0xffffffff),
		out(nullptr),
		low(initial_low),
		high(initial_high) {
}

// ----------------------------------------------------------------------------------------

entropy_range_encoder::
~entropy_range_encoder(
) {
	try {
		if (out != nullptr) {
			flush();
		}
	} catch (...) {}
}

// ----------------------------------------------------------------------------------------

void entropy_range_encoder::
clear(
) {
	if (out != nullptr) {
		flush();
	}
	out = 0;
}

// ----------------------------------------------------------------------------------------

void entropy_range_encoder::
set_stream(
		std::ostream& out_
) {
	if (out != nullptr) {
		// if a stream is currently set then flush the buffers to it before
		// we switch to the new stream
		flush();
	}

	out = &out_;
	streambuf = out_.rdbuf();

	// reset the encoder state
	low = initial_low;
	high = initial_high;
}

// ----------------------------------------------------------------------------------------

bool entropy_range_encoder::
stream_is_set(
) const {
	if (out != nullptr)
		return true;
	else
		return false;
}

// ----------------------------------------------------------------------------------------

std::ostream& entropy_range_encoder::
get_stream(
) const {
	return *out;
}

// ----------------------------------------------------------------------------------------

void entropy_range_encoder::
encode(
		uint32 low_count,
		uint32 high_count,
		uint32 total
) {
	// note that we must add one because of the convention that
	// high == the real upper range minus 1
	uint32 r = (high - low + 1) / total;

	// note that we must subtract 1 to preserve the convention that
	// high == the real upper range - 1
	high = low + r * high_count - 1;
	low = low + r * low_count;


	while (true) {
		// if high and low don't have the same 8 high order bits
		if ((high & 0xFF000000) != (low & 0xFF000000)) {
			// if the distance between high and low is small and there aren't
			// any bits we can roll off then force high and low to have common high
			// order bits.
			if ((high - low < 0x10000)) {
				if (high - low > 0x1000) {
					high >>= 1;
					low >>= 1;
					high = low = high + low;
					high += 0xFF;
					low -= 0xFF;
				} else /**/
				{
					high >>= 1;
					low >>= 1;
					high = low = high + low;
				}
			} else {
				// there are no bits to roll off and high and low are not
				// too close so just quit the loop
				break;
			}

		}
			// else if there are 8 bits we can roll off
		else {
			// write the 8 high order bits from low into buf
			auto buf = static_cast<unsigned char>(low >> 24);


			// roll off the bits we just wrote to buf
			high <<= 8;
			low <<= 8;
			high |= 0xFF;  // note that it is ok to add 0xFF to high here because
			// of the convention that high == real upper range - 1.
			// so that means that if we want to shift the upper range
			// left by one then we must shift a one into high also
			// since real upper range == high + 0.999999999...

			// make sure low is never zero
			if (low == 0)
				low = 1;

			// write buf to the output stream
			if (streambuf->sputn(reinterpret_cast<char*>(&buf), 1) == 0) {
				throw std::ios_base::failure("error occurred in the entropy_encoder object");
			}

		}

	} // while (true)

}

// ----------------------------------------------------------------------------------------

void entropy_range_encoder::
flush(
) {

	// flush low to the output stream


	unsigned char buf;


	buf = static_cast<unsigned char>((low >> 24) & 0xFF);
	if (streambuf->sputn(reinterpret_cast<char*>(&buf), 1) == 0){
		throw std::ios_base::failure("error occurred in the entropy_encoder object");
	}

	buf = static_cast<unsigned char>((low >> 16) & 0xFF);
	if (streambuf->sputn(reinterpret_cast<char*>(&buf), 1) == 0) {
		throw std::ios_base::failure("error occurred in the entropy_encoder object");
	}

	buf = static_cast<unsigned char>((low >> 8) & 0xFF);
	if (streambuf->sputn(reinterpret_cast<char*>(&buf), 1) == 0){
		throw std::ios_base::failure("error occurred in the entropy_encoder object");
	}

	buf = static_cast<unsigned char>((low) & 0xFF);
	if (streambuf->sputn(reinterpret_cast<char*>(&buf), 1) == 0){
		throw std::ios_base::failure("error occurred in the entropy_encoder object");
	}

	// make sure the stream buffer flushes to its I/O channel
	streambuf->pubsync();

	// reset the encoder state
	low = initial_low;
	high = initial_high;
}

// ----------------------------------------------------------------------------------------
} // namespace test