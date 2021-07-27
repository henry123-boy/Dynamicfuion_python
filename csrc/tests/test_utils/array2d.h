// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See https://github.com/davisking/dlib/blob/master/LICENSE.txt for the full license.
#pragma once
namespace test {

template<typename T>
class array2d /* : public enumerable<T> */ /* original dlib had an enumerable pure abstract class / interface */ {

	/*!
		INITIAL VALUE
			- nc_ == 0
			- nr_ == 0
			- data == nullptr
			- at_start_ == true
			- cur == 0
			- last == 0

		CONVENTION
			- nc_ == nc()
			- nr_ == nc()
			- if (data != nullptr) then
				- last == a pointer to the last element in the data array
				- data == pointer to an array of nc_*nr_ T objects
			- else
				- nc_ == 0
				- nr_ == 0
				- data == nullptr
				- last == 0


			- nr_ * nc_ == size()
			- if (cur == 0) then
				- current_element_valid() == false
			- else
				- current_element_valid() == true
				- *cur == element()

			- at_start_ == at_start()
	!*/


	class row_helper;
public:

	typedef T type;
	typedef T* iterator;
	typedef const T* const_iterator;


	// -----------------------------------

	class row {
		/*!
			CONVENTION
				- nc_ == nc()
				- for all x < nc_:
					- (*this)[x] == data[x]
		!*/

		friend class array2d<T>;
		friend class row_helper;

	public:
		long nc() const { return nc_; }

		const T& operator[](long column) const {
			return data[column];
		}

		T& operator[](long column) {
			return data[column];
		}

		// restricted functions
		row() = delete;
		row& operator=(row&) = delete;

	private:
		row(T* data_, long cols) : data(data_), nc_(cols) {}

		T* data;
		long nc_;
	};

	// -----------------------------------

	array2d() : data(0), nc_(0), nr_(0), cur(0), last(0), at_start_(true) {}

	array2d(long rows, long cols) :
			data(0), nc_(0), nr_(0), cur(0), last(0), at_start_(true) {
		set_size(rows, cols);
	}

	array2d(const array2d&) = delete;        // copy constructor
	array2d& operator=(const array2d&) = delete;    // assignment operator


	array2d(array2d&& item) : array2d() {
		swap(item);
	}

	array2d& operator=(
			array2d&& rhs
	) {
		swap(rhs);
		return *this;
	}


	virtual ~array2d() { clear(); }

	long nc() const { return nc_; }

	long nr() const { return nr_; }

	row operator[](long row_) {
		return row(data + row_ * nc_, nc_);
	}

	const row operator[](long row_) const {
		return row(data + row_ * nc_, nc_);
	}

	void swap(array2d& item) {
		std::swap(data, item.data);
		std::swap(nr_, item.nr_);
		std::swap(nc_, item.nc_);
		std::swap(at_start_, item.at_start_);
		std::swap(cur, item.cur);
		std::swap(last, item.last);
	}

	void clear() {
		if (data != nullptr) {
			delete data;
			nc_ = 0;
			nr_ = 0;
			data = 0;
			at_start_ = true;
			cur = 0;
			last = 0;
		}
	}

	void set_size(long rows, long cols);

	bool at_start(
	) const { return at_start_; }

	void reset(
	) const {
		at_start_ = true;
		cur = 0;
	}

	bool current_element_valid(
	) const { return (cur != 0); }

	const T& element() const {
		return *cur;
	}

	T& element() {
		return *cur;
	}

	bool move_next() const {
		if (cur != 0) {
			if (cur != last) {
				++cur;
				return true;
			}
			cur = 0;
			return false;
		} else if (at_start_) {
			cur = data;
			at_start_ = false;
			return (data != nullptr);
		} else {
			return false;
		}
	}

	size_t size() const { return static_cast<size_t>(nc_) * static_cast<size_t>(nr_); }

	long width_step() const {
		return nc_ * sizeof(T);
	}

	iterator begin() {
		return data;
	}

	iterator end() {
		return data + size();
	}

	const_iterator begin() const {
		return data;
	}

	const_iterator end() const {
		return data + size();
	}


private:


	T* data;
	long nc_;
	long nr_;

	mutable T* cur;
	T* last;
	mutable bool at_start_;

};

// ----------------------------------------------------------------------------------------

template<typename T>
inline void swap(array2d<T>& a, array2d<T>& b) {
	a.swap(b);
}

//
// template< typename T>
// void serialize( const array2d<T>& item, std::ostream& out) {
// 	try {
// 		// The reason the serialization is a little funny is because we are trying to
// 		// maintain backwards compatibility with an older serialization format used by
// 		// dlib while also encoding things in a way that lets the array2d and matrix
// 		// objects have compatible serialization formats.
// 		serialize(-item.nr(), out);
// 		serialize(-item.nc(), out);
//
// 		item.reset();
// 		while (item.move_next())
// 			serialize(item.element(), out);
// 		item.reset();
// 	}
// 	catch (serialization_error& e) {
// 		throw serialization_error(e.info + "\n   while serializing object of type array2d");
// 	}
// }
//
// template< typename T>
// void deserialize( array2d<T>& item, std::istream& in) {
// 	try {
// 		long nr, nc;
// 		deserialize(nr, in);
// 		deserialize(nc, in);
//
// 		// this is the newer serialization format
// 		if (nr < 0 || nc < 0) {
// 			nr *= -1;
// 			nc *= -1;
// 		} else {
// 			std::swap(nr, nc);
// 		}
//
// 		item.set_size(nr, nc);
//
// 		while (item.move_next())
// 			deserialize(item.element(), in);
// 		item.reset();
// 	} catch (serialization_error& e) {
// 		item.clear();
// 		throw serialization_error(e.info + "\n   while deserializing object of type array2d");
// 	}
// }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// member function definitions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template<typename T>
void array2d<T>::set_size(long rows, long cols) {
	// set the enumerator back at the start
	at_start_ = true;
	cur = 0;

	// don't do anything if we are already the right size.
	if (nc_ == cols && nr_ == rows) {
		return;
	}

	nc_ = cols;
	nr_ = rows;

	// free any existing memory
	if (data != nullptr) {
		delete data;
		data = 0;
	}

	// now setup this object to have the new size
	try {
		if (nr_ > 0) {
			data = new T[nr_ * nc_];
			last = data + nr_ * nc_ - 1;
		}
	}
	catch (...) {
		if (data) {
			delete data;
		}

		data = 0;
		nc_ = 0;
		nr_ = 0;
		last = 0;
		throw;
	}
}

// ----------------------------------------------------------------------------------------
struct default_is_type_value {
	static const bool value = false;
};

template<typename T>
struct is_array2d : public default_is_type_value {
	/*!
		- if (T is an implementation of array2d/array2d_kernel_abstract.h) then
			- is_array2d<T>::value == true
		- else
			- is_array2d<T>::value == false
	!*/
};

template<typename T>
struct is_array2d<array2d<T> > {
	const static bool value = true;
};

// ----------------------------------------------------------------------------------------
// region ================================================== generic image functions =================================================================
template<typename T>
struct image_traits<array2d<T>> {
	typedef T pixel_type;
};
template<typename T>
struct image_traits<const array2d<T>> {
	typedef T pixel_type;
};

template<typename T>
inline long num_rows(const array2d<T>& img) { return img.nr(); }

template<typename T>
inline long num_columns(const array2d<T>& img) { return img.nc(); }

template<typename T>
inline void set_image_size(array2d<T>& img, long rows, long cols) { img.set_size(rows, cols); }

template<typename T>
inline void* image_data(array2d<T>& img) {
	if (img.size() != 0)
		return &img[0][0];
	else
		return 0;
}

template<typename T>
inline const void* image_data(
		const array2d<T>& img
) {
	if (img.size() != 0)
		return &img[0][0];
	else
		return 0;
}

template<typename T>
inline long width_step(const array2d<T>& img) {
	return img.width_step();
}

// endregion =========================================================================================================================================
} // namespace test