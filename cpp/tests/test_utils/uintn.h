// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See https://github.com/davisking/dlib/blob/master/LICENSE.txt for the full license.
#pragma once
namespace test{

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

} // namespace test