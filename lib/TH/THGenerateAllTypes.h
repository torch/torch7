#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#define THTypeIdxByte   1
#define THTypeIdxChar   2
#define THTypeIdxShort  3
#define THTypeIdxInt    4
#define THTypeIdxLong   5
#define THTypeIdxFloat  6
#define THTypeIdxDouble 7
#define THTypeIdxHalf   8
#define THTypeIdx_(T) TH_CONCAT_2(THTypeIdx,T)

#define real unsigned char
#define accreal long
#define Real Byte
#define THInf UCHAR_MAX
#define TH_REAL_IS_BYTE
#line 1 TH_GENERIC_FILE
/*#line 1 "THByteStorage.h"*/
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_BYTE

#define real char
#define accreal long
#define Real Char
#define THInf CHAR_MAX
#define TH_REAL_IS_CHAR
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_CHAR

#define real short
#define accreal long
#define Real Short
#define THInf SHRT_MAX
#define TH_REAL_IS_SHORT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_SHORT

#define real int
#define accreal long
#define Real Int
#define THInf INT_MAX
#define TH_REAL_IS_INT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_INT

#define real long
#define accreal long
#define Real Long
#define THInf LONG_MAX
#define TH_REAL_IS_LONG
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_LONG

#define real float
#define accreal double
#define Real Float
#define THInf FLT_MAX
#define TH_REAL_IS_FLOAT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define THInf DBL_MAX
#define TH_REAL_IS_DOUBLE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_DOUBLE

#if TH_GENERIC_USE_HALF
#include "THHalf.h"
#define real half
#define accreal float
#define Real Half
#define THInf TH_HALF_MAX
#define TH_REAL_IS_HALF
#if !TH_NATIVE_HALF
# define TH_GENERIC_NO_MATH 1
#endif
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_HALF
#undef TH_GENERIC_NO_MATH
#endif

#undef TH_GENERIC_FILE
