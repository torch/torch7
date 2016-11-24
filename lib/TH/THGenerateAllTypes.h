#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#include <stdint.h>

#define real uint8_t
#define accreal int64_t
#define Real Byte
#define THInf UINT8_MAX
#define TH_REAL_IS_BYTE
#line 1 TH_GENERIC_FILE
/*#line 1 "THByteStorage.h"*/
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_BYTE

#define real int8_t
#define accreal int64_t
#define Real Char
#define THInf INT8_MAX
#define TH_REAL_IS_CHAR
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_CHAR

#define real int16_t
#define accreal int64_t
#define Real Short
#define THInf INT16_MAX
#define TH_REAL_IS_SHORT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_SHORT

#define real int32_t
#define accreal int64_t
#define Real Int
#define THInf INT32_MAX
#define TH_REAL_IS_INT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_INT

#define real int64_t
#define accreal int64_t
#define Real Long
#define THInf INT64_MAX
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

#undef TH_GENERIC_FILE
