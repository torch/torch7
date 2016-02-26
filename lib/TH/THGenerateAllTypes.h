#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#define real unsigned char
#define accreal long
#define Real Byte
#define TH_REAL_IS_BYTE
#line 1 TH_GENERIC_FILE
/*#line 1 "THByteStorage.h"*/
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_BYTE

#define real unsigned short
#define accreal long
#define Real UShort
#define TH_REAL_IS_USHORT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_USHORT

#define real unsigned int
#define accreal unsigned long
#define Real UInt
#define TH_REAL_IS_UINT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_UINT

#define real unsigned long
#define accreal unsigned long
#define Real ULong
#define TH_REAL_IS_ULONG
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_ULONG

#define real char
#define accreal long
#define Real Char
#define TH_REAL_IS_CHAR
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_CHAR

#define real short
#define accreal long
#define Real Short
#define TH_REAL_IS_SHORT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_SHORT

#define real int
#define accreal long
#define Real Int
#define TH_REAL_IS_INT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_INT

#define real long
#define accreal long
#define Real Long
#define TH_REAL_IS_LONG
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_LONG

#define real float
#define accreal double
#define Real Float
#define TH_REAL_IS_FLOAT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define TH_REAL_IS_DOUBLE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef TH_REAL_IS_DOUBLE

#undef TH_GENERIC_FILE
