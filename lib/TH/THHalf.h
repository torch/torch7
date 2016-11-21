#ifndef _THHALF_H
# define _THHALF_H

#include <stdint.h>

#include "THGeneral.h"

# if defined (TH_HALF_TYPE)
typedef TH_HALF_TYPE half;
# else
/* Neither built-in nor included from Cutorch, use our definition lifted from CUDA */
#if defined(__GNUC__)
#define __align__(n) __attribute__((aligned(n)))
#elif defined(_WIN32)
#define __align__(n) __declspec(align(n))
#else
#define __align__(n)
#endif

typedef struct __align__(2){
  unsigned short x;
} __half;

typedef struct __align__(4) {
    unsigned int x;
} __half2;

typedef __half half;
typedef __half2 half2;
# endif

/* numeric limits */


TH_API half TH_float2half(float a);
TH_API float TH_half2float(half a);

#ifndef TH_HALF_BITS_TO_LITERAL
# define TH_HALF_BITS_TO_LITERAL(n) { n }
#endif

#define TH_HALF_ZERO TH_HALF_BITS_TO_LITERAL(0x0)
#define TH_HALF_MIN TH_HALF_BITS_TO_LITERAL(0x0400)
#define TH_HALF_MAX TH_HALF_BITS_TO_LITERAL(0x7BFF)
#define TH_HALF_EPSILON TH_HALF_BITS_TO_LITERAL(0x1400)
#define TH_HALF_INF  TH_HALF_BITS_TO_LITERAL(0x7C00)
#define TH_HALF_QNAN TH_HALF_BITS_TO_LITERAL(0x7FFF)
#define TH_HALF_SNAN TH_HALF_BITS_TO_LITERAL(0x7DFF)
#define TH_HALF_DENORM_MIN TH_HALF_BITS_TO_LITERAL(0x0001)
#define TH_HALF_DIGITS 11
#define TH_HALF_DIGITS10 3
#define TH_HALF_DIGITS10_MAX 5
#define TH_HALF_MAX_EXPONENT   16
#define TH_HALF_MAX_EXPONENT10 4

#endif
