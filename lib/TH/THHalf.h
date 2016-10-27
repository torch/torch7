#ifndef _THHALF_H
# define _THHALF_H

#include <stdint.h>

#include "THGeneral.h"

/* Lifted from CUDA */
typedef struct {
  unsigned short x;
} TH_half;

typedef struct {
    unsigned int x;
} TH_half2;

/* numeric limits */

#define TH_HALF_ZERO 0x0
#define TH_HALF_MIN 0x0400
#define TH_HALF_MAX 0x7BFF
#define TH_HALF_EPSILON 0x1400
#define TH_HALF_INF  0x7C00
#define TH_HALF_QNAN 0x7FFF
#define TH_HALF_SNAN 0x7DFF
#define TH_HALF_DENORM_MIN 0x0001
#define TH_HALF_DIGITS 11
#define TH_HALF_DIGITS10 3
#define TH_HALF_DIGITS10_MAX 5
#define TH_HALF_MAX_EXPONENT   16
#define TH_HALF_MAX_EXPONENT10 4

TH_API TH_half TH_float2half(float a);
TH_API float TH_half2float(TH_half a);

#endif
