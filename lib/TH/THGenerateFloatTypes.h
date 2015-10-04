#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#define TH_ABS fabsf
#define TH_ACOS acosf
#define TH_ACOSH acoshf
#define TH_ASIN asinf
#define TH_ASINH asinhf
#define TH_ATAN atanf
#define TH_ATANH atanhf
#define TH_COS cosf
#define TH_COSH coshf
#define TH_EXP expf
#define TH_LOG logf
#define TH_POW powf
#define TH_SIN sinf
#define TH_SINH sinhf
#define TH_SQRT sqrtf
#define TH_TAN tanf
#define TH_TANH tanhf

#define real float
#define accreal double
#define Real Float
#define TH_REAL_IS_FLOAT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef real
#undef Real
#undef TH_REAL_IS_FLOAT

#undef TH_ABS
#undef TH_ACOS
#undef TH_ACOSH
#undef TH_ASIN
#undef TH_ASINH
#undef TH_ATAN
#undef TH_ATANH
#undef TH_COS
#undef TH_COSH
#undef TH_EXP
#undef TH_LOG
#undef TH_POW
#undef TH_SIN
#undef TH_SINH
#undef TH_SQRT
#undef TH_TAN
#undef TH_TANH

#define TH_ABS fabs
#define TH_ACOS acos
#define TH_ACOSH acosh
#define TH_ASIN asin
#define TH_ASINH asinh
#define TH_ATAN atan
#define TH_ATANH atanh
#define TH_COS cos
#define TH_COSH cosh
#define TH_EXP exp
#define TH_LOG log
#define TH_POW pow
#define TH_SIN sin
#define TH_SINH sinh
#define TH_SQRT sqrt
#define TH_TAN tan
#define TH_TANH tanh

#define real double
#define accreal double
#define Real Double
#define TH_REAL_IS_DOUBLE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef real
#undef Real
#undef TH_REAL_IS_DOUBLE

#undef TH_ABS
#undef TH_ACOS
#undef TH_ACOSH
#undef TH_ASIN
#undef TH_ASINH
#undef TH_ATAN
#undef TH_ATANH
#undef TH_COS
#undef TH_COSH
#undef TH_EXP
#undef TH_LOG
#undef TH_POW
#undef TH_SIN
#undef TH_SINH
#undef TH_SQRT
#undef TH_TAN
#undef TH_TANH

#undef TH_GENERIC_FILE
