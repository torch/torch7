#ifndef TH_VECTORIMPL_INC
#define TH_VECTORIMPL_INC

#include "THGeneral.h"

#define THVector_(NAME) TH_CONCAT_4(TH,Real,Vector_,NAME)

TH_API void THDoubleVector_fill_SSE(double *x, const double c, const long n);
TH_API void THDoubleVector_add_SSE(double *y, const double *x, const double c, const long n);
TH_API void THDoubleVector_diff_SSE(double *z, const double *x, const double *y, const long n);
TH_API void THDoubleVector_scale_SSE(double *y, const double c, const long n);
TH_API void THDoubleVector_mul_SSE(double *y, const double *x, const long n);
TH_API void THFloatVector_fill_SSE(float *x, const float c, const long n);
TH_API void THFloatVector_add_SSE(float *y, const float *x, const float c, const long n);
TH_API void THFloatVector_diff_SSE(float *z, const float *x, const float *y, const long n);
TH_API void THFloatVector_scale_SSE(float *y, const float c, const long n);
TH_API void THFloatVector_mul_SSE(float *y, const float *x, const long n);

#include "generic/THVectorDefault.h"
#include "THGenerateAllTypes.h"

#endif
