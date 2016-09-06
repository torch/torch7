#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVectorDefault.h"
#else

TH_API void THVector_(fill_DEFAULT)(real *x, const real c, const long n);
TH_API void THVector_(add_DEFAULT)(real *y, const real *x, const real c, const long n);
TH_API void THVector_(diff_DEFAULT)(real *z, const real *x, const real *y, const long n);
TH_API void THVector_(scale_DEFAULT)(real *y, const real c, const long n);
TH_API void THVector_(mul_DEFAULT)(real *y, const real *x, const long n);

#endif
