#ifndef TH_LAPACK_INC
#define TH_LAPACK_INC

#include "THGeneral.h"

#define THErrorHandler(X) X

#define THLapack_(NAME) TH_CONCAT_4(TH,Real,Lapack_,NAME)

#define THLapackCheck(fmt, func, info , ...) \
  THLapackCheckWithHandler(fmt, THErrorHandler(), func, info, ##__VA_ARGS__)

#define THLapackCheckWithHandler(fmt, cleanup, func, info , ...)    \
if (info < 0) {                                                     \
  cleanup                                                           \
  THError("Lapack Error in %s : Illegal Argument %d", func, -info); \
} else if(info > 0) {                                               \
  cleanup                                                           \
  THError(fmt, func, info, ##__VA_ARGS__);                          \
}                                                                   \

#include "generic/THLapack.h"
#include "THGenerateAllTypes.h"

#endif
