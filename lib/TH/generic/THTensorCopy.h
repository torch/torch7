#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.h"
#else

/* Support for copy between different Tensor types */

TH_API void THTensor_(copy)(THTensor *tensor, THTensor *src);
#ifndef TH_GENERIC_NO_BYTE
TH_API void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
#endif
#ifndef TH_GENERIC_NO_CHAR
TH_API void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
#endif
#ifndef TH_GENERIC_NO_SHORT
TH_API void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
#endif
#ifndef TH_GENERIC_NO_INT
TH_API void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
#endif
#ifndef TH_GENERIC_NO_LONG
TH_API void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
#endif
TH_API void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
#ifndef TH_GENERIC_NO_DOUBLE
TH_API void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);
#endif

#endif
