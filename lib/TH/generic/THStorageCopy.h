#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorageCopy.h"
#else

/* Support for copy between different Storage types */

TH_API void THStorage_(rawCopy)(THStorage *storage, real *src);
TH_API void THStorage_(copy)(THStorage *storage, THStorage *src);
#ifndef TH_GENERIC_NO_BYTE
TH_API void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
#endif
#ifndef TH_GENERIC_NO_CHAR
TH_API void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
#endif
#ifndef TH_GENERIC_NO_SHORT
TH_API void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
#endif
#ifndef TH_GENERIC_NO_INT
TH_API void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
#endif
#ifndef TH_GENERIC_NO_LONG
TH_API void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
#endif
TH_API void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
#ifndef TH_GENERIC_NO_DOUBLE
TH_API void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
#endif
#endif
