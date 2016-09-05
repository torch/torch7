#include "THVector.h"
#include <assert.h>
#include "generic/simd/simd.h"

#ifdef __NEON__
#include "vector/NEON.c"
#else
#include "vector/SSE.c"
#endif

#include "generic/THVectorDefault.c"
#include "THGenerateAllTypes.h"

#include "generic/THVectorDispatch.c"
#include "THGenerateAllTypes.h"
