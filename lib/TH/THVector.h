#ifndef TH_VECTOR_INC
#define TH_VECTOR_INC

#define THVector_(NAME) TH_CONCAT_4(TH,Real,Vector_,NAME)

#if defined(__NEON__)

/* We don't have dynamic dispatch for ARM, so we include all of the vector
 * functions as inlines in this header */
#include "THVectorImpls.h"

#else

/* We are going to use dynamic dispatch, and want only to generate declarations
 * of the vector functions */
#include "generic/THVector.h"
#include "THGenerateAllTypes.h"

#endif

#endif // TH_VECTOR_INC
