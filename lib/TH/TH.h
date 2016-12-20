#ifndef TH_INC
#define TH_INC

#include "THGeneral.h"

#if TH_GENERIC_USE_HALF
# include "THHalf.h"
#endif

#include "THBlas.h"
#ifdef USE_LAPACK
#include "THLapack.h"
#endif

#include "THAtomic.h"
#include "THVector.h"
#include "THLogAdd.h"
#include "THRandom.h"
#include "THStorage.h"
#include "THTensor.h"
#include "THTensorApply.h"
#include "THTensorDimApply.h"

#include "THFile.h"
#include "THDiskFile.h"
#include "THMemoryFile.h"

#endif
