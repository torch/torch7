#ifndef TORCH_GENERAL_INC
#define TORCH_GENERAL_INC

#include <stdlib.h>
#include <string.h>

#include "luaT.h"
#include "TH.h"

#if (defined(_MSC_VER) || defined(__MINGW32__))

#define snprintf _snprintf
#define popen _popen
#define pclose _pclose

#endif

#endif
