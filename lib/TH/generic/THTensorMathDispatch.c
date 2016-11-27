#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMathDispatch.c"
#else

// This file will contain static function pointers that will be initialized by
// the initialization call. It will also have globally linked dispatch stubs
// which delegate to the function pointers. The dispatch stubs will be the symbols
// called by clients
static void (*THTensor_(add_DISPATCHPTR))(THTensor *, THTensor *, const real) = &THTensor_(add_DEFAULT);
static FunctionDescription THTensor_(add_DISPATCHTABLE)[] = {
  #if defined(USE_AVX)
    #if defined(TH_REAL_IS_DOUBLE)
      FUNCTION_IMPL(THTensor_(add_AVX), SIMDExtension_AVX),
    #endif
  #endif

  FUNCTION_IMPL(THTensor_(add_DEFAULT), SIMDExtension_DEFAULT)
};

// Dispatch stubs that just call the pointers
TH_API void THTensor_(add)(THTensor *r_, THTensor *t, real value) {
  THTensor_(add_DISPATCHPTR)(r_, t, value);
}

int THTensor_(tensorMathDispatchInit)()
{
  uint32_t hostSimdExts = detectHostSIMDExtensions();
  // Initialize the dispatch pointers to point to the correct functions
  INIT_TENSOR_DISPATCH_PTR(add);
  return 0;
}

#endif
