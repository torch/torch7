#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else

void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = (real)(*src_data);)
}

#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

#ifndef TH_GENERIC_NO_BYTE
IMPLEMENT_THTensor_COPY(Byte, unsigned char)
#endif
#ifndef TH_GENERIC_NO_CHAR
IMPLEMENT_THTensor_COPY(Char, char)
#endif
#ifndef TH_GENERIC_NO_SHORT
IMPLEMENT_THTensor_COPY(Short, short)
#endif
#ifndef TH_GENERIC_NO_INT
IMPLEMENT_THTensor_COPY(Int, int)
#endif
#ifndef TH_GENERIC_NO_LONG
IMPLEMENT_THTensor_COPY(Long, long)
#endif
IMPLEMENT_THTensor_COPY(Float, float)
#ifndef TH_GENERIC_NO_DOUBLE
IMPLEMENT_THTensor_COPY(Double, double)
#endif
#endif
