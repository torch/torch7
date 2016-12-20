#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorageCopy.c"
#else

void THStorage_(rawCopy)(THStorage *storage, real *src)
{
  ptrdiff_t i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = src[i];
}

void THStorage_(copy)(THStorage *storage, THStorage *src)
{
  THArgCheck(storage->size == src->size, 2, "size mismatch");
  THStorage_(rawCopy)(storage, src->data);
}

#define IMPLEMENT_THStorage_COPY(TYPENAMESRC) \
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  if(THTypeIdx_(Real) == THTypeIdx_(TYPENAMESRC)) {                   \
    memcpy(storage->data, src->data, sizeof(real)*storage->size);   /* cast just removes compiler warning */ \
  } else {                                                              \
    ptrdiff_t i;                                                        \
    for(i = 0; i < storage->size; i++)                                  \
      storage->data[i] = (real)src->data[i];                            \
    }                                                                   \
}

#define IMPLEMENT_THStorage_COPY_FROM_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size; i++)					\
    storage->data[i] = (real)TH_half2float(src->data[i]);		\
}

#define IMPLEMENT_THStorage_COPY_TO_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size; i++)					\
    storage->data[i] = TH_float2half((float)(src->data[i]));		\
}

#ifndef TH_REAL_IS_HALF
IMPLEMENT_THStorage_COPY(Byte)
IMPLEMENT_THStorage_COPY(Char)
IMPLEMENT_THStorage_COPY(Short)
IMPLEMENT_THStorage_COPY(Int)
IMPLEMENT_THStorage_COPY(Long)
IMPLEMENT_THStorage_COPY(Float)
IMPLEMENT_THStorage_COPY(Double)
#if TH_GENERIC_USE_HALF
IMPLEMENT_THStorage_COPY_FROM_HALF(Half)
#endif
#else
/* only allow pass-through for Half */
IMPLEMENT_THStorage_COPY(Half)
IMPLEMENT_THStorage_COPY_TO_HALF(Byte)
IMPLEMENT_THStorage_COPY_TO_HALF(Char)
IMPLEMENT_THStorage_COPY_TO_HALF(Short)
IMPLEMENT_THStorage_COPY_TO_HALF(Int)
IMPLEMENT_THStorage_COPY_TO_HALF(Long)
IMPLEMENT_THStorage_COPY_TO_HALF(Float)
IMPLEMENT_THStorage_COPY_TO_HALF(Double)
#endif


#endif
