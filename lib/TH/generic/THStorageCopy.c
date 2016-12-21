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
  ptrdiff_t i; \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  for(i = 0; i < storage->size; i++) \
    storage->data[i] = (real)src->data[i]; \
}

#ifndef TH_GENERIC_NO_BYTE
IMPLEMENT_THStorage_COPY(Byte)
#endif
#ifndef TH_GENERIC_NO_CHAR
IMPLEMENT_THStorage_COPY(Char)
#endif
#ifndef TH_GENERIC_NO_SHORT
IMPLEMENT_THStorage_COPY(Short)
#endif
#ifndef TH_GENERIC_NO_INT
IMPLEMENT_THStorage_COPY(Int)
#endif
#ifndef TH_GENERIC_NO_LONG
IMPLEMENT_THStorage_COPY(Long)
#endif
/* float is always implemented */
IMPLEMENT_THStorage_COPY(Float)
#ifndef TH_GENERIC_NO_DOUBLE
IMPLEMENT_THStorage_COPY(Double)
#endif
#endif
