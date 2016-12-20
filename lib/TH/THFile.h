#ifndef TH_FILE_INC
#define TH_FILE_INC

#include "THStorage.h"

typedef struct THFile__ THFile;

TH_API int THFile_isOpened(THFile *self);
TH_API int THFile_isQuiet(THFile *self);
TH_API int THFile_isReadable(THFile *self);
TH_API int THFile_isWritable(THFile *self);
TH_API int THFile_isBinary(THFile *self);
TH_API int THFile_isAutoSpacing(THFile *self);
TH_API int THFile_hasError(THFile *self);

TH_API void THFile_binary(THFile *self);
TH_API void THFile_ascii(THFile *self);
TH_API void THFile_autoSpacing(THFile *self);
TH_API void THFile_noAutoSpacing(THFile *self);
TH_API void THFile_quiet(THFile *self);
TH_API void THFile_pedantic(THFile *self);
TH_API void THFile_clearError(THFile *self);

/* scalar */
TH_API unsigned char THFile_readByteScalar(THFile *self);
TH_API char THFile_readCharScalar(THFile *self);
TH_API short THFile_readShortScalar(THFile *self);
TH_API int THFile_readIntScalar(THFile *self);
TH_API long THFile_readLongScalar(THFile *self);
TH_API float THFile_readFloatScalar(THFile *self);
TH_API double THFile_readDoubleScalar(THFile *self);

TH_API void THFile_writeByteScalar(THFile *self, unsigned char scalar);
TH_API void THFile_writeCharScalar(THFile *self, char scalar);
TH_API void THFile_writeShortScalar(THFile *self, short scalar);
TH_API void THFile_writeIntScalar(THFile *self, int scalar);
TH_API void THFile_writeLongScalar(THFile *self, long scalar);
TH_API void THFile_writeFloatScalar(THFile *self, float scalar);
TH_API void THFile_writeDoubleScalar(THFile *self, double scalar);

/* storage */
TH_API size_t THFile_readByte(THFile *self, THByteStorage *storage);
TH_API size_t THFile_readChar(THFile *self, THCharStorage *storage);
TH_API size_t THFile_readShort(THFile *self, THShortStorage *storage);
TH_API size_t THFile_readInt(THFile *self, THIntStorage *storage);
TH_API size_t THFile_readLong(THFile *self, THLongStorage *storage);
TH_API size_t THFile_readFloat(THFile *self, THFloatStorage *storage);
TH_API size_t THFile_readDouble(THFile *self, THDoubleStorage *storage);

TH_API size_t THFile_writeByte(THFile *self, THByteStorage *storage);
TH_API size_t THFile_writeChar(THFile *self, THCharStorage *storage);
TH_API size_t THFile_writeShort(THFile *self, THShortStorage *storage);
TH_API size_t THFile_writeInt(THFile *self, THIntStorage *storage);
TH_API size_t THFile_writeLong(THFile *self, THLongStorage *storage);
TH_API size_t THFile_writeFloat(THFile *self, THFloatStorage *storage);
TH_API size_t THFile_writeDouble(THFile *self, THDoubleStorage *storage);

/* raw */
TH_API size_t THFile_readByteRaw(THFile *self, unsigned char *data, size_t n);
TH_API size_t THFile_readCharRaw(THFile *self, char *data, size_t n);
TH_API size_t THFile_readShortRaw(THFile *self, short *data, size_t n);
TH_API size_t THFile_readIntRaw(THFile *self, int *data, size_t n);
TH_API size_t THFile_readLongRaw(THFile *self, long *data, size_t n);
TH_API size_t THFile_readFloatRaw(THFile *self, float *data, size_t n);
TH_API size_t THFile_readDoubleRaw(THFile *self, double *data, size_t n);
TH_API size_t THFile_readStringRaw(THFile *self, const char *format, char **str_); /* you must deallocate str_ */

TH_API size_t THFile_writeByteRaw(THFile *self, unsigned char *data, size_t n);
TH_API size_t THFile_writeCharRaw(THFile *self, char *data, size_t n);
TH_API size_t THFile_writeShortRaw(THFile *self, short *data, size_t n);
TH_API size_t THFile_writeIntRaw(THFile *self, int *data, size_t n);
TH_API size_t THFile_writeLongRaw(THFile *self, long *data, size_t n);
TH_API size_t THFile_writeFloatRaw(THFile *self, float *data, size_t n);
TH_API size_t THFile_writeDoubleRaw(THFile *self, double *data, size_t n);
TH_API size_t THFile_writeStringRaw(THFile *self, const char *str, size_t size);

#if TH_GENERIC_USE_HALF
TH_API half THFile_readHalfScalar(THFile *self);
TH_API void THFile_writeHalfScalar(THFile *self, half scalar);
TH_API size_t THFile_readHalf(THFile *self, THHalfStorage *storage);
TH_API size_t THFile_writeHalf(THFile *self, THHalfStorage *storage);
TH_API size_t THFile_readHalfRaw(THFile *self, half* data, size_t size);
TH_API size_t THFile_writeHalfRaw(THFile *self, half* data, size_t size);
#endif

TH_API void THFile_synchronize(THFile *self);
TH_API void THFile_seek(THFile *self, size_t position);
TH_API void THFile_seekEnd(THFile *self);
TH_API size_t THFile_position(THFile *self);
TH_API void THFile_close(THFile *self);
TH_API void THFile_free(THFile *self);

#endif
