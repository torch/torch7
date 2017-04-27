#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.c"
#else

/**** access methods ****/
THStorage *THTensor_(storage)(const THTensor *self)
{
  return self->storage;
}

ptrdiff_t THTensor_(storageOffset)(const THTensor *self)
{
  return self->storageOffset;
}

int THTensor_(nDimension)(const THTensor *self)
{
  return self->nDimension;
}

long THTensor_(size)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_(nDimension)(self));
  return self->size[dim];
}

long THTensor_(stride)(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_(nDimension)(self));
  return self->stride[dim];
}

THLongStorage *THTensor_(newSizeOf)(THTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THTensor_(newStrideOf)(THTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

real *THTensor_(data)(const THTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

void THTensor_(setFlag)(THTensor *self, const char flag)
{
  self->flag |= flag;
}

void THTensor_(clearFlag)(THTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THTensor_(rawInit)(THTensor *self);


/* Empty init */
THTensor *THTensor_(new)(void)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  return self;
}

/* Pointer-copy init */
THTensor *THTensor_(newWithTensor)(THTensor *tensor)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(setStorageNd)(self,
                          tensor->storage,
                          tensor->storageOffset,
                          tensor->nDimension,
                          tensor->size,
                          tensor->stride);
  return self;
}

/* Storage init */
THTensor *THTensor_(newWithStorage)(THStorage *storage, ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THTensor_(rawInit)(self);
#ifdef DEBUG
  THAssert((size ? size->size : (stride ? stride->size : 0)) <= INT_MAX);
#endif
  THTensor_(setStorageNd)(self,
                          storage,
                          storageOffset,
                          (size ? size->size : (stride ? stride->size : 0)),
                          (size ? size->data : NULL),
                          (stride ? stride->data : NULL));

  return self;
}
THTensor *THTensor_(newWithStorage1d)(THStorage *storage, ptrdiff_t storageOffset,
                               long size0, long stride0)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THTensor *THTensor_(newWithStorage2d)(THStorage *storage, ptrdiff_t storageOffset,
                               long size0, long stride0,
                               long size1, long stride1)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THTensor *THTensor_(newWithStorage3d)(THStorage *storage, ptrdiff_t storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THTensor *THTensor_(newWithStorage4d)(THStorage *storage, ptrdiff_t storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2,
                               long size3, long stride3)
{
  long size[4] = {size0, size1, size2, size3};
  long stride[4] = {stride0, stride1, stride2, stride3};

  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(setStorageNd)(self, storage, storageOffset, 4, size, stride);

  return self;
}

THTensor *THTensor_(newWithSize)(THLongStorage *size, THLongStorage *stride)
{
  return THTensor_(newWithStorage)(NULL, 0, size, stride);
}

THTensor *THTensor_(newWithSize1d)(long size0)
{
  return THTensor_(newWithSize4d)(size0, -1, -1, -1);
}

THTensor *THTensor_(newWithSize2d)(long size0, long size1)
{
  return THTensor_(newWithSize4d)(size0, size1, -1, -1);
}

THTensor *THTensor_(newWithSize3d)(long size0, long size1, long size2)
{
  return THTensor_(newWithSize4d)(size0, size1, size2, -1);
}

THTensor *THTensor_(newWithSize4d)(long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(resizeNd)(self, 4, size, NULL);

  return self;
}

THTensor *THTensor_(newClone)(THTensor *self)
{
  THTensor *tensor = THTensor_(new)();
  THTensor_(resizeAs)(tensor, self);
  THTensor_(copy)(tensor, self);
  return tensor;
}

THTensor *THTensor_(newContiguous)(THTensor *self)
{
  if(!THTensor_(isContiguous)(self))
    return THTensor_(newClone)(self);
  else
  {
    THTensor_(retain)(self);
    return self;
  }
}

THTensor *THTensor_(newSelect)(THTensor *tensor, int dimension_, long sliceIndex_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(select)(self, NULL, dimension_, sliceIndex_);
  return self;
}

THTensor *THTensor_(newNarrow)(THTensor *tensor, int dimension_, long firstIndex_, long size_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(narrow)(self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THTensor *THTensor_(newTranspose)(THTensor *tensor, int dimension1_, int dimension2_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(transpose)(self, NULL, dimension1_, dimension2_);
  return self;
}

THTensor *THTensor_(newUnfold)(THTensor *tensor, int dimension_, long size_, long step_)
{
  THTensor *self = THTensor_(newWithTensor)(tensor);
  THTensor_(unfold)(self, NULL, dimension_, size_, step_);
  return self;
}

THTensor *THTensor_(newView)(THTensor *tensor, THLongStorage *size)
{
  THArgCheck(THTensor_(isContiguous)(tensor), 1, "input is not contiguous");
  ptrdiff_t numel = THTensor_(nElement)(tensor);
  THTensor *self = THTensor_(new)();
  THLongStorage *inferred_size = THLongStorage_newInferSize(size, numel);
  THTensor_(setStorage)(self, tensor->storage, tensor->storageOffset, inferred_size, NULL);
  THLongStorage_free(inferred_size);
  return self;
}

/* Resize */
void THTensor_(resize)(THTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

#ifdef DEBUG
  THAssert(size->size <= INT_MAX);
#endif
  THTensor_(resizeNd)(self, size->size, size->data, (stride ? stride->data : NULL));
}

void THTensor_(resizeAs)(THTensor *self, THTensor *src)
{
  if(!THTensor_(isSameSizeAs)(self, src))
    THTensor_(resizeNd)(self, src->nDimension, src->size, NULL);
}

void THTensor_(resize1d)(THTensor *tensor, long size0)
{
  THTensor_(resize4d)(tensor, size0, -1, -1, -1);
}

void THTensor_(resize2d)(THTensor *tensor, long size0, long size1)
{
  THTensor_(resize4d)(tensor, size0, size1, -1, -1);
}

void THTensor_(resize3d)(THTensor *tensor, long size0, long size1, long size2)
{
  THTensor_(resize4d)(tensor, size0, size1, size2, -1);
}

void THTensor_(resize4d)(THTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THTensor_(resizeNd)(self, 4, size, NULL);
}

void THTensor_(resize5d)(THTensor *self, long size0, long size1, long size2, long size3, long size4)
{
    long size[5] = {size0, size1, size2, size3, size4};

  THTensor_(resizeNd)(self, 5, size, NULL);
}

void THTensor_(expandNd)(THTensor *tensor, long *sizes, ptrdiff_t ndim) {
  THArgCheck(tensor->nDimension > 0, 0, "can't expand an empty tensor");
  long numUnsqueezed = ndim - tensor->nDimension;

  long *expandedSizes = THAlloc(sizeof(long)*ndim);
  long *expandedStrides = THAlloc(sizeof(long)*ndim);

  for (long i = numUnsqueezed; i < ndim; ++i) {
    expandedSizes[i] = THTensor_(size)(tensor, i);
    expandedStrides[i] = THTensor_(stride)(tensor, i);
  }

  for (long i = numUnsqueezed - 1; i > -1; --i) {
    expandedSizes[i] = 1;
    expandedStrides[i] = expandedSizes[i+1] * expandedStrides[i+1];
  }

  // create a new geometry for the tensor
  for (long i = 0; i < ndim; ++i) {
    long size = expandedSizes[i];
    long targetSize = sizes[i];
    if (size == 1) {
      if (targetSize != 1) {
        expandedSizes[i] = targetSize;
        expandedStrides[i] = 0;
      }
    } else if (size != targetSize) {
      THFree(expandedSizes);
      THFree(expandedStrides);
      THError("The expanded size of the tensor (%d) must match the existing size (%d) at \
              non-singleton dimension %ld.", targetSize, size, i);
    }
  }

  THTensor_(setStorageNd)(tensor, tensor->storage, tensor->storageOffset, ndim, expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THTensor_(expand)(THTensor *tensor, THLongStorage *sizes) {
  THArgCheck(sizes->size >= tensor->nDimension, 1, "the number of sizes provided \
      must be greater or equal to the number of dimensions in the tensor");
  THTensor_(expandNd)(tensor, sizes->data, sizes->size);
}

void THTensor_(expandAs)(THTensor *tensor, THTensor *src) {
  THArgCheck(src->nDimension >= tensor->nDimension, 1, "the number of dimensions of the provided \
      tensor must be greater or equal to the number of dimensions in the tensor to be expanded");
  THTensor_(expandNd)(tensor, src->size, src->nDimension);
}

void THTensor_(set)(THTensor *self, THTensor *src)
{
  if(self != src)
    THTensor_(setStorageNd)(self,
                            src->storage,
                            src->storageOffset,
                            src->nDimension,
                            src->size,
                            src->stride);
}

void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

#ifdef DEBUG
  THAssert((size_ ? size_->size : (stride_ ? stride_->size : 0)) <= INT_MAX);
#endif
  THTensor_(setStorageNd)(self,
                          storage_,
                          storageOffset_,
                          (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                          (size_ ? size_->data : NULL),
                          (stride_ ? stride_->data : NULL));
}

void THTensor_(setStorage1d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             long size0_, long stride0_)
{
  THTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          -1, -1,
                          -1, -1,
                          -1, -1);
}

void THTensor_(setStorage2d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_)
{
  THTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          -1, -1,
                          -1, -1);
}

void THTensor_(setStorage3d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_)
{
  THTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          size2_, stride2_,
                          -1, -1);
}

void THTensor_(setStorage4d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_,
                             long size3_, long stride3_)
{

  long size[4] = {size0_, size1_, size2_, size3_};
  long stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THTensor_(setStorageNd)(self, storage_, storageOffset_, 4, size, stride);
}


void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension, long firstIndex, long size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex <= src->size[dimension] - size), 4, "out of range");

  THTensor_(set)(self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THTensor_(select)(THTensor *self, THTensor *src, int dimension, long sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");

  THTensor_(set)(self, src);
  THTensor_(narrow)(self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THTensor_(transpose)(THTensor *self, THTensor *src, int dimension1, int dimension2)
{
  long z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THTensor_(set)(self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THTensor_(unfold)(THTensor *self, THTensor *src, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THTensor_(set)(self, src);

  newSize = THAlloc(sizeof(long)*(self->nDimension+1));
  newStride = THAlloc(sizeof(long)*(self->nDimension+1));

  newSize[self->nDimension] = size;
  newStride[self->nDimension] = self->stride[dimension];
  for(d = 0; d < self->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (self->size[d] - size) / step + 1;
      newStride[d] = step*self->stride[d];
    }
    else
    {
      newSize[d] = self->size[d];
      newStride[d] = self->stride[d];
    }
  }

  THFree(self->size);
  THFree(self->stride);

  self->size = newSize;
  self->stride = newStride;
  self->nDimension++;
}

/* we have to handle the case where the result is a number */
void THTensor_(squeeze)(THTensor *self, THTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THTensor_(set)(self, src);

  for(d = 0; d < src->nDimension; d++)
  {
    if(src->size[d] != 1)
    {
      if(d != ndim)
      {
        self->size[ndim] = src->size[d];
        self->stride[ndim] = src->stride[d];
      }
      ndim++;
    }
  }

  /* right now, we do not handle 0-dimension tensors */
  if(ndim == 0 && src->nDimension > 0)
  {
    self->size[0] = 1;
    self->stride[0] = 1;
    ndim = 1;
  }
  self->nDimension = ndim;
}

void THTensor_(squeeze1d)(THTensor *self, THTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "dimension out of range");

  THTensor_(set)(self, src);

  if(src->size[dimension] == 1 && src->nDimension > 1)
  {
    for(d = dimension; d < self->nDimension-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->nDimension--;
  }
}

void THTensor_(unsqueeze1d)(THTensor *self, THTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->nDimension), 2, "dimension out of range");
  THArgCheck(src->nDimension > 0, 2, "cannot unsqueeze empty tensor");

  THTensor_(set)(self, src);

  self->size = (long*)THRealloc(self->size, sizeof(long)*(self->nDimension+1));
  self->stride = (long*)THRealloc(self->stride, sizeof(long)*(self->nDimension+1));
  self->nDimension++;
  for (d = self->nDimension-1; d > dimension; d--) {
    self->size[d] = self->size[d-1];
    self->stride[d] = self->stride[d-1];
  }
  if (dimension+1 < self->nDimension) {
    self->stride[dimension] = self->size[dimension+1] * self->stride[dimension+1];
  } else {
    self->stride[dimension] = 1;
  }
  self->size[dimension] = 1;
}

int THTensor_(isTransposed)(const THTensor *self)
{
  if (THTensor_(isContiguous)(self)) {
    return 0;
  }
  long max_stride = 1;
  long size_max_stride = 1;
  long z = 1;
  int d;
  for (d = 0; d < self->nDimension; ++d) {
    if (self->stride[d] == 0 && self->size[d] != 1)
      return 0;
    if (self->stride[d] > max_stride) {
      max_stride = self->stride[d];
      size_max_stride = self->size[d];
    }
    z *= self->size[d];
  }
  if (z == max_stride * size_max_stride) {
    return 1;
  }
  return 0;
}

int THTensor_(isContiguous)(const THTensor *self)
{
  long z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}

int THTensor_(isSize)(const THTensor *self, const THLongStorage *dims)
{
  int d;
  if (self->nDimension != dims->size)
    return 0;

  for(d = 0; d < self->nDimension; ++d)
  {
    if(self->size[d] != dims->data[d])
      return 0;
  }
  return 1;
}

int THTensor_(isSameSizeAs)(const THTensor *self, const THTensor* src)
{
  int d;
  if (self->nDimension != src->nDimension)
    return 0;
  for(d = 0; d < self->nDimension; ++d)
  {
    if(self->size[d] != src->size[d])
      return 0;
  }
  return 1;
}

int THTensor_(isSetTo)(const THTensor *self, const THTensor* src)
{
  if (!self->storage)
    return 0;
  if (self->storage == src->storage &&
      self->storageOffset == src->storageOffset &&
      self->nDimension == src->nDimension)
  {
    int d;
    for (d = 0; d < self->nDimension; ++d)
    {
      if (self->size[d] != src->size[d] || self->stride[d] != src->stride[d])
        return 0;
    }
    return 1;
  }
  return 0;
}

ptrdiff_t THTensor_(nElement)(const THTensor *self)
{
  if(self->nDimension == 0)
    return 0;
  else
  {
    ptrdiff_t nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THTensor_(retain)(THTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    THAtomicIncrementRef(&self->refcount);
}

void THTensor_(free)(THTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(THAtomicDecrementRef(&self->refcount))
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THStorage_(free)(self->storage);
      THFree(self);
    }
  }
}

void THTensor_(freeCopyTo)(THTensor *self, THTensor *dst)
{
  if(self != dst)
    THTensor_(copy)(dst, self);

  THTensor_(free)(self);
}

/*******************************************************************************/

static void THTensor_(rawInit)(THTensor *self)
{
  self->refcount = 1;
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

void THTensor_(setStorageNd)(THTensor *self, THStorage *storage, ptrdiff_t storageOffset, int nDimension, long *size, long *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THStorage_(free)(self->storage);

    if(storage)
    {
      self->storage = storage;
      THStorage_(retain)(self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THTensor_(resizeNd)(self, nDimension, size, stride);
}

void THTensor_(resizeNd)(THTensor *self, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  ptrdiff_t totalSize;
  int hascorrectsize = 1;

  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
    {
      nDimension_++;
      if((self->nDimension > d) && (size[d] != self->size[d]))
        hascorrectsize = 0;

      if((self->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hascorrectsize = 0;
    }
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension != self->nDimension)
    hascorrectsize = 0;

  if(hascorrectsize)
    return;

  if(nDimension > 0)
  {
    if(nDimension != self->nDimension)
    {
      self->size = THRealloc(self->size, sizeof(long)*nDimension);
      self->stride = THRealloc(self->stride, sizeof(long)*nDimension);
      self->nDimension = nDimension;
    }

    totalSize = 1;
    for(d = self->nDimension-1; d >= 0; d--)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == self->nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if(totalSize+self->storageOffset > 0)
    {
      if(!self->storage)
        self->storage = THStorage_(new)();
      if(totalSize+self->storageOffset > self->storage->size)
        THStorage_(resize)(self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THTensor_(set1d)(THTensor *tensor, long x0, real value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

real THTensor_(get1d)(const THTensor *tensor, long x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

real THTensor_(get2d)(const THTensor *tensor, long x0, long x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

real THTensor_(get3d)(const THTensor *tensor, long x0, long x1, long x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

real THTensor_(get4d)(const THTensor *tensor, long x0, long x1, long x2, long x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return THStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
}

THDescBuff THTensor_(desc)(const THTensor *tensor) {
  const int L = TH_DESC_BUFF_LEN;
  THDescBuff buf;
  char *str = buf.str;
  int n = 0;
#define _stringify(x) #x
  n += snprintf(str, L-n, "torch." _stringify(x) "Tensor of size ");
#undef _stringify
  int i;
  for(i = 0; i < tensor->nDimension; i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%ld", tensor->size[i]);
    if(i < tensor->nDimension-1) {
      n += snprintf(str+n, L-n, "x");
    }
  }
  if(n >= L) {
    snprintf(str+L-4, 4, "...");
  }
  return buf;
}

THDescBuff THTensor_(sizeDesc)(const THTensor *tensor) {
  THLongStorage *size = THTensor_(newSizeOf)((THTensor*)tensor);
  THDescBuff buf = THLongStorage_sizeDesc(size);
  THLongStorage_free(size);
  return buf;
}

#endif
