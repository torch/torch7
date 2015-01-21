#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.c"
#else

#define TH_OMP_OVERHEAD_THRESHOLD 100000

void THTensor_(fill)(THTensor *r_, real value)
{
  TH_TENSOR_APPLY(real, r_, 
                  THVector_(fill)(r__data, value, r__size); break;);
}

void THTensor_(zero)(THTensor *r_)
{
  TH_TENSOR_APPLY(real, r_, 
                  THVector_(fill)(r__data, 0, r__size); break;);
}

void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value)
{
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
                   if (*mask_data > 1) THError("Mask tensor can take 0 and 1 values only");
                   else if (*mask_data == 1) *tensor_data = value;);
}

void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src )
{
  THTensor *srct = THTensor_(newContiguous)(src);
  real *src_data = THTensor_(data)(srct);
  long cntr = 0;
  long nelem = THTensor_(nElement)(srct);
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
		   if (*mask_data > 1)
		   {
		     THError("Mask tensor can take 0 and 1 values only");
		   }
		   else if (*mask_data == 1)
		   {
		     *tensor_data = *src_data;
		     src_data++;
		     cntr++;
		     if (cntr > nelem)
		       THError("Number of elements of src != mask");
		   });
  if (cntr != nelem)
    THError("Number of elements of src != mask");
  THTensor_(free)(srct);
}

void THTensor_(maskedSelect)(THTensor *tensor, THTensor *src, THByteTensor *mask)
{
  long numel = THByteTensor_sumall(mask);
  real *tensor_data;

  THTensor_(resize1d)(tensor,numel);
  tensor_data = THTensor_(data)(tensor);
  TH_TENSOR_APPLY2(real, src, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     *tensor_data = *src_data;
                     tensor_data++;
                   });
}

void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
  long i, numel;
  THLongStorage *newSize;
  THTensor *tSlice, *sSlice;
  long *index_data;

  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0,2,"Source tensor is empty");

  numel = THLongTensor_nElement(index);

  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize,src->size);
  newSize->data[dim] = numel;
  THTensor_(resize)(tensor,newSize,NULL);
  THLongStorage_free(newSize);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);
  for (i=0; i<numel; i++)
  {
    if (src->nDimension > 1)
    {
      tSlice = THTensor_(new)();
      sSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor, dim, i);
      THTensor_(select)(sSlice, src, dim, index_data[i]-1);
      THTensor_(copy)(tSlice, sSlice);
      THTensor_(free)(tSlice);
      THTensor_(free)(sSlice);
    }
    else
    {
      THTensor_(set1d)(tensor,i,THTensor_(get1d)(src,index_data[i]-1));
    }
  }
  THLongTensor_free(index);
}

void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  long i, numel;
  THTensor *tSlice, *sSlice;
  long *index_data;
  
  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(numel == src->size[dim],4,"Number of indices should be equal to source:size(dim)");
  
  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);
  
  if (tensor->nDimension > 1 )
  {
    tSlice = THTensor_(new)();
    sSlice = THTensor_(new)();
    
    for (i=0; i<numel; i++)
    {
      THTensor_(select)(tSlice, tensor, dim, index_data[i]-1);
      THTensor_(select)(sSlice, src, dim, i);
      THTensor_(copy)(tSlice, sSlice);
    }
    
    THTensor_(free)(tSlice);
    THTensor_(free)(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THTensor_(set1d)(tensor,index_data[i]-1,THTensor_(get1d)(src,i));
    }
  }
  THLongTensor_free(index);
}

void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val)
{
  long i, numel;
  THTensor *tSlice;
  long *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->nDimension,4,"Indexing dim is out of bounds");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);
  
  for (i=0; i<numel; i++)
  {
    if (tensor->nDimension > 1 )
    {
      tSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor,dim,index_data[i]-1);
      THTensor_(fill)(tSlice, val);
      THTensor_(free)(tSlice);
    }
    else
    {
      THTensor_(set1d)(tensor,index_data[i]-1,val);
    }
  }
  THLongTensor_free(index);
}

accreal THTensor_(dot)(THTensor *tensor, THTensor *src)
{
  accreal sum = 0;
  /* we use a trick here. careful with that. */
  TH_TENSOR_APPLY2(real, tensor, real, src,
                   long sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   sum += THBlas_(dot)(sz, src_data, src_stride, tensor_data, tensor_stride);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride; 
                   break;);
  return sum; 
}

real THTensor_(minall)(THTensor *tensor)
{
  real theMin;
  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMin = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor, if(*tensor_data < theMin) theMin = *tensor_data;);
  return theMin; 
}

real THTensor_(maxall)(THTensor *tensor)
{
  real theMax;
  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMax = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor, if(*tensor_data > theMax) theMax = *tensor_data;);
  return theMax; 
}

accreal THTensor_(sumall)(THTensor *tensor)
{
  accreal sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += *tensor_data;);
  return sum;
}

accreal THTensor_(prodall)(THTensor *tensor)
{
  accreal prod = 1;
  TH_TENSOR_APPLY(real, tensor, prod *= *tensor_data;);
  return prod;
}

void THTensor_(add)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] + value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}

void THTensor_(mul)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] * value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * value;);
  }
}

void THTensor_(div)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] / value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data / value;);
  }
}

void THTensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      real t_val;
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = (tp[i] < min_value) ? min_value : (tp[i] > max_value ? max_value : tp[i]);
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (*t_data < min_value) ? min_value : (*t_data > max_value ? max_value : *t_data););
  }
}

void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
    if(r_ == t) {
      THBlas_(axpy)(THTensor_(nElement)(t), value, THTensor_(data)(src), 1, THTensor_(data)(r_), 1);
    } else {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i< sz; i++)
          rp[i] = tp[i] + value * sp[i];
    }
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data + value * *src_data;);
  }
}

void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
        rp[i] = tp[i] * sp[i];
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data * *src_data;);
  }
}

void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
        rp[i] = tp[i] / sp[i];
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data / *src_data;);
  }
}

void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2)
{
  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  TH_TENSOR_APPLY3(real, r_, real, src1, real, src2, *r__data += value * *src1_data * *src2_data;);
}


void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2)
{
  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  TH_TENSOR_APPLY3(real, r_, real, src1, real, src2, *r__data += value * *src1_data / *src2_data;);
}

void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat, THTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");
 
  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(t->nDimension != 1)
    THError("size mismatch");
    
  if(t->size[0] != mat->size[0])
    THError("size mismatch");

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THBlas_(gemv)('n', mat->size[0], mat->size[1],
                  alpha, THTensor_(data)(mat), mat->stride[1],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(mat), mat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(r_), r_->stride[0]);
  }
  else
  {
    THTensor *cmat = THTensor_(newContiguous)(mat);

    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(cmat), cmat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(r_), r_->stride[0]);

    THTensor_(free)(cmat);
  }
}

void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, real gain)
{
  long N1 = m1->size[0];
  long N2 = m2->size[0];
  long dim;
  real *m1_p;
  real *m2_p;
  real *r_p;
  long i;

  THTensor_(resize2d)(r_, N1, N2);
  
  m1 = THTensor_(newContiguous)(m1);
  m2 = THTensor_(newContiguous)(m2);

  THTensor_(resize2d)(m1, N1, THTensor_(nElement)(m1) / N1);
  THTensor_(resize2d)(m2, N2, THTensor_(nElement)(m2) / N2);
    
  dim = m1->size[1];
  THArgCheck(m1->size[1] == m2->size[1], 3, "m1 and m2 must have the same inner vector dim");

  m1_p = THTensor_(data)(m1);
  m2_p = THTensor_(data)(m2);
  r_p = THTensor_(data)(r_);

#pragma omp parallel for private(i)
  for (i=0; i<N1; i++) {
    long j,k;
    for (j=0; j<N2; j++) {
      real sum = 0;
      for (k=0; k<dim; k++) {
        real term = m1_p[ i*dim + k ] - m2_p[ j*dim + k ];
        sum += term*term;
      }
      r_p[ i*N2 + j ] = gain * sum;
    }
  }

  THTensor_(free)(m1);
  THTensor_(free)(m2);
}

void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *m1, THTensor *m2)
{ 
  char transpose_r, transpose_m1, transpose_m2;
  THTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) ) 
    THError("matrix and matrix expected"); 
 
  if(t->nDimension != 2)
    THError("size mismatch"); 

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) ) 
    THError("size mismatch"); 

  if(t != r_)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

/*  printf("%ldx%ld = %ldx%ld X %ldx%ld\n", r_->size[0], r_->size[1], m1->size[0], m1->size[1], m2->size[0], m2->size[1]); */

  /* r_ */
  if(r_->stride[0] == 1)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1)
  {
    THTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';
    
    r__ = THTensor_(newWithSize2d)(r_->size[1], r_->size[0]);
    THTensor_(copy)(r__, r_);
    THTensor_(transpose)(r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THTensor_(newContiguous)(m1);
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THTensor_(newContiguous)(m2);
  }

  /* do the operation */
  THBlas_(gemm)(transpose_m1,
                transpose_m2,
                r__->size[(transpose_r == 'n' ? 0 : 1)],
                r__->size[(transpose_r == 'n' ? 1 : 0)],
                m1_->size[(transpose_r == 'n' ? 1 : 0)],
                alpha,
                THTensor_(data)(m1_),
                (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                THTensor_(data)(m2_),
                (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                beta,
                THTensor_(data)(r__),
                r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(m1_ != m1)
    THTensor_(free)(m1_);

  if(m2_ != m2)
    THTensor_(free)(m2_);

  if(r__ != r_)
    THTensor_(freeCopyTo)(r__, r_);
} 

void THTensor_(addr)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(t->nDimension != 2)
    THError("size mismatch");
    
  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  if(beta != 1)
    THTensor_(mul)(r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THBlas_(ger)(vec1->size[0], vec2->size[0],
                 alpha, THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(r_), r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
    THBlas_(ger)(vec2->size[0], vec1->size[0],
                 alpha, THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(r_), r_->stride[0]);
  }
  else
  {
    THTensor *cr = THTensor_(newClone)(r_);

    THBlas_(ger)(vec2->size[0], vec1->size[0],
                 alpha, THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(cr), cr->stride[0]);

    THTensor_(freeCopyTo)(cr, r_);
  }
}

void THTensor_(bmm)(THTensor *result, THTensor *batch1, THTensor *batch2)
{
  long batch;

  THArgCheck(THTensor_(nDimension)(batch1) == 3, 1, "expected 3D tensor");
  THArgCheck(THTensor_(nDimension)(batch2) == 3, 2, "expected 3D tensor");
  THArgCheck(THTensor_(size)(batch1, 0) == THTensor_(size)(batch2, 0), 2,
             "equal number of batches expected");
  THArgCheck(THTensor_(size)(batch1, 2) == THTensor_(size)(batch2, 1), 2,
             "wrong matrix size");

  THLongStorage* result_size = THLongStorage_newWithSize3(THTensor_(size)(batch1, 0),
                                                          THTensor_(size)(batch1, 1),
                                                          THTensor_(size)(batch2, 2));
  THTensor_(resize)(result, result_size, NULL);
  THLongStorage_free(result_size);

  for (batch = 0; batch < THTensor_(size)(batch1, 0); ++batch) {
    THTensor *matrix1 = THTensor_(newWithTensor)(batch1);
    THTensor *matrix2 = THTensor_(newWithTensor)(batch2);
    THTensor *result_matrix = THTensor_(newWithTensor(result));

    THTensor_(select)(matrix1, NULL, 0, batch);
    THTensor_(select)(matrix2, NULL, 0, batch);
    THTensor_(select)(result_matrix, NULL, 0, batch);

    THTensor_(addmm)(result_matrix, 0, result_matrix, 1, matrix1, matrix2);

    THTensor_(free)(matrix1);
    THTensor_(free)(matrix2);
    THTensor_(free)(result_matrix);
  }
}

long THTensor_(numel)(THTensor *t)
{
  return THTensor_(nElement)(t);
}

void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension)
{
  THLongStorage *dim;
  long i;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, long, indices_, dimension,
                       long theIndex = 0;
                       real theMax = t_data[0];
                       for(i = 1; i < t_size; i++)
                       {
                         if(t_data[i*t_stride] > theMax)
                         {
                           theIndex = i;
                           theMax = t_data[i*t_stride];
                         }
                       }
                       *indices__data = theIndex;
                       *values__data = theMax;);  

}

void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension)
{
  THLongStorage *dim;
  long i;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, long, indices_, dimension,
                       long theIndex = 0;
                       real theMin = t_data[0];
                       for(i = 1; i < t_size; i++)
                       {
                         if(t_data[i*t_stride] < theMin)
                         {
                           theIndex = i;
                           theMin = t_data[i*t_stride];
                         }
                       }
                       *indices__data = theIndex;
                       *values__data = theMin;);  

}


void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");
  
  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                         sum += t_data[i*t_stride];
                       *r__data = (real)sum;);
}

void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);
  
  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal prod = 1;
                       long i;
                       for(i = 0; i < t_size; i++)
                         prod *= t_data[i*t_stride];
                       *r__data = (real)prod;);

}

void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");
  
  THTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal cumsum = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumsum += t_data[i*t_stride];
                         r__data[i*r__stride] = (real)cumsum;
                       });
}

void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");
  
  THTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal cumprod = 1;
                       long i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumprod *= t_data[i*t_stride];
                         r__data[i*r__stride] = (real)cumprod;
                       });
}


void THTensor_(sign)(THTensor *r_, THTensor *t)
{
  THTensor_(resizeAs)(r_, t);

#if defined (TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY2(real, r_, real, t, 
		   if (*t_data > 0) *r__data = 1;
		   else *r__data = 0;);
#else
  TH_TENSOR_APPLY2(real, r_, real, t, 
		   if (*t_data > 0) *r__data = 1;
		   else if (*t_data < 0) *r__data = -1;
		   else *r__data = 0;);
#endif
}


accreal THTensor_(trace)(THTensor *t)
{
  real *t_data = THTensor_(data)(t);
  accreal sum = 0;
  long i = 0;
  long t_stride_0, t_stride_1, t_diag_size;

  THArgCheck(THTensor_(nDimension)(t) == 2, 1, "not a matrix");

  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  t_diag_size = THMin(THTensor_(size)(t, 0), THTensor_(size)(t, 1));
  while(i < t_diag_size)
  {
    sum += t_data[i*(t_stride_0+t_stride_1)];
    i++;
  }

  return sum;
}

void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension)
{
  int i;

  if(THTensor_(nDimension)(a) != THTensor_(nDimension)(b))
    THError("inconsitent tensor sizes");
  
  for(i = 0; i < THTensor_(nDimension)(a); i++)
  {
    if(THTensor_(size)(a, i) != THTensor_(size)(b, i))
      THError("inconsistent tensor sizes");
  }
  
  if(dimension < 0)
  {
    for(i = 0; i < THTensor_(nDimension)(a); i++)
    {
      if(THTensor_(size)(a, i) == 3)
      {
        dimension = i;
        break;
      }
    }
    if(dimension < 0)
      THError("no dimension of size 3");
  }

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(a), 3, "dimension out of range");
  THArgCheck(THTensor_(size)(a, dimension) == 3, 3, "dimension size is not 3");

  THTensor_(resizeAs)(r_, a);

  TH_TENSOR_DIM_APPLY3(real, a, real, b, real, r_, dimension,
                       r__data[0*r__stride] = a_data[1*a_stride]*b_data[2*b_stride] - a_data[2*a_stride]*b_data[1*b_stride];
                       r__data[1*r__stride] = a_data[2*a_stride]*b_data[0*b_stride] - a_data[0*a_stride]*b_data[2*b_stride];
                       r__data[2*r__stride] = a_data[0*a_stride]*b_data[1*b_stride] - a_data[1*a_stride]*b_data[0*b_stride];);
}

void THTensor_(zeros)(THTensor *r_, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(zero)(r_);
}

void THTensor_(ones)(THTensor *r_, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(fill)(r_, 1);
}

void THTensor_(diag)(THTensor *r_, THTensor *t, int k)
{
  THArgCheck(THTensor_(nDimension)(t) == 1 || THTensor_(nDimension)(t) == 2, 1, "matrix or a vector expected");

  if(THTensor_(nDimension)(t) == 1)
  {
    real *t_data = THTensor_(data)(t);
    long t_stride_0 = THTensor_(stride)(t, 0);
    long t_size = THTensor_(size)(t, 0);
    long sz = t_size + (k >= 0 ? k : -k);
    real *r__data;
    long r__stride_0;
    long r__stride_1;
    long i;

    THTensor_(resize2d)(r_, sz, sz);    
    THTensor_(zero)(r_);
    r__data = THTensor_(data)(r_);
    r__stride_0 = THTensor_(stride)(r_, 0);
    r__stride_1 = THTensor_(stride)(r_, 1);
    r__data += (k >= 0 ? k*r__stride_1 : -k*r__stride_0);

    for(i = 0; i < t_size; i++)
      r__data[i*(r__stride_0+r__stride_1)] = t_data[i*t_stride_0];
  }
  else
  {
    real *t_data = THTensor_(data)(t);
    long t_stride_0 = THTensor_(stride)(t, 0);
    long t_stride_1 = THTensor_(stride)(t, 1);
    long sz;
    real *r__data;
    long r__stride_0;
    long i;

    if(k >= 0)
      sz = THMin(THTensor_(size)(t, 0), THTensor_(size)(t, 1)-k);
    else
      sz = THMin(THTensor_(size)(t, 0)+k, THTensor_(size)(t, 1));
    THTensor_(resize1d)(r_, sz);
    r__data = THTensor_(data)(r_);
    r__stride_0 = THTensor_(stride)(r_, 0);

    t_data += (k >= 0 ? k*t_stride_1 : -k*t_stride_0);
    for(i = 0; i < sz; i++)
      r__data[i*r__stride_0] = t_data[i*(t_stride_0+t_stride_1)];
  }
}

void THTensor_(eye)(THTensor *r_, long n, long m)
{
  real *r__data;
  long i, sz;

  THArgCheck(n > 0, 1, "invalid argument");

  if(m <= 0)
    m = n;

  THTensor_(resize2d)(r_, n, m);
  THTensor_(zero)(r_);

  i = 0;
  r__data = THTensor_(data)(r_);
  sz = THMin(THTensor_(size)(r_, 0), THTensor_(size)(r_, 1));
  for(i = 0; i < sz; i++)
    r__data[i*(r_->stride[0]+r_->stride[1])] = 1;
}


void THTensor_(range)(THTensor *r_, real xmin, real xmax, real step)
{
  long size;
  real i = 0;

  THArgCheck(step > 0 || step < 0, 3, "step must be a non-null number");
  THArgCheck(((step > 0) && (xmax >= xmin)) || ((step < 0) && (xmax <= xmin))
              , 2, "upper bound and larger bound incoherent with step sign");

  size = (long)((xmax-xmin)/step+1);
  
  THTensor_(resize1d)(r_, size);

  TH_TENSOR_APPLY(real, r_, *r__data = xmin + (i++)*step;);
}

void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, long n)
{
  real *r__data;
  long r__stride_0;
  long i;

  THArgCheck(n > 0, 1, "must be strictly positive");

  THTensor_(resize1d)(r_, n);
  r__data = THTensor_(data)(r_);
  r__stride_0 = THTensor_(stride)(r_,0);

  for(i = 0; i < n; i++)
    r__data[i*r__stride_0] = (real)(i);

  for(i = 0; i < n-1; i++)
  {    
    long z = THRandom_random(_generator) % (n-i);
    real sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}

void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(copy)(r_, t);
}

/* I cut and pasted (slightly adapted) the quicksort code from
   http://www.alienryderflex.com/quicksort/
   This public-domain C implementation by Darel Rex Finley.
   Thanks man :)

    Updated Oct 16 2013: change choice of pivot to avoid worst-case being a pre-sorted input - Daniel and Julien
    Updated Oct 24 2013: change pivot comparison to strict inequality to avoid worst-case on constant input, see Sedgewick, Algorithms in C, Addison Wesley, 1990, p. 120 - Julien
*/
#define  MAX_LEVELS  300
static void THTensor_(quicksortascend)(real *arr, long *idx, long elements, long stride)
{
  long beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R, P, swap, pid;
  real rswap, piv;
  
  beg[0]=0; end[0]=elements;
  while (i>=0) {
    L=beg[i]; R=end[i]-1;
    if (L<R) {
      P=(L+R)>>1; /* Choose pivot as middle element of the current block */
      piv=arr[P*stride];
      pid=idx[P*stride];
      rswap=arr[L*stride];
      swap=idx[L*stride];
      arr[L*stride]=piv;
      idx[L*stride]=pid;
      arr[P*stride]=rswap;
      idx[P*stride]=swap;
      while (L<R) {
        while (arr[R*stride]>piv && L<R)
            R--;
        if (L<R) {
            idx[L*stride]=idx[R*stride];
            arr[L*stride]=arr[R*stride];
            L++;
        }
        while (arr[L*stride]<piv && L<R)
            L++;
        if (L<R) {
            idx[R*stride]=idx[L*stride];
            arr[R*stride]=arr[L*stride];
            R--;
        }
      }
      idx[L*stride]=pid;
      arr[L*stride]=piv;
      beg[i+1]=L+1;
      end[i+1]=end[i];
      end[i++]=L;
      if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
        swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
        swap=end[i]; end[i]=end[i-1]; end[i-1]=swap;
      }
    }
    else {
      i--;
    }
  }
}

static void THTensor_(quicksortdescend)(real *arr, long *idx, long elements, long stride)
{
  long beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R, P, swap, pid;
  real rswap, piv;
  
  beg[0]=0; end[0]=elements;
  while (i>=0) {
    L=beg[i]; R=end[i]-1;
    if (L<R) {
      P=(L+R)>>1; /* Choose pivot as middle element of the current block */
      piv=arr[P*stride];
      pid=idx[P*stride];
      rswap=arr[L*stride];
      swap=idx[L*stride];
      arr[L*stride]=piv;
      idx[L*stride]=pid;
      arr[P*stride]=rswap;
      idx[P*stride]=swap;
      while (L<R) {
        while (arr[R*stride]<piv && L<R)
            R--;
        if (L<R) {
            idx[L*stride]=idx[R*stride];
            arr[L*stride]=arr[R*stride];
            L++;
        }
        while (arr[L*stride]>piv && L<R)
            L++;
        if (L<R) {
            idx[R*stride]=idx[L*stride];
            arr[R*stride]=arr[L*stride];
            R--;
        }
      }
      idx[L*stride]=pid;
      arr[L*stride]=piv;
      beg[i+1]=L+1;
      end[i+1]=end[i];
      end[i++]=L;
      if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
        swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
        swap=end[i]; end[i]=end[i-1]; end[i-1]=swap;
      }
    }
    else {
      i--;
    }
  }
}

void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "invalid dimension");

  THTensor_(resizeAs)(rt_, t);
  THTensor_(copy)(rt_, t);

  {
    THLongStorage *size = THTensor_(newSizeOf)(t);
    THLongTensor_resize(ri_, size, NULL);
    THLongStorage_free(size);
  }

  if(descendingOrder)
  {
    TH_TENSOR_DIM_APPLY2(real, rt_, long, ri_, dimension, 
                         long i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THTensor_(quicksortdescend)(rt__data, ri__data, rt__size, rt__stride);)
      }
  else
  {
    TH_TENSOR_DIM_APPLY2(real, rt_, long, ri_, dimension,
                         long i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THTensor_(quicksortascend)(rt__data, ri__data, rt__size, rt__stride);)
      }
}

void THTensor_(tril)(THTensor *r_, THTensor *t, long k)
{
  long t_size_0, t_size_1;
  long t_stride_0, t_stride_1;
  long r__stride_0, r__stride_1;
  real *t_data, *r__data;
  long r, c;

  THArgCheck(THTensor_(nDimension)(t) == 2, 1, "not a matrix");

  THTensor_(resizeAs)(r_, t);

  t_size_0 = THTensor_(size)(t, 0);
  t_size_1 = THTensor_(size)(t, 1);
  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  r__stride_0 = THTensor_(stride)(r_, 0);
  r__stride_1 = THTensor_(stride)(r_, 1);
  r__data = THTensor_(data)(r_);
  t_data = THTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    long sz = THMin(r+k+1, t_size_1);
    for(c = THMax(0, r+k); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
  }
}

void THTensor_(triu)(THTensor *r_, THTensor *t, long k)
{
  long t_size_0, t_size_1;
  long t_stride_0, t_stride_1;
  long r__stride_0, r__stride_1;
  real *t_data, *r__data;
  long r, c;

  THArgCheck(THTensor_(nDimension)(t) == 2, 1, "not a matrix");

  THTensor_(resizeAs)(r_, t);

  t_size_0 = THTensor_(size)(t, 0);
  t_size_1 = THTensor_(size)(t, 1);
  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  r__stride_0 = THTensor_(stride)(r_, 0);
  r__stride_1 = THTensor_(stride)(r_, 1);
  r__data = THTensor_(data)(r_);
  t_data = THTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    long sz = THMin(r+k, t_size_1);
    for(c = THMax(0, r+k); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
  }
}

void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension)
{
  THLongStorage *size;
  int i;
  int ndim = THMax(ta->nDimension, tb->nDimension);
  ndim = THMax(ndim, dimension+1);

  THArgCheck(dimension >= 0, 4, "invalid dimension");

  size = THLongStorage_newWithSize(ndim);
  for(i = 0; i < ndim; i++)
  {
    int tadi = (i < ta->nDimension ? ta->size[i] : 1);
    int tbdi = (i < tb->nDimension ? tb->size[i] : 1);

    if(i == dimension)
      size->data[i] = tadi+tbdi;
    else
    {
      if(tadi != tbdi)
      {
        THLongStorage_free(size);
        THError("inconsistent tensor sizes");
      }
      size->data[i] = tadi;
    }
  }

  THTensor_(resize)(r_, size, NULL);
  THLongStorage_free(size);

  {
    THTensor *nta = THTensor_(newWithTensor)(r_);
    THTensor_(narrow)(nta, NULL, dimension, 0, (dimension < ta->nDimension ? ta->size[dimension] : 1));
    THTensor_(copy)(nta, ta);
    THTensor_(free)(nta);
  }

  {
    THTensor *ntb = THTensor_(newWithTensor)(r_);
    THTensor_(narrow)(ntb, NULL, dimension, (dimension < ta->nDimension ? ta->size[dimension] : 1), (dimension < tb->nDimension ? tb->size[dimension] : 1));
    THTensor_(copy)(ntb, tb);
    THTensor_(free)(ntb);
  }
}

#define TENSOR_IMPLEMENT_LOGICAL(NAME,OP)				\
  void THTensor_(NAME##Value)(THByteTensor *r_, THTensor* t, real value)	\
  {									\
    THByteTensor_rawResize(r_, t->nDimension, t->size, NULL);		\
    THByteTensor_zero(r_);						\
    TH_TENSOR_APPLY2(unsigned char, r_, real, t,			\
		     if (*t_data OP value) *r__data = 1;);		\
  }									\
  void THTensor_(NAME##ValueT)(THTensor* r_, THTensor* t, real value)	\
  {									\
    THTensor_(rawResize)(r_, t->nDimension, t->size, NULL);		\
    THTensor_(zero)(r_);						\
    TH_TENSOR_APPLY2(real, r_, real, t,					\
		     if (*t_data OP value) *r__data = 1;);		\
  }									\
  void THTensor_(NAME##Tensor)(THByteTensor *r_, THTensor *ta, THTensor *tb) \
  {									\
    THByteTensor_rawResize(r_, ta->nDimension, ta->size, NULL);		\
    THByteTensor_zero(r_);						\
    TH_TENSOR_APPLY3(unsigned char, r_, real, ta, real, tb,		\
		     if(*ta_data OP *tb_data) *r__data = 1;);		\
  }									\
  void THTensor_(NAME##TensorT)(THTensor *r_, THTensor *ta, THTensor *tb) \
  {									\
    THTensor_(rawResize)(r_, ta->nDimension, ta->size, NULL);		\
    THTensor_(zero)(r_);						\
    TH_TENSOR_APPLY3(real, r_, real, ta, real, tb,			\
		     if(*ta_data OP *tb_data) *r__data = 1;);		\
  }									\


TENSOR_IMPLEMENT_LOGICAL(lt,<)
TENSOR_IMPLEMENT_LOGICAL(gt,>)
TENSOR_IMPLEMENT_LOGICAL(le,<=)
TENSOR_IMPLEMENT_LOGICAL(ge,>=)
TENSOR_IMPLEMENT_LOGICAL(eq,==)
TENSOR_IMPLEMENT_LOGICAL(ne,!=)

#define LAB_IMPLEMENT_BASIC_FUNCTION(NAME, CFUNC)             \
  void THTensor_(NAME)(THTensor *r_, THTensor *t)                \
  {                                                           \
    THTensor_(resizeAs)(r_, t);                               \
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = CFUNC(*t_data);); \
  }                                                           \

#define LAB_IMPLEMENT_BASIC_FUNCTION_VALUE(NAME, CFUNC)                 \
  void THTensor_(NAME)(THTensor *r_, THTensor *t, real value)              \
  {                                                                     \
    THTensor_(resizeAs)(r_, t);                                         \
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = CFUNC(*t_data, value);); \
  }                                                                     \

#if defined(TH_REAL_IS_LONG)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,labs)
#endif /* long only part */

#if defined(TH_REAL_IS_INT)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,abs)
#endif /* int only part */

#if defined(TH_REAL_IS_BYTE)

#define TENSOR_IMPLEMENT_LOGICAL_SUM(NAME, OP, INIT_VALUE) \
  int THTensor_(NAME)(THTensor *tensor) \
  { \
    THArgCheck(tensor->nDimension > 0, 1, "empty Tensor"); \
    int sum = INIT_VALUE;                               \
    TH_TENSOR_APPLY(real, tensor, sum OP *tensor_data;); \
    return sum; \
  }

TENSOR_IMPLEMENT_LOGICAL_SUM(logicalall, &=, 1)
TENSOR_IMPLEMENT_LOGICAL_SUM(logicalany, |=, 0)

#endif /* Byte only part */

/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

LAB_IMPLEMENT_BASIC_FUNCTION(log,log)
LAB_IMPLEMENT_BASIC_FUNCTION(log1p,log1p)
LAB_IMPLEMENT_BASIC_FUNCTION(exp,exp)
LAB_IMPLEMENT_BASIC_FUNCTION(cos,cos)
LAB_IMPLEMENT_BASIC_FUNCTION(acos,acos)
LAB_IMPLEMENT_BASIC_FUNCTION(cosh,cosh)
LAB_IMPLEMENT_BASIC_FUNCTION(sin,sin)
LAB_IMPLEMENT_BASIC_FUNCTION(asin,asin)
LAB_IMPLEMENT_BASIC_FUNCTION(sinh,sinh)
LAB_IMPLEMENT_BASIC_FUNCTION(tan,tan)
LAB_IMPLEMENT_BASIC_FUNCTION(atan,atan)
LAB_IMPLEMENT_BASIC_FUNCTION(tanh,tanh)
LAB_IMPLEMENT_BASIC_FUNCTION_VALUE(pow,pow)
LAB_IMPLEMENT_BASIC_FUNCTION(sqrt,sqrt)
LAB_IMPLEMENT_BASIC_FUNCTION(ceil,ceil)
LAB_IMPLEMENT_BASIC_FUNCTION(floor,floor)
LAB_IMPLEMENT_BASIC_FUNCTION(round,round)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,fabs)

void THTensor_(atan2)(THTensor *r_, THTensor *tx, THTensor *ty)
{
  THTensor_(resizeAs)(r_, tx);
  TH_TENSOR_APPLY3(real, r_, real, tx, real, ty, *r__data = atan2(*tx_data,*ty_data););
}

void THTensor_(mean)(THTensor *r_, THTensor *t, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "invalid dimension");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                         sum += t_data[i*t_stride];
                       *r__data = (real)sum/t_size;);
}

void THTensor_(std)(THTensor *r_, THTensor *t, int dimension, int flag)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "invalid dimension");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       accreal sum2 = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                       {
                         real z = t_data[i*t_stride];
                         sum += z;
                         sum2 += z*z;
                       }

                       if(flag)
                       {
                         sum /= t_size;
                         sum2 /= t_size;
                         sum2 -= sum*sum;
                         sum2 = (sum2 < 0 ? 0 : sum2);
                         *r__data = (real)sqrt(sum2);
                       }
                       else
                       {
                         sum /= t_size;
                         sum2 /= t_size-1;
                         sum2 -= ((real)t_size)/((real)(t_size-1))*sum*sum;
                         sum2 = (sum2 < 0 ? 0 : sum2);
                         *r__data = (real)sqrt(sum2);
                       });
}

void THTensor_(var)(THTensor *r_, THTensor *t, int dimension, int flag)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "invalid dimension");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       accreal sum2 = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                       {
                         real z = t_data[i*t_stride];
                         sum += z;
                         sum2 += z*z;
                       }

                       if(flag)
                       {
                         sum /= t_size;
                         sum2 /= t_size;
                         sum2 -= sum*sum;
                         sum2 = (sum2 < 0 ? 0 : sum2);
                         *r__data = sum2;
                       }
                       else
                       {
                         sum /= t_size;
                         sum2 /= t_size-1;
                         sum2 -= ((real)t_size)/((real)(t_size-1))*sum*sum;
                         sum2 = (sum2 < 0 ? 0 : sum2);
                         *r__data = (real)sum2;
                       });
}

void THTensor_(norm)(THTensor *r_, THTensor *t, real value, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "invalid dimension");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  if(value == 0) {
    TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                         accreal sum = 0;
                         long i;
                         for(i = 0; i < t_size; i++)
                           sum += t_data[i*t_stride] != 0.0;
                         *r__data = sum;)
  } else {
    TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                         accreal sum = 0;
                         long i;
                         for(i = 0; i < t_size; i++)
                           sum += pow(fabs(t_data[i*t_stride]), value);
                         *r__data = pow(sum, 1.0/value);)
  }
}

accreal THTensor_(normall)(THTensor *tensor, real value)
{ 
  accreal sum = 0;
  if(value == 0) {
    TH_TENSOR_APPLY(real, tensor, sum += *tensor_data != 0.0;);
    return sum;
  } else if(value == 1) {
    TH_TENSOR_APPLY(real, tensor, sum += fabs(*tensor_data););
    return sum;
  } else if(value == 2) {
    TH_TENSOR_APPLY(real, tensor, accreal z = *tensor_data; sum += z*z;);
    return sqrt(sum);
  } else {
    TH_TENSOR_APPLY(real, tensor, sum += pow(fabs(*tensor_data), value););
    return pow(sum, 1.0/value);
  }
}

void THTensor_(renorm)(THTensor *res, THTensor *src, real value, int dimension, real maxnorm)
{
  int i;
  THTensor *rowR, *rowS;
  
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(src), 3, "invalid dimension");
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THTensor_(nDimension)(src) > 1, 1, "need at least 2 dimensions");
  
  rowR = THTensor_(new)();
  rowS = THTensor_(new)();
  
  THTensor_(resizeAs)(res, src);
   
  for (i=0; i<src->size[dimension]; i++)
  {
    real norm = 0;
    real new_norm;
    
    THTensor_(select)(rowS, src, dimension, i);
    THTensor_(select)(rowR, res, dimension, i);
    if (value == 1) {
      TH_TENSOR_APPLY(real, rowS, norm += fabs(*rowS_data););
    } else if (value == 2) {
      TH_TENSOR_APPLY(real, rowS, accreal z = *rowS_data; norm += z*z;);
    } else {
      TH_TENSOR_APPLY(real, rowS, norm += pow(fabs(*rowS_data), value););
    }

    norm = pow(norm, 1/value);

    if (norm > maxnorm)
    {
      new_norm = maxnorm / (norm + 1e-7);
      
      TH_TENSOR_APPLY2(
        real, rowR, real, rowS, 
        *rowR_data = (*rowS_data) * new_norm;
      )
    }
    else
      THTensor_(copy)(rowR, rowS);
  }
  
  THTensor_(free)(rowR);
  THTensor_(free)(rowS);
}

accreal THTensor_(dist)(THTensor *tensor, THTensor *src, real value)
{ 
  real sum = 0;
  TH_TENSOR_APPLY2(real, tensor, real, src, 
	sum += pow(fabs(*tensor_data - *src_data), value);)
  return pow(sum, 1.0/value);
}

accreal THTensor_(meanall)(THTensor *tensor)
{ 
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");
  return THTensor_(sumall)(tensor)/THTensor_(nElement)(tensor);
}  

accreal THTensor_(varall)(THTensor *tensor)
{ 
  accreal mean = THTensor_(meanall)(tensor);
  accreal sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += (*tensor_data - mean)*(*tensor_data - mean););
  sum /= (THTensor_(nElement)(tensor)-1);
  return sum;
}

accreal THTensor_(stdall)(THTensor *tensor)
{ 
  return sqrt(THTensor_(varall)(tensor));
}

void THTensor_(linspace)(THTensor *r_, real a, real b, long n)
{
  real i = 0;

  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");
  THArgCheck(a <= b, 2, "end range should be greater than start range");
  
  THTensor_(resize1d)(r_, n);

  if(n == 1) {
     TH_TENSOR_APPLY(real, r_,
             *r__data = a;
             i++;
           );
  } else {
     TH_TENSOR_APPLY(real, r_,
             *r__data = a + i*(b-a)/((real)(n-1));
             i++;
           );
  }
}

void THTensor_(logspace)(THTensor *r_, real a, real b, long n)
{
  real i = 0;

  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");
  THArgCheck(a <= b, 2, "end range should be greater than start range");
  
  THTensor_(resize1d)(r_, n);
  if(n == 1) {
    TH_TENSOR_APPLY(real, r_,
        *r__data = pow(10.0, a);
        i++;
        );
  } else {
    TH_TENSOR_APPLY(real, r_,
        *r__data = pow(10.0, a + i*(b-a)/((real)(n-1)));
        i++;
        );
  }
}

void THTensor_(rand)(THTensor *r_, THGenerator *_generator, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(uniform)(r_, _generator, 0, 1);
}

void THTensor_(randn)(THTensor *r_, THGenerator *_generator, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(normal)(r_, _generator, 0, 1);
}

void THTensor_(histc)(THTensor *hist, THTensor *tensor, long nbins, real minvalue, real maxvalue)
{
  THTensor *clone;
  real minval;
  real maxval;
  real bins;
  real *h_data;

  THTensor_(resize1d)(hist, nbins);
  THTensor_(zero)(hist);
  minval = minvalue;
  maxval = maxvalue;
  if (minval == maxval) 
  {
    minval = THTensor_(minall)(tensor);
    maxval = THTensor_(maxall)(tensor);
  }
  if (minval == maxval)
  {
    minval = minval - 1;
    maxval = maxval + 1;
  }
  bins = (real)(nbins)-1e-6;

  
  clone = THTensor_(newWithSize1d)(THTensor_(nElement)(tensor));
  THTensor_(copy)(clone,tensor);
  THTensor_(add)(clone, clone, -minval);
  THTensor_(div)(clone, clone, (maxval-minval));
  THTensor_(mul)(clone, clone, bins);
  THTensor_(floor)(clone, clone);
  THTensor_(add)(clone, clone, 1);
  
  h_data = THTensor_(data)(hist);

  TH_TENSOR_APPLY(real, clone,                                         \
                  if ((*clone_data <= nbins) && (*clone_data >= 1)) {  \
                    *(h_data + (int)(*clone_data) - 1) += 1;           \
                  });

  THTensor_(free)(clone);
}

#endif /* floating point only part */
#endif
