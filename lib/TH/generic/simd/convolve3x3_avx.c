#include <immintrin.h>
#include "common_simd.h"

#define CLEAR_AVX() _mm256_zeroupper()

void convolve_3x3_1_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_1()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS_3X3(1, 3, i)
  }
}

void convolve_3x3_2_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_2()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS_3X3(2, 4, i)
  }
}

void convolve_3x3_4_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_4()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS_3X3(4, 6, i)
  }
}

void convolve_3x3_5_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_5()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS_3X3(5, 7, i)
  }
}

void convolve_3x3_6_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_6()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS_3X3(6, 8, i)
  }
}

void convolve_3x3_7_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_7()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS_3X3(7, 9, i)
  }
}

void convolve_3x3_8_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount = count & 0xFFFFFFF8;
  DECLARE_OUTPUT_8()
  for (; i < alignedCount; i+=8) {
    CONVOLVE_8COLS_XROWS_3X3(8, 10, i)
  }
}

void convolve_3x3_64x64_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 64; i+=8)
  {
    DECLARE_OUTPUT_8()
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 0)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 8)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 16)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 24)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 32)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 40)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 48)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 56)
    output += outputStride * 8;
    image += inputStride * 8;
  }
}

void convolve_3x3_32x32_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 32; i+=8)
  {
    DECLARE_OUTPUT_8()
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 0)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 8)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 16)
    CONVOLVE_8COLS_XROWS_3X3(8, 10, 24)
    output += outputStride * 8;
    image += inputStride * 8;
  }
}

void convolve_3x3_16x16_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 12; i+=6)
  {
    DECLARE_OUTPUT_6()
    CONVOLVE_8COLS_XROWS_3X3(6, 8, 0)
    CONVOLVE_8COLS_XROWS_3X3(6, 8, 8)
    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_4()
  CONVOLVE_8COLS_XROWS_3X3(4, 6, 0)
  CONVOLVE_8COLS_XROWS_3X3(4, 6, 8)
}

void convolve_3x3_8x8_avx(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  DECLARE_OUTPUT_8()
  CONVOLVE_8COLS_XROWS_3X3(8, 10, 0)
}

void convolve_3x3_sse(float* output, float* input, float* kernel, long outRows, long outCols, long outStride, long inCols);

void convolve_3x3_avx(float* output, float* input, float* kernel, long outRows, long outCols, long outStride, long inCols) {
  long ic = inCols;
  long yy = 0;
  float* t_ = input;
  float* r_ = output;
  float* k_ = kernel;

  if((outRows == 64) && (outCols == 64)) {
    convolve_3x3_64x64_avx(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 32) && (outCols == 32)) {
    convolve_3x3_32x32_avx(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 16) && (outCols == 16)) {
    convolve_3x3_16x16_avx(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 8) && (outCols == 8)) {
    convolve_3x3_8x8_avx(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  for(; yy < (outRows & 0xFFFFFFF8); yy += 8) {
    float *pi_ = t_ + yy*ic;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_3x3_8_avx(r_, pis_, pw_, outCols, outStride, ic);
    r_ += (outStride * 8);
  }

  for(; yy < (outRows & 0xFFFFFFFC); yy += 4) {
    float *pi_ = t_ + yy*ic;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_3x3_4_avx(r_, pis_, pw_, outCols, outStride, ic);
    r_ += (outStride * 4);
  }

  for(; yy < (outRows & 0xFFFFFFFE); yy += 2) {
    float *pi_ = t_ + yy*ic;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_3x3_2_avx(r_, pis_, pw_, outCols, outStride, ic);
    r_ += (outStride * 2);
  }

  for(; yy < outRows; yy += 1) {
    float *pi_ = t_ + yy*ic;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_3x3_1_avx(r_, pis_, pw_, outCols, outStride, ic);
    r_ += (outStride * 1);
  }

  long procCols = outCols & 0xFFFFFFF8; // avx version processes 8 cols at a time
  long remCols = outCols - procCols;

  //process the rest using sse
  if( remCols > 0) {
    CLEAR_AVX();
    convolve_3x3_sse(&output[procCols], &input[procCols], kernel, outRows, remCols, outStride, inCols);
  }
}