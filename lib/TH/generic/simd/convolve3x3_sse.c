#include <smmintrin.h>
#include "common_simd.h"


/* SSE variants */
void convolve_3x3_1_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_1()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS_3X3(1, 5, i)
  }
  for (; i < (count); i++) {
    float output0 = output[i + outputStride * 0];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
  }
}

void convolve_3x3_2_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_2()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS_3X3(2, 6, i)
  }
  for (; i < (count); i++) {
    float output0 = output[i + outputStride * 0];
    float output1 = output[i + outputStride * 1];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
        output1 += weight[5 * row + col] * image[i + (row + 1) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
    output[i + outputStride * 1] = output1;
  }
}

void convolve_3x3_4_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_4()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS_3X3(4, 8, i)
  }
  for (; i < (count); i++) {
    float output0 = output[i + outputStride * 0];
    float output1 = output[i + outputStride * 1];
    float output2 = output[i + outputStride * 2];
    float output3 = output[i + outputStride * 3];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
        output1 += weight[5 * row + col] * image[i + (row + 1) * inputStride + col];
        output2 += weight[5 * row + col] * image[i + (row + 2) * inputStride + col];
        output3 += weight[5 * row + col] * image[i + (row + 3) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
    output[i + outputStride * 1] = output1;
    output[i + outputStride * 2] = output2;
    output[i + outputStride * 3] = output3;
  }
}

void convolve_3x3_6_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_6()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS_3X3(6, 10, i)
  }
  for (; i<(count); i++) {
    float output0 = output[i + outputStride * 0];
    float output1 = output[i + outputStride * 1];
    float output2 = output[i + outputStride * 2];
    float output3 = output[i + outputStride * 3];
    float output4 = output[i + outputStride * 4];
    float output5 = output[i + outputStride * 5];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
        output1 += weight[5 * row + col] * image[i + (row + 1) * inputStride + col];
        output2 += weight[5 * row + col] * image[i + (row + 2) * inputStride + col];
        output3 += weight[5 * row + col] * image[i + (row + 3) * inputStride + col];
        output4 += weight[5 * row + col] * image[i + (row + 4) * inputStride + col];
        output5 += weight[5 * row + col] * image[i + (row + 5) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
    output[i + outputStride * 1] = output1;
    output[i + outputStride * 2] = output2;
    output[i + outputStride * 3] = output3;
    output[i + outputStride * 4] = output4;
    output[i + outputStride * 5] = output5;
  }
}

void convolve_3x3_8_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  long i = 0;
  long alignedCount4 = count & 0xFFFFFFFC;
  DECLARE_OUTPUT_8()
  for (; i < alignedCount4; i+=4) {
    CONVOLVE_4COLS_XROWS_3X3(8, 12, i)
  }
  for (; i<(count); i++) {
    float output0 = output[i + outputStride * 0];
    float output1 = output[i + outputStride * 1];
    float output2 = output[i + outputStride * 2];
    float output3 = output[i + outputStride * 3];
    float output4 = output[i + outputStride * 4];
    float output5 = output[i + outputStride * 5];
    float output6 = output[i + outputStride * 6];
    float output7 = output[i + outputStride * 7];
    int row;
    for (row = 0; row < 5; row++) {
      int col;
      for (col = 0; col < 5; col++) {
        output0 += weight[5 * row + col] * image[i + (row + 0) * inputStride + col];
        output1 += weight[5 * row + col] * image[i + (row + 1) * inputStride + col];
        output2 += weight[5 * row + col] * image[i + (row + 2) * inputStride + col];
        output3 += weight[5 * row + col] * image[i + (row + 3) * inputStride + col];
        output4 += weight[5 * row + col] * image[i + (row + 4) * inputStride + col];
        output5 += weight[5 * row + col] * image[i + (row + 5) * inputStride + col];
        output6 += weight[5 * row + col] * image[i + (row + 6) * inputStride + col];
        output7 += weight[5 * row + col] * image[i + (row + 7) * inputStride + col];
      }
    }
    output[i + outputStride * 0] = output0;
    output[i + outputStride * 1] = output1;
    output[i + outputStride * 2] = output2;
    output[i + outputStride * 3] = output3;
    output[i + outputStride * 4] = output4;
    output[i + outputStride * 5] = output5;
    output[i + outputStride * 6] = output6;
    output[i + outputStride * 7] = output7;
  }
}

#define UNROLL_SSE_CONVOLUTION 0
#if (UNROLL_SSE_CONVOLUTION)

void convolve_3x3_64x64_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 60; i+=6)
  {
    DECLARE_OUTPUT_6()
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 0)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 4)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 8)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 12)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 16)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 20)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 24)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 28)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 32)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 36)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 40)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 44)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 48)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 52)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 56)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 60)
    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_4()
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 0)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 4)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 8)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 12)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 16)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 20)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 24)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 28)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 32)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 36)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 40)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 44)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 48)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 52)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 56)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 60)
}

void convolve_3x3_32x32_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 30; i+=6)
  {
    DECLARE_OUTPUT_6()

      CONVOLVE_4COLS_XROWS_3X3(6, 10, 0)
      CONVOLVE_4COLS_XROWS_3X3(6, 10, 4)
      CONVOLVE_4COLS_XROWS_3X3(6, 10, 8)
      CONVOLVE_4COLS_XROWS_3X3(6, 10, 12)
      CONVOLVE_4COLS_XROWS_3X3(6, 10, 16)
      CONVOLVE_4COLS_XROWS_3X3(6, 10, 20)
      CONVOLVE_4COLS_XROWS_3X3(6, 10, 24)
      CONVOLVE_4COLS_XROWS_3X3(6, 10, 28)

    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_2()
  CONVOLVE_4COLS_XROWS_3X3(2, 6, 0)
  CONVOLVE_4COLS_XROWS_3X3(2, 6, 4)
  CONVOLVE_4COLS_XROWS_3X3(2, 6, 8)
  CONVOLVE_4COLS_XROWS_3X3(2, 6, 12)
  CONVOLVE_4COLS_XROWS_3X3(2, 6, 16)
  CONVOLVE_4COLS_XROWS_3X3(2, 6, 20)
  CONVOLVE_4COLS_XROWS_3X3(2, 6, 24)
  CONVOLVE_4COLS_XROWS_3X3(2, 6, 28)
}

void convolve_3x3_16x16_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  for(int i = 0; i < 12; i+=6)
  {
    DECLARE_OUTPUT_6()
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 0)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 4)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 8)
    CONVOLVE_4COLS_XROWS_3X3(6, 10, 12)
    output += outputStride * 6;
    image += inputStride * 6;
  }
  DECLARE_OUTPUT_4()
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 0)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 4)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 8)
  CONVOLVE_4COLS_XROWS_3X3(4, 8, 12)
}

void convolve_3x3_8x8_sse(float* output, float* image, float* weight, long count, long outputStride, long inputStride) {
  DECLARE_OUTPUT_8()
  CONVOLVE_4COLS_XROWS_3X3(8, 0)
  CONVOLVE_4COLS_XROWS_3X3(8, 4)
}

#endif

void convolve_3x3_sse(float* output, float* input, float* kernel, long outRows, long outCols, long outStride, long inCols) {
  long yy = 0;
  float* t_ = input;
  float* r_ = output;
  float* k_ = kernel;
#if (UNROLL_SSE_CONVOLUTION)
  if((outRows == 64) && (outCols == 64)) {
    convolve_3x3_64x64_sse(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 32) && (outCols == 32)) {
    convolve_3x3_32x32_sse(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 16) && (outCols == 16)) {
    convolve_3x3_16x16_sse(output, input, kernel, outRows, outStride, inCols);
    return;
  }

  if((outRows == 8) && (outCols == 8)) {
    convolve_3x3_8x8_sse(output, input, kernel, outRows, outStride, inCols);
    return;
  }
#endif
  for(; yy < (outRows & 0xFFFFFFF0); yy += 8) {
    float *pi_ = t_ + yy*inCols;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_3x3_8_sse(r_, pis_, pw_, outCols, outStride, inCols);
    r_ += (outStride * 8);
  }

  for(; yy < (outRows & 0xFFFFFFFC); yy += 4) {
    float *pi_ = t_ + yy*inCols;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_3x3_4_sse(r_, pis_, pw_, outCols, outStride, inCols);
    r_ += (outStride * 4);
  }

  for(; yy < (outRows & 0xFFFFFFFE); yy += 2) {
    float *pi_ = t_ + yy*inCols;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_3x3_2_sse(r_, pis_, pw_, outCols, outStride, inCols);
    r_ += (outStride * 2);
  }

  for(; yy < outRows; yy += 1) {
    float *pi_ = t_ + yy*inCols;
    float *pw_ = k_;
    float *pis_ = pi_;
    convolve_3x3_1_sse(r_, pis_, pw_, outCols, outStride, inCols);
    r_ += (outStride * 1);
  }
}
