#ifndef _THMATH_H
#define _THMATH_H

static inline double TH_sigmoid(double value) {
  // faster version of 1.0 / (1.0 + exp(-value));
  double x = fabs(value);
  double x2 = x*x;
  double e = 1.0f + x + x2*0.555f + x2*x2*0.143f;
  return 1.0f / (1.0f + (value > 0 ? 1.0f / e : e));
}

static inline double TH_frac(double x) {
  return x - trunc(x);
}

static inline double TH_rsqrt(double x) {
  return 1.0 / sqrt(x);
}

static inline double TH_lerp(double a, double b, double weight) {
  return a + weight * (b-a);
}

static inline float TH_sigmoidf(float value) {
  return 1.0f / (1.0f + expf(-value));
}

static inline float TH_fracf(float x) {
  return x - truncf(x);
}

static inline float TH_rsqrtf(float x) {
  return 1.0f / sqrtf(x);
}

static inline float TH_lerpf(float a, float b, float weight) {
  return a + weight * (b-a);
}

#endif // _THMATH_H
