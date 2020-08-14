/* ================================================================================
x86 intrinsic function wrapper for AVX, AVX-2, AVX-512 instruction sets
Authors: Hua Huang   <huangh223@gatech.edu>
         Xin Xing    <xxing02@gmail.com>
         Edmond Chow <echow@cc.gatech.edu>

Tested platforms:
1. Intel Xeon Platinum 8160, AVX-512, CentOS 7.6,             ICC 18.0.4
2. Intel Xeon Gold 6226,     AVX-512, RHEL   7.6,             ICC 19.0.5
3. Intel Core i7-8550U,      AVX-2,   WSL Ubuntu 18.04.4 LTS, GCC 7.4.0
4. Intel Xeon E5-2670,       AVX,     WSL Ubuntu 18.04.4 LTS, GCC 7.4.0
5. Intel Xeon E5-1620,       AVX,     Ubuntu 18.04.4 LTS,     ICC 18.0.5, GCC 7.5.0
6. AMD Threadripper 2950X,   AVX-2,   Ubuntu 18.04.4 LTS,     GCC 7.5.0  <-- Thanks to KaraRyougi@GitHub !
7. Intel Core i5-7300U,      AVX-2,   macOS 10.15.4,          GCC 9.2.0
8. Intel Core i5-4570S,      AVX-2,   Ubuntu 18.04.4 LTS,     ICC 19.1.1

Naming: vec_<operation>_<s/d>, suffix s is for float, d is for double 
vec_zero_*()            : Set all lanes of a vector to zero and return this vector
vec_set1_*(a)           : Set all lanes of a vector to value <a> and return this vector
vec_bcast_*(a)          : Set all lanes of a vector to value <a[0]> and return this vector
vec_load_*(a)           : Load a vector from an address <a> which must be aligned to required bits
vec_loadu_*(a)          : Load a vector from an address <a> which may not be aligned to required bits
vec_store_*(a, b)       : Store a vector <b> to an address <a> which must be aligned to required bits
vec_storeu_*(a, b)      : Store a vector <b> to an address <a> which may not be aligned to required bits
vec_add_*(a, b)         : Return lane-wise <a[i]> + <b[i]>
vec_sub_*(a, b)         : Return lane-wise <a[i]> - <b[i]>
vec_mul_*(a, b)         : Return lane-wise <a[i]> * <b[i]>
vec_div_*(a, b)         : Return lane-wise <a[i]> / <b[i]>
vec_abs_*(a)            : Return lane-wise abs(<a[i]>)
vec_sqrt_*(a)           : Return lane-wise sqrt(<a[i]>)
vec_fmadd_* (a, b, c)   : Return lane-wise Fused Multiply-Add            <a[i]> * <b[i]> + <c[i]>
vec_fnmadd_*(a, b, c)   : Return lane-wise Fused Negative Multiply-Add  -<a[i]> * <b[i]> + <c[i]>
vec_fmsub_* (a, b, c)   : Return lane-wise Fused Multiply-Sub intrinsic  <a[i]> * <b[i]> - <c[i]>
vec_max_*(a, b)         : Return lane-wise max(<a[i]>, <b[i]>)
vec_min_*(a, b)         : Return lane-wise min(<a[i]>, <b[i]>)
vec_cmp_eq_*(a, b)      : Return lane-wise if(<a[i]> == <b[i]>)
vec_cmp_neq_*(a, b)     : Return lane-wise if(<a[i]> != <b[i]>)
vec_cmp_lt_*(a, b)      : Return lane-wise if(<a[i]> <  <b[i]>)
vec_cmp_le_*(a, b)      : Return lane-wise if(<a[i]> <= <b[i]>)
vec_cmp_gt_*(a, b)      : Return lane-wise if(<a[i]> >  <b[i]>)
vec_cmp_ge_*(a, b)      : Return lane-wise if(<a[i]> >= <b[i]>)
vec_blend_*(a, b, m)    : Return lane-wise (<m[i]> == 1 ? <b[i]> : <a[i]>).
vec_reduce_add_*(a)     : Return a single value sum(<a[i]>)
vec_frsqrt_pf_*()       : Return scaling prefactor for vec_frsqrt_*()
vec_frsqrt_*(r2)        : Return lane-wise fast reverse square root ( <r2[i]> == 0 ? 0 : 1/sqrt(<r2[i]>)/vec_frsqrt_pf_*() ).
vec_log_*(a)            : Return lane-wise natural logarithm ln(<a[i]>)
vec_log2_*(a)           : Return lane-wise base-2  logarithm log2(<a[i]>)
vec_log10_*(a)          : Return lane-wise base-10 logarithm log10(<a[i]>)
vec_exp_*(a)            : Return lane-wise e^(<a[i]>)
vec_exp2_*(a)           : Return lane-wise 2^(<a[i]>)
vec_exp10_*(a)          : Return lane-wise 10^(<a[i]>)
vec_pow_*(a, b)         : Return lane-wise (<a[i]>)^(<b[i]>)
vec_sin_*(a)            : Return lane-wise sin(<a[i]>)
vec_cos_*(a)            : Return lane-wise cos(<a[i]>)
vec_erf_*(a)            : Return lane-wise erf(<a[i]>)

Reference:
1. Intel Intrinsic Guide    : https://software.intel.com/sites/landingpage/IntrinsicsGuide/
2. Compiler Explorer        : https://godbolt.org/
3. AVX vec_reduce_add_s     : https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
4. AVX vec_reduce_add_d     : https://www.oipapio.com/question-771803
5. Fast inverse square root : https://github.com/dmalhotra/pvfmm/blob/develop/include/intrin_wrapper.hpp
6. GCC SIMD math functions  : https://stackoverflow.com/questions/40475140/mathematical-functions-for-simd-registers
================================================================================ */ 

#ifndef __X86_INTRIN_WRAPPER_H__
#define __X86_INTRIN_WRAPPER_H__

#include <math.h>
#include <x86intrin.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NEWTON_ITER
#define NEWTON_ITER 2   // Two Newton iterations is usually sufficient for rsqrt using double type
#endif

#if !defined(__AVX__)
#error Your processor or compiler does not support AVX instruction set, cannot use this x86_intrin_wrapper.h
#endif

#if !defined(USE_AVX) && !defined(USE_AVX512)
#if defined(__AVX512F__)
#define USE_AVX512
#elif defined(__AVX__)
#define USE_AVX
#endif
#endif

#if defined(__AVX512F__) && defined(USE_AVX)
#warning AVX-512 instruction set detected, but AVX/AVX2 instructions will be used due to -DUSE_AVX flag
#endif

#ifdef USE_AVX

#define SIMD_LEN_S  8
#define SIMD_LEN_D  4

#ifdef __AVX__

#define vec_f       __m256
#define vec_d       __m256d
#define vec_cmp_f   __m256
#define vec_cmp_d   __m256d

union vec8f
{
    __m256 v;
    float  f[SIMD_LEN_S];
};

union vec4d
{
    __m256d v;
    double  d[SIMD_LEN_D];
};

static inline __m256  vec_zero_s() { return _mm256_setzero_ps(); }
static inline __m256d vec_zero_d() { return _mm256_setzero_pd(); }

static inline __m256  vec_set1_s(const float  a)  { return _mm256_set1_ps(a); }
static inline __m256d vec_set1_d(const double a)  { return _mm256_set1_pd(a); }

static inline __m256  vec_bcast_s(float  const *a)  { return _mm256_broadcast_ss(a); }
static inline __m256d vec_bcast_d(double const *a)  { return _mm256_broadcast_sd(a); }

static inline __m256  vec_load_s (float  const *a)  { return _mm256_load_ps(a);  }
static inline __m256d vec_load_d (double const *a)  { return _mm256_load_pd(a);  }

static inline __m256  vec_loadu_s(float  const *a)  { return _mm256_loadu_ps(a); }
static inline __m256d vec_loadu_d(double const *a)  { return _mm256_loadu_pd(a); }

static inline void vec_store_s (float  *a, const __m256  b) { _mm256_store_ps(a, b);  }
static inline void vec_store_d (double *a, const __m256d b) { _mm256_store_pd(a, b);  }

static inline void vec_storeu_s(float  *a, const __m256  b) { _mm256_storeu_ps(a, b); }
static inline void vec_storeu_d(double *a, const __m256d b) { _mm256_storeu_pd(a, b); }

static inline __m256  vec_add_s(const __m256  a, const __m256  b) { return _mm256_add_ps(a, b); }
static inline __m256d vec_add_d(const __m256d a, const __m256d b) { return _mm256_add_pd(a, b); }

static inline __m256  vec_sub_s(const __m256  a, const __m256  b) { return _mm256_sub_ps(a, b); }
static inline __m256d vec_sub_d(const __m256d a, const __m256d b) { return _mm256_sub_pd(a, b); }

static inline __m256  vec_mul_s(const __m256  a, const __m256  b) { return _mm256_mul_ps(a, b); }
static inline __m256d vec_mul_d(const __m256d a, const __m256d b) { return _mm256_mul_pd(a, b); }

static inline __m256  vec_div_s(const __m256  a, const __m256  b) { return _mm256_div_ps(a, b); }
static inline __m256d vec_div_d(const __m256d a, const __m256d b) { return _mm256_div_pd(a, b); }

static inline __m256  vec_abs_s(const __m256  a) { return _mm256_max_ps(a, _mm256_sub_ps(_mm256_setzero_ps(), a)); }
static inline __m256d vec_abs_d(const __m256d a) { return _mm256_max_pd(a, _mm256_sub_pd(_mm256_setzero_pd(), a)); }

static inline __m256  vec_sqrt_s(const __m256  a) { return _mm256_sqrt_ps(a); }
static inline __m256d vec_sqrt_d(const __m256d a) { return _mm256_sqrt_pd(a); }

#ifdef __AVX2__
static inline __m256  vec_fmadd_s (const __m256  a, const __m256  b, const __m256  c) { return _mm256_fmadd_ps(a, b, c);  }
static inline __m256d vec_fmadd_d (const __m256d a, const __m256d b, const __m256d c) { return _mm256_fmadd_pd(a, b, c);  }

static inline __m256  vec_fnmadd_s(const __m256  a, const __m256  b, const __m256  c) { return _mm256_fnmadd_ps(a, b, c); }
static inline __m256d vec_fnmadd_d(const __m256d a, const __m256d b, const __m256d c) { return _mm256_fnmadd_pd(a, b, c); }

static inline __m256  vec_fmsub_s (const __m256  a, const __m256  b, const __m256  c) { return _mm256_fmsub_ps(a, b, c);  }
static inline __m256d vec_fmsub_d (const __m256d a, const __m256d b, const __m256d c) { return _mm256_fmsub_pd(a, b, c);  }
#else   // Else of "#ifdef __AVX2__"
static inline __m256  vec_fmadd_s (const __m256  a, const __m256  b, const __m256  c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
static inline __m256d vec_fmadd_d (const __m256d a, const __m256d b, const __m256d c) { return _mm256_add_pd(_mm256_mul_pd(a, b), c); }

static inline __m256  vec_fnmadd_s(const __m256  a, const __m256  b, const __m256  c) { return _mm256_sub_ps(c, _mm256_mul_ps(a, b)); }
static inline __m256d vec_fnmadd_d(const __m256d a, const __m256d b, const __m256d c) { return _mm256_sub_pd(c, _mm256_mul_pd(a, b)); }

static inline __m256  vec_fmsub_s (const __m256  a, const __m256  b, const __m256  c) { return _mm256_sub_ps(_mm256_mul_ps(a, b), c); }
static inline __m256d vec_fmsub_d (const __m256d a, const __m256d b, const __m256d c) { return _mm256_sub_pd(_mm256_mul_pd(a, b), c); }
#endif  // End of "#ifdef __AVX2__"

static inline __m256  vec_max_s(const __m256  a, const __m256  b) { return _mm256_max_ps(a, b); }
static inline __m256d vec_max_d(const __m256d a, const __m256d b) { return _mm256_max_pd(a, b); }

static inline __m256  vec_min_s(const __m256  a, const __m256  b) { return _mm256_min_ps(a, b); }
static inline __m256d vec_min_d(const __m256d a, const __m256d b) { return _mm256_min_pd(a, b); }

static inline __m256  vec_cmp_eq_s (const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OS);  }
static inline __m256d vec_cmp_eq_d (const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OS);  }

static inline __m256  vec_cmp_neq_s(const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_OS); }
static inline __m256d vec_cmp_neq_d(const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_OS); }

static inline __m256  vec_cmp_lt_s (const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS);  }
static inline __m256d vec_cmp_lt_d (const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS);  }

static inline __m256  vec_cmp_le_s (const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS);  }
static inline __m256d vec_cmp_le_d (const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LE_OS);  }

static inline __m256  vec_cmp_gt_s (const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_GT_OS);  }
static inline __m256d vec_cmp_gt_d (const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_GT_OS);  }

static inline __m256  vec_cmp_ge_s (const __m256  a, const __m256  b) { return _mm256_cmp_ps(a, b, _CMP_GE_OS);  }
static inline __m256d vec_cmp_ge_d (const __m256d a, const __m256d b) { return _mm256_cmp_pd(a, b, _CMP_GE_OS);  }

static inline __m256  vec_blend_s(const __m256  a, const __m256  b, const __m256  mask) { return _mm256_blendv_ps(a, b, mask); }
static inline __m256d vec_blend_d(const __m256d a, const __m256d b, const __m256d mask) { return _mm256_blendv_pd(a, b, mask); }

static inline float vec_reduce_add_s(const __m256  a) 
{
    __m128 hi4  = _mm256_extractf128_ps(a, 1);
    __m128 lo4  = _mm256_castps256_ps128(a);
    __m128 sum4 = _mm_add_ps(lo4, hi4);
    __m128 lo2  = sum4;
    __m128 hi2  = _mm_movehl_ps(sum4, sum4);
    __m128 sum2 = _mm_add_ps(lo2, hi2);
    __m128 lo   = sum2;
    __m128 hi   = _mm_shuffle_ps(sum2, sum2, 0x1);
    __m128 sum  = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}
static inline double vec_reduce_add_d(const __m256d a) 
{
    __m128d lo = _mm256_castpd256_pd128(a);
    __m128d hi = _mm256_extractf128_pd(a, 1); 
    lo = _mm_add_pd(lo, hi);
    __m128d hi64 = _mm_unpackhi_pd(lo, lo);
    return  _mm_cvtsd_f64(_mm_add_sd(lo, hi64));
}

static inline __m256  vec_arsqrt_s(const __m256  r2)
{
    __m256 rsqrt = _mm256_rsqrt_ps(r2);
    __m256 cmp0  = _mm256_cmp_ps(r2, vec_zero_s(), _CMP_EQ_OS);
    return _mm256_andnot_ps(cmp0, rsqrt);
}
static inline __m256d vec_arsqrt_d(const __m256d r2)
{ 
    __m128 r2_s    = _mm256_cvtpd_ps(r2);
    __m128 rsqrt_s = _mm_rsqrt_ps(r2_s);
    __m128 cmp0    = _mm_cmpeq_ps(r2_s, _mm_setzero_ps());
    __m128 ret_s   = _mm_andnot_ps(cmp0, rsqrt_s);
    return _mm256_cvtps_pd(ret_s); 
}

#ifdef __INTEL_COMPILER
static inline __m256  vec_log_s  (const __m256  a) { return _mm256_log_ps(a);   }
static inline __m256d vec_log_d  (const __m256d a) { return _mm256_log_pd(a);   }

static inline __m256  vec_log2_s (const __m256  a) { return _mm256_log2_ps(a);  }
static inline __m256d vec_log2_d (const __m256d a) { return _mm256_log2_pd(a);  }

static inline __m256  vec_log10_s(const __m256  a) { return _mm256_log10_ps(a); }
static inline __m256d vec_log10_d(const __m256d a) { return _mm256_log10_pd(a); }

static inline __m256  vec_exp_s  (const __m256  a) { return _mm256_exp_ps(a);   }
static inline __m256d vec_exp_d  (const __m256d a) { return _mm256_exp_pd(a);   }

static inline __m256  vec_exp2_s (const __m256  a) { return _mm256_exp2_ps(a);  }
static inline __m256d vec_exp2_d (const __m256d a) { return _mm256_exp2_pd(a);  }

static inline __m256  vec_exp10_s(const __m256  a) { return _mm256_exp10_ps(a); }
static inline __m256d vec_exp10_d(const __m256d a) { return _mm256_exp10_pd(a); }

static inline __m256  vec_pow_s  (const __m256  a, const __m256  b) { return _mm256_pow_ps(a, b); }
static inline __m256d vec_pow_d  (const __m256d a, const __m256d b) { return _mm256_pow_pd(a, b); }

static inline __m256  vec_sin_s  (const __m256  a) { return _mm256_sin_ps(a);   }
static inline __m256d vec_sin_d  (const __m256d a) { return _mm256_sin_pd(a);   }

static inline __m256  vec_cos_s  (const __m256  a) { return _mm256_cos_ps(a);   }
static inline __m256d vec_cos_d  (const __m256d a) { return _mm256_cos_pd(a);   }

static inline __m256  vec_erf_s  (const __m256  a) { return _mm256_erf_ps(a);   }
static inline __m256d vec_erf_d  (const __m256d a) { return _mm256_erf_pd(a);   }

#else  // Else of "#ifdef __INTEL_COMPILER"

#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22
__m256  _ZGVdN8v_logf(__m256  a);
__m256d _ZGVdN4v_log (__m256d a);
static inline __m256  vec_log_s  (const __m256  a) { return _ZGVdN8v_logf(a);   }
static inline __m256d vec_log_d  (const __m256d a) { return _ZGVdN4v_log (a);   }

__m256  _ZGVdN8v_expf(__m256  a);
__m256d _ZGVdN4v_exp (__m256d a);
static inline __m256  vec_exp_s  (const __m256  a) { return _ZGVdN8v_expf(a);   }
static inline __m256d vec_exp_d  (const __m256d a) { return _ZGVdN4v_exp (a);   }

__m256  _ZGVdN8vv_powf(__m256  a, __m256  b);
__m256d _ZGVdN4vv_pow (__m256d a, __m256d b);
static inline __m256  vec_pow_s  (const __m256  a, const __m256  b) { return _ZGVdN8vv_powf(a, b); }
static inline __m256d vec_pow_d  (const __m256d a, const __m256d b) { return _ZGVdN4vv_pow (a, b); }

__m256  _ZGVdN8v_sinf(__m256  a);
__m256d _ZGVdN4v_sin (__m256d a);
static inline __m256  vec_sin_s  (const __m256  a) { return _ZGVdN8v_sinf(a);   }
static inline __m256d vec_sin_d  (const __m256d a) { return _ZGVdN4v_sin (a);   }

__m256  _ZGVdN8v_cosf(__m256  a);
__m256d _ZGVdN4v_cos (__m256d a);
static inline __m256  vec_cos_s  (const __m256  a) { return _ZGVdN8v_cosf(a);   }
static inline __m256d vec_cos_d  (const __m256d a) { return _ZGVdN4v_cos (a);   }

#else   // Else of "#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22"

#warning Your compiler or GLIBC does not support vectorized log(), pow(), and exp(), x86_intrin_wrapper.h will use simulated implementations. 
static inline __m256  vec_log_s(__m256  x)
{
    int i;
    union vec8f u = {x}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = logf(u.f[i]);
    return res.v;
}
static inline __m256d vec_log_d(__m256d x)
{
    int i;
    union vec4d u = {x}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = log(u.d[i]);
    return res.v;
}

static inline __m256  vec_exp_s(__m256  x)
{
    int i;
    union vec8f u = {x}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = expf(u.f[i]);
    return res.v;
}
static inline __m256d vec_exp_d(__m256d x)
{
    int i;
    union vec4d u = {x}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = exp(u.d[i]);
    return res.v;
}

static inline __m256  vec_pow_s(__m256  a, __m256  b)
{
    int i;
    union vec8f ua = {a}, ub = {b}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = powf(ua.f[i], ub.f[i]);
    return res.v;
}
static inline __m256d vec_pow_d(__m256d a, __m256d b)
{
    int i;
    union vec4d ua = {a}, ub = {b}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = pow(ua.d[i], ub.d[i]);
    return res.v;
}

static inline __m256  vec_sin_s(__m256  a)
{
    int i;
    union vec8f ua = {a}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = sinf(ua.f[i]);
    return res.v;
}
static inline __m256d vec_sin_d(__m256d a)
{
    int i;
    union vec4d ua = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = sin(ua.d[i]);
    return res.v;
}

static inline __m256  vec_cos_s(__m256  a)
{
    int i;
    union vec8f ua = {a}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = cosf(ua.f[i]);
    return res.v;
}
static inline __m256d vec_cos_d(__m256d a)
{
    int i;
    union vec4d ua = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = cos(ua.d[i]);
    return res.v;
}

#endif  // End of "#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22"

static inline __m256  vec_erf_s(__m256  a)
{
    int i;
    union vec8f ua = {a}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = erff(ua.f[i]);
    return res.v;
}
static inline __m256d vec_erf_d(__m256d a)
{
    int i;
    union vec4d ua = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = erf(ua.d[i]);
    return res.v;
}

#endif  // End of "#ifdef __INTEL_COMPILER"

#endif  // End of "#ifdef __AVX__"
#endif  // End of "#ifdef USE_AVX"


#ifdef USE_AVX512

#define SIMD_LEN_S  16
#define SIMD_LEN_D  8

#ifdef __AVX512F__

#define vec_f       __m512
#define vec_d       __m512d
#define vec_cmp_f   __mmask16
#define vec_cmp_d   __mmask8

union vec16f
{
    __m512 v;
    float  f[SIMD_LEN_S];
};

union vec8d
{
    __m512d v;
    double  d[SIMD_LEN_D];
};

static inline __m512  vec_zero_s() { return _mm512_setzero_ps(); }
static inline __m512d vec_zero_d() { return _mm512_setzero_pd(); }

static inline __m512  vec_set1_s(const float  a)  { return _mm512_set1_ps(a); }
static inline __m512d vec_set1_d(const double a)  { return _mm512_set1_pd(a); }

static inline __m512  vec_bcast_s(float  const *a)  { return _mm512_set1_ps(a[0]); }
static inline __m512d vec_bcast_d(double const *a)  { return _mm512_set1_pd(a[0]); }

static inline __m512  vec_load_s (float  const *a)  { return _mm512_load_ps(a);  }
static inline __m512d vec_load_d (double const *a)  { return _mm512_load_pd(a);  }

static inline __m512  vec_loadu_s(float  const *a)  { return _mm512_loadu_ps(a); }
static inline __m512d vec_loadu_d(double const *a)  { return _mm512_loadu_pd(a); }

static inline void vec_store_s (float  *a, const __m512  b) { _mm512_store_ps(a, b);  }
static inline void vec_store_d (double *a, const __m512d b) { _mm512_store_pd(a, b);  }

static inline void vec_storeu_s(float  *a, const __m512  b) { _mm512_storeu_ps(a, b); }
static inline void vec_storeu_d(double *a, const __m512d b) { _mm512_storeu_pd(a, b); }

static inline __m512  vec_add_s(const __m512  a, const __m512  b) { return _mm512_add_ps(a, b); }
static inline __m512d vec_add_d(const __m512d a, const __m512d b) { return _mm512_add_pd(a, b); }

static inline __m512  vec_sub_s(const __m512  a, const __m512  b) { return _mm512_sub_ps(a, b); }
static inline __m512d vec_sub_d(const __m512d a, const __m512d b) { return _mm512_sub_pd(a, b); }

static inline __m512  vec_mul_s(const __m512  a, const __m512  b) { return _mm512_mul_ps(a, b); }
static inline __m512d vec_mul_d(const __m512d a, const __m512d b) { return _mm512_mul_pd(a, b); }

static inline __m512  vec_div_s(const __m512  a, const __m512  b) { return _mm512_div_ps(a, b); }
static inline __m512d vec_div_d(const __m512d a, const __m512d b) { return _mm512_div_pd(a, b); }

static inline __m512  vec_abs_s(const __m512  a) { return _mm512_abs_ps(a); }
static inline __m512d vec_abs_d(const __m512d a) { return _mm512_abs_pd(a); }

static inline __m512  vec_sqrt_s(const __m512  a) { return _mm512_sqrt_ps(a); }
static inline __m512d vec_sqrt_d(const __m512d a) { return _mm512_sqrt_pd(a); }

static inline __m512  vec_fmadd_s (const __m512  a, const __m512  b, const __m512  c) { return _mm512_fmadd_ps(a, b, c);  }
static inline __m512d vec_fmadd_d (const __m512d a, const __m512d b, const __m512d c) { return _mm512_fmadd_pd(a, b, c);  }

static inline __m512  vec_fnmadd_s(const __m512  a, const __m512  b, const __m512  c) { return _mm512_fnmadd_ps(a, b, c); }
static inline __m512d vec_fnmadd_d(const __m512d a, const __m512d b, const __m512d c) { return _mm512_fnmadd_pd(a, b, c); }

static inline __m512  vec_fmsub_s (const __m512  a, const __m512  b, const __m512  c) { return _mm512_fmsub_ps(a, b, c);  }
static inline __m512d vec_fmsub_d (const __m512d a, const __m512d b, const __m512d c) { return _mm512_fmsub_pd(a, b, c);  }

static inline __m512  vec_max_s(const __m512  a, const __m512  b) { return _mm512_max_ps(a, b); }
static inline __m512d vec_max_d(const __m512d a, const __m512d b) { return _mm512_max_pd(a, b); }

static inline __m512  vec_min_s(const __m512  a, const __m512  b) { return _mm512_min_ps(a, b); }
static inline __m512d vec_min_d(const __m512d a, const __m512d b) { return _mm512_min_pd(a, b); }

static inline __mmask16 vec_cmp_eq_s (const __m512  a, const __m512  b) { return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OS);  }
static inline __mmask8  vec_cmp_eq_d (const __m512d a, const __m512d b) { return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OS);  }

static inline __mmask16 vec_cmp_neq_s(const __m512  a, const __m512  b) { return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OS); }
static inline __mmask8  vec_cmp_neq_d(const __m512d a, const __m512d b) { return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OS); }

static inline __mmask16 vec_cmp_lt_s (const __m512  a, const __m512  b) { return _mm512_cmp_ps_mask(a, b, _CMP_LT_OS);  }
static inline __mmask8  vec_cmp_lt_d (const __m512d a, const __m512d b) { return _mm512_cmp_pd_mask(a, b, _CMP_LT_OS);  }

static inline __mmask16 vec_cmp_le_s (const __m512  a, const __m512  b) { return _mm512_cmp_ps_mask(a, b, _CMP_LE_OS);  }
static inline __mmask8  vec_cmp_le_d (const __m512d a, const __m512d b) { return _mm512_cmp_pd_mask(a, b, _CMP_LE_OS);  }

static inline __mmask16 vec_cmp_gt_s (const __m512  a, const __m512  b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OS);  }
static inline __mmask8  vec_cmp_gt_d (const __m512d a, const __m512d b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OS);  }

static inline __mmask16 vec_cmp_ge_s (const __m512  a, const __m512  b) { return _mm512_cmp_ps_mask(a, b, _CMP_GE_OS);  }
static inline __mmask8  vec_cmp_ge_d (const __m512d a, const __m512d b) { return _mm512_cmp_pd_mask(a, b, _CMP_GE_OS);  }

static inline __m512  vec_blend_s(const __m512  a, const __m512  b, const __mmask16 mask) { return _mm512_mask_blend_ps(mask, a, b); }
static inline __m512d vec_blend_d(const __m512d a, const __m512d b, const __mmask8  mask) { return _mm512_mask_blend_pd(mask, a, b); }

static inline float  vec_reduce_add_s(const __m512  a) { return _mm512_reduce_add_ps(a); }
static inline double vec_reduce_add_d(const __m512d a) { return _mm512_reduce_add_pd(a); }

#ifdef __AVX512ER__
static inline __m512  vec_arsqrt_s(const __m512  r2)
{
    __m512 zero  = vec_zero_s();
    __m512 rsqrt = _mm512_rsqrt28_ps(r2);
    __mmask16 cmp0 = _mm512_cmp_ps_mask(r2, zero, _CMP_EQ_OS);
    return _mm512_mask_mov_ps(rsqrt, cmp0, zero);
}
static inline __m512d vec_arsqrt_d(const __m512d r2)
{ 
    __m512d zero  = vec_zero_d();
    __m512d rsqrt = _mm512_rsqrt28_pd(r2);
    __mmask8 cmp0 = _mm512_cmp_pd_mask(r2, zero, _CMP_EQ_OS);
    return _mm512_mask_mov_pd(rsqrt, cmp0, zero);
}
#else   // Else of "#ifdef __AVX512ER__"
static inline __m512  vec_arsqrt_s(const __m512  r2)
{
    __m512 zero  = vec_zero_s();
    __m512 rsqrt = _mm512_rsqrt14_ps(r2);
    __mmask16 cmp0 = _mm512_cmp_ps_mask(r2, zero, _CMP_EQ_OS);
    return _mm512_mask_mov_ps(rsqrt, cmp0, zero);
}
static inline __m512d vec_arsqrt_d(const __m512d r2)
{ 
    __m512d zero  = vec_zero_d();
    __m512d rsqrt = _mm512_rsqrt14_pd(r2);
    __mmask8 cmp0 = _mm512_cmp_pd_mask(r2, zero, _CMP_EQ_OS);
    return _mm512_mask_mov_pd(rsqrt, cmp0, zero);
}
#endif  // End of "#ifdef __AVX512ER__"

#ifdef __INTEL_COMPILER
static inline __m512  vec_log_s  (const __m512  a) { return _mm512_log_ps(a);   }
static inline __m512d vec_log_d  (const __m512d a) { return _mm512_log_pd(a);   }

static inline __m512  vec_log2_s (const __m512  a) { return _mm512_log2_ps(a);  }
static inline __m512d vec_log2_d (const __m512d a) { return _mm512_log2_pd(a);  }

static inline __m512  vec_log10_s(const __m512  a) { return _mm512_log10_ps(a); }
static inline __m512d vec_log10_d(const __m512d a) { return _mm512_log10_pd(a); }

static inline __m512  vec_exp_s  (const __m512  a) { return _mm512_exp_ps(a);   }
static inline __m512d vec_exp_d  (const __m512d a) { return _mm512_exp_pd(a);   }

static inline __m512  vec_exp2_s (const __m512  a) { return _mm512_exp2_ps(a);  }
static inline __m512d vec_exp2_d (const __m512d a) { return _mm512_exp2_pd(a);  }

static inline __m512  vec_exp10_s(const __m512  a) { return _mm512_exp10_ps(a); }
static inline __m512d vec_exp10_d(const __m512d a) { return _mm512_exp10_pd(a); }

static inline __m512  vec_pow_s  (const __m512  a, const __m512  b) { return _mm512_pow_ps(a, b); }
static inline __m512d vec_pow_d  (const __m512d a, const __m512d b) { return _mm512_pow_pd(a, b); }

static inline __m512  vec_sin_s  (const __m512  a) { return _mm512_sin_ps(a);   }
static inline __m512d vec_sin_d  (const __m512d a) { return _mm512_sin_pd(a);   }

static inline __m512  vec_cos_s  (const __m512  a) { return _mm512_cos_ps(a);   }
static inline __m512d vec_cos_d  (const __m512d a) { return _mm512_cos_pd(a);   }

static inline __m512  vec_erf_s  (const __m512  a) { return _mm512_erf_ps(a);   }
static inline __m512d vec_erf_d  (const __m512d a) { return _mm512_erf_pd(a);   }

#else   // Else of "#ifdef __INTEL_COMPILER"

#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22
__m512  _ZGVeN16v_logf(__m512  a);
__m512d _ZGVeN8v_log  (__m512d a);
static inline __m512  vec_log_s  (const __m512  a) { return _ZGVeN16v_logf(a);  }
static inline __m512d vec_log_d  (const __m512d a) { return _ZGVeN8v_log  (a);  }

__m512  _ZGVeN16v_expf(__m512  a);
__m512d _ZGVeN8v_exp  (__m512d a);
static inline __m512  vec_exp_s  (const __m512  a) { return _ZGVeN16v_expf(a);  }
static inline __m512d vec_exp_d  (const __m512d a) { return _ZGVeN8v_exp  (a);  }

__m512  _ZGVeN16vv_powf(__m512  a, __m512  b);
__m512d _ZGVeN8vv_pow  (__m512d a, __m512d b);
static inline __m512  vec_pow_s  (const __m512  a, const __m512  b) { return _ZGVeN16vv_powf(a, b); }
static inline __m512d vec_pow_d  (const __m512d a, const __m512d b) { return _ZGVeN8vv_pow  (a, b); }

__m512  _ZGVdN16v_sinf(__m512  a);
__m512d _ZGVdN8v_sin  (__m512d a);
static inline __m512  vec_sin_s  (const __m512  a) { return _ZGVdN16v_sinf(a);  }
static inline __m512d vec_sin_d  (const __m512d a) { return _ZGVdN8v_sin  (a);  }

__m512  _ZGVdN16v_cosf(__m512  a);
__m512d _ZGVdN8v_cos  (__m512d a);
static inline __m512  vec_cos_s  (const __m512  a) { return _ZGVdN16v_cosf(a);  }
static inline __m512d vec_cos_d  (const __m512d a) { return _ZGVdN8v_cos (a);   }

#else   // Else of "#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22"

#warning Your compiler or GLIBC does not support vectorized log(), pow(), and exp(), x86_intrin_wrapper.h will use simulated implementations. 
static inline __m512  vec_log_s(__m512  x)
{
    int i;
    union vec16f u = {x}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = logf(u.f[i]);
    return res.v;
}
static inline __m512d vec_log_d(__m512d x)
{
    int i;
    union vec8d u = {x}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = log(u.d[i]);
    return res.v;
}

static inline __m512  vec_exp_s(__m512  x)
{
    int i;
    union vec16f u = {x}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = expf(u.f[i]);
    return res.v;
}
static inline __m512d vec_exp_d(__m512d x)
{
    int i;
    union vec8d u = {x}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = exp(u.d[i]);
    return res.v;
}

static inline __m512  vec_pow_s(__m512  a, __m512  b)
{
    int i;
    union vec16f ua = {a}, ub = {b}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = powf(ua.f[i], ub.f[i]);
    return res.v;
}
static inline __m512d vec_pow_d(__m512d a, __m512d b)
{
    int i;
    union vec8d ua = {a}, ub = {b}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = pow(ua.d[i], ub.d[i]);
    return res.v;
}

static inline __m512  vec_sin_s(__m512  a)
{
    int i;
    union vec16f ua = {a}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = sinf(ua.f[i]);
    return res.v;
}
static inline __m512d vec_sin_d(__m512d a)
{
    int i;
    union vec8d ua = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = sin(ua.d[i]);
    return res.v;
}

static inline __m512  vec_cos_s(__m512  a)
{
    int i;
    union vec16f ua = {a}, ub = {b}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = cosf(ua.f[i]);
    return res.v;
}
static inline __m512d vec_cos_d(__m512d a)
{
    int i;
    union vec8d ua = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = cos(ua.d[i]);
    return res.v;
}
#endif  // End of "#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22"

static inline __m512  vec_erf_s(__m512  a)
{
    int i;
    union vec16f ua = {a}, ub = {b}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = erff(ua.f[i]);
    return res.v;
}
static inline __m512d vec_erf_d(__m512d a)
{
    int i;
    union vec8d ua = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = erf(ua.d[i]);
    return res.v;
}

#endif  // End of "#ifdef __INTEL_COMPILER"

#endif  // End of "#ifdef __AVX512F__"
#endif  // End of "#ifdef USE_AVX512"


#ifndef __INTEL_COMPILER
static inline vec_f vec_log2_s (const vec_f a) { return vec_div_s(vec_log_s(a), vec_set1_s(M_LN2));  }
static inline vec_d vec_log2_d (const vec_d a) { return vec_div_d(vec_log_d(a), vec_set1_d(M_LN2));  }

static inline vec_f vec_log10_s(const vec_f a) { return vec_div_s(vec_log_s(a), vec_set1_s(M_LN10)); }
static inline vec_d vec_log10_d(const vec_d a) { return vec_div_d(vec_log_d(a), vec_set1_d(M_LN10)); }

static inline vec_f vec_exp2_s (const vec_f a) { return vec_exp_s(vec_mul_s(a, vec_set1_s(M_LN2)));  }
static inline vec_d vec_exp2_d (const vec_d a) { return vec_exp_d(vec_mul_d(a, vec_set1_d(M_LN2)));  }

static inline vec_f vec_exp10_s(const vec_f a) { return vec_exp_s(vec_mul_s(a, vec_set1_s(M_LN10))); }
static inline vec_d vec_exp10_d(const vec_d a) { return vec_exp_d(vec_mul_d(a, vec_set1_d(M_LN10))); }
#endif  // End of #ifdef __INTEL_COMPILER

// Newton iteration step for reverse square root, rsqrt' = 0.5 * rsqrt * (C - r2 * rsqrt^2),
// 0.5 is ignored here and need to be adjusted outside. 
static inline vec_f vec_rsqrt_ntit_s(const vec_f r2, vec_f rsqrt, const float  C_)
{
    vec_f C  = vec_set1_s(C_);
    vec_f t1 = vec_mul_s(rsqrt, rsqrt);
    vec_f t2 = vec_fnmadd_s(r2, t1, C);
    return vec_mul_s(rsqrt, t2);
}
static inline vec_d vec_rsqrt_ntit_d(const vec_d r2, vec_d rsqrt, const double C_)
{
    vec_d C  = vec_set1_d(C_);
    vec_d t1 = vec_mul_d(rsqrt, rsqrt);
    vec_d t2 = vec_fnmadd_d(r2, t1, C);
    return vec_mul_d(rsqrt, t2);
}

static inline vec_f vec_frsqrt_pf_s()
{
    float newton_pf = 1.0;
    for (int i = 0; i < NEWTON_ITER; i++)
        newton_pf = 2.0 * newton_pf * newton_pf * newton_pf;
    newton_pf = 1.0 / newton_pf;
    return vec_set1_s(newton_pf);
}
static inline vec_d vec_frsqrt_pf_d()
{
    double newton_pf = 1.0;
    for (int i = 0; i < NEWTON_ITER; i++)
        newton_pf = 2.0 * newton_pf * newton_pf * newton_pf;
    newton_pf = 1.0 / newton_pf;
    return vec_set1_d(newton_pf);
}

static inline vec_f vec_frsqrt_s(const vec_f r2)
{
    vec_f rsqrt = vec_arsqrt_s(r2);
    #if NEWTON_ITER >= 1
    rsqrt = vec_rsqrt_ntit_s(r2, rsqrt, 3);
    #endif
    #if NEWTON_ITER >= 2
    rsqrt = vec_rsqrt_ntit_s(r2, rsqrt, 12);
    #endif
    #if NEWTON_ITER >= 3
    rsqrt = vec_rsqrt_ntit_s(r2, rsqrt, 768);
    #endif
    return rsqrt;
}
static inline vec_d vec_frsqrt_d(const vec_d r2)
{
    vec_d rsqrt = vec_arsqrt_d(r2);
    #if NEWTON_ITER >= 1
    rsqrt = vec_rsqrt_ntit_d(r2, rsqrt, 3);
    #endif
    #if NEWTON_ITER >= 2
    rsqrt = vec_rsqrt_ntit_d(r2, rsqrt, 12);
    #endif
    #if NEWTON_ITER >= 3
    rsqrt = vec_rsqrt_ntit_d(r2, rsqrt, 768);
    #endif
    return rsqrt;
}


#ifdef __cplusplus
}
#endif

#endif  // End of header file 
