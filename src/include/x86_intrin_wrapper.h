#ifndef __X86_INTRIN_WRAPPER_H__
#define __X86_INTRIN_WRAPPER_H__

#include <x86intrin.h>

/*
    Naming: vec_<operation>_<s/d>, suffix s is for float, d is for double 
    vec_zero_*       : Set an intrinsic vector to zero
    vec_set1_*       : Set all lanes of an intrinsic vector to a value
    vec_bcast_*      : Set all lanes of an intrinsic vector to the 1st value in an address
    vec_load_*       : Load an intrinsic vector from an address aligned to required bits
    vec_loadu_*      : Load an intrinsic vector from an address which may not aligned to required bits
    vec_store_*      : Store an intrinsic vector to an address aligned to required bits
    vec_storeu_*     : Store an intrinsic vector to an address which may not aligned to required bits
    vec_add_*        : Add two intrinsic vectors a + b
    vec_sub_*        : Subtract intrinsic vector a by b
    vec_mul_*        : Multiply two intrinsic vectors a * b
    vec_div_*        : Divide intrinsic vector a by b
    vec_sqrt_*       : Return the square root of an intrinsic vector's each lane
    vec_fmadd_*      : Fused Multiply-Add intrinsic vectors a * b + c
    vec_fnmadd_*     : Fused negative Multiply-Add intrinsic vectors -(a * b) + c
    vec_fmsub_*      : Fused Multiply-Sub intrinsic vectors a * b - c
    vec_max_*        : Return the maximum values of each lane in two intrinsic vectors 
    vec_min_*        : Return the minimum values of each lane in two intrinsic vectors   
    vec_cmp_eq_*     : Return in each lane if a == b
    vec_cmp_neq_*    : Return in each lane if a != b
    vec_cmp_lt_*     : Return in each lane if a <  b
    vec_cmp_le_*     : Return in each lane if a <= b
    vec_cmp_gt_*     : Return in each lane if a >  b
    vec_cmp_ge_*     : Return in each lane if a >= b
    vec_blend_*      : Blend elements from two intrinsic vectors a, b using mask, mask == 1 --> use b, else use a
    vec_reduce_add_* : Return the sum of values in an intrinsic vector
    vec_arsqrt_*     : Approximate reverse squart root, returns 0 if r2 == 0
    vec_rsqrt_ntit_* : Newton iteration step for reverse squart root,
                       rsqrt' = 0.5 * rsqrt * (C - r2 * rsqrt^2),
                       0.5 is ignored here and need to be adjusted outside. 
    vec_frsqrt_pf_*  : Return the scaling prefactor for vec_frsqrt_*()
    vec_frsqrt_*     : Fast reverse squart root using Newton iteration, 
                       returns 0 if r2 == 0, otherwise result need to be 
                       multipled with vec_ntfrac_*() to get the correct one.
    vec_log_*        : Return the natural logarithm of each lane 
    vec_log2_*       : Return the base-2  logarithm of each lane 
    vec_log10_*      : Return the base-10 logarithm of each lane 
    vec_exp_*        : Return the exponential value of e  raised to the power of values in each lane
    vec_exp2_*       : Return the exponential value of 2  raised to the power of values in each lane
    vec_exp10_*      : Return the exponential value of 10 raised to the power of values in each lane

    Reference:
    1. Intel Intrinsic Guide: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
    2. Compiler Explorer: https://godbolt.org/
    3. For AVX vec_reduce_add_s: https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
    4. For AVX vec_reduce_add_d: https://www.oipapio.com/question-771803
    5. For fast inverse square root: https://en.wikipedia.org/wiki/Fast_inverse_square_root
*/ 

#ifndef NEWTON_ITER
#define NEWTON_ITER 2   // Two Newton iterations is usually sufficient for rsqrt using double type
#endif

#ifdef USE_AVX
#ifdef __AVX__

#define SIMD_LEN_S  8
#define SIMD_LEN_D  4
#define vec_f       __m256
#define vec_d       __m256d
#define vec_cmp_f   __m256
#define vec_cmp_d   __m256d

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

static inline __m256  vec_sqrt_s(const __m256  a) { return _mm256_sqrt_ps(a); }
static inline __m256d vec_sqrt_d(const __m256d a) { return _mm256_sqrt_pd(a); }

#ifdef __AVX2__
static inline __m256  vec_fmadd_s(const __m256  a, const __m256  b, const __m256  c)  { return _mm256_fmadd_ps(a, b, c);  }
static inline __m256d vec_fmadd_d(const __m256d a, const __m256d b, const __m256d c)  { return _mm256_fmadd_pd(a, b, c);  }

static inline __m256  vec_fnmadd_s(const __m256  a, const __m256  b, const __m256  c) { return _mm256_fnmadd_ps(a, b, c); }
static inline __m256d vec_fnmadd_d(const __m256d a, const __m256d b, const __m256d c) { return _mm256_fnmadd_pd(a, b, c); }

static inline __m256  vec_fmsub_s(const __m256  a, const __m256  b, const __m256  c)  { return _mm256_fmsub_ps(a, b, c);  }
static inline __m256d vec_fmsub_d(const __m256d a, const __m256d b, const __m256d c)  { return _mm256_fmsub_pd(a, b, c);  }
#else
static inline __m256  vec_fmadd_s(const __m256  a, const __m256  b, const __m256  c)  { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
static inline __m256d vec_fmadd_d(const __m256d a, const __m256d b, const __m256d c)  { return _mm256_add_pd(_mm256_mul_pd(a, b), c); }

static inline __m256  vec_fnmadd_s(const __m256  a, const __m256  b, const __m256  c) { return _mm256_sub_ps(c, _mm256_mul_ps(a, b)); }
static inline __m256d vec_fnmadd_d(const __m256d a, const __m256d b, const __m256d c) { return _mm256_sub_pd(c, _mm256_mul_pd(a, b)); }

static inline __m256  vec_fmsub_s(const __m256  a, const __m256  b, const __m256  c)  { return _mm256_sub_ps(_mm256_mul_ps(a, b), c); }
static inline __m256d vec_fmsub_d(const __m256d a, const __m256d b, const __m256d c)  { return _mm256_sub_pd(_mm256_mul_pd(a, b), c); }
#endif // End of #ifdef __AVX2__

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

#endif  // End of #ifdef __AVX__
#endif  // End of #ifdef USE_AVX


#ifdef USE_AVX512
#ifdef __AVX512F__

#define SIMD_LEN_S  16
#define SIMD_LEN_D  8
#define vec_f       __m512
#define vec_d       __m512d
#define vec_cmp_f   __mmask16
#define vec_cmp_d   __mmask8

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

static inline __m512  vec_sqrt_s(const __m512  a) { return _mm512_sqrt_ps(a); }
static inline __m512d vec_sqrt_d(const __m512d a) { return _mm512_sqrt_pd(a); }

static inline __m512  vec_fmadd_s(const __m512  a, const __m512  b, const __m512  c)  { return _mm512_fmadd_ps(a, b, c);  }
static inline __m512d vec_fmadd_d(const __m512d a, const __m512d b, const __m512d c)  { return _mm512_fmadd_pd(a, b, c);  }

static inline __m512  vec_fnmadd_s(const __m512  a, const __m512  b, const __m512  c) { return _mm512_fnmadd_ps(a, b, c); }
static inline __m512d vec_fnmadd_d(const __m512d a, const __m512d b, const __m512d c) { return _mm512_fnmadd_pd(a, b, c); }

static inline __m512  vec_fmsub_s(const __m512  a, const __m512  b, const __m512  c)  { return _mm512_fmsub_ps(a, b, c);  }
static inline __m512d vec_fmsub_d(const __m512d a, const __m512d b, const __m512d c)  { return _mm512_fmsub_pd(a, b, c);  }

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
#else
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
#endif  // End of #ifdef __AVX512ER__

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

#endif  // End of #ifdef __AVX512F__
#endif  // End of #ifdef USE_AVX512


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

#endif  // End of header file 
