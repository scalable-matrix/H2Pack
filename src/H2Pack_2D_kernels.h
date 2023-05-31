#ifndef __H2PACK_2D_KERNELS_H__
#define __H2PACK_2D_KERNELS_H__

#include <math.h>

#include "H2Pack_config.h"
#include "ASTER/include/aster.h"

#ifndef KRNL_EVAL_PARAM 
#define KRNL_EVAL_PARAM \
    const DTYPE *coord0, const int ld0, const int n0, \
    const DTYPE *coord1, const int ld1, const int n1, \
    const void *param, DTYPE * __restrict mat, const int ldm 
#endif

#ifndef KRNL_BIMV_PARAM
#define KRNL_BIMV_PARAM \
    const DTYPE *coord0, const int ld0, const int n0,            \
    const DTYPE *coord1, const int ld1, const int n1,            \
    const void *param, const DTYPE *x_in_0, const DTYPE *x_in_1, \
    DTYPE * __restrict x_out_0, DTYPE * __restrict x_out_1
#endif

#define EXTRACT_2D_COORD() \
    const DTYPE *x0 = coord0 + ld0 * 0; \
    const DTYPE *y0 = coord0 + ld0 * 1; \
    const DTYPE *x1 = coord1 + ld1 * 0; \
    const DTYPE *y1 = coord1 + ld1 * 1; 

// When counting bimv flops, report effective flops (1 / sqrt(x) == 2 flops)
// instead of achieved flops (1 / sqrt(x) == 1 + NEWTON_ITER * 4 flops)

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================ //
// ====================   Laplace Kernel   ==================== //
// ============================================================ //

const  int  Laplace_2D_krnl_bimv_flop = 11;

static void Laplace_2D_eval_intrin_t(KRNL_EVAL_PARAM)
{
    EXTRACT_2D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE diag_val = (param_ == NULL) ? 0.0 : param_[0];
    const vec_t v_0   = vec_zero_t();
    const vec_t v_n05 = vec_set1_t(-0.5);
    const vec_t v_dv  = vec_set1_t(diag_val);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_t x0_iv = vec_bcast_t(x0 + i);
        vec_t y0_iv = vec_bcast_t(y0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_t dx = vec_sub_t(x0_iv, vec_loadu_t(x1 + j));
            vec_t dy = vec_sub_t(y0_iv, vec_loadu_t(y1 + j));
            
            vec_t r2 = vec_mul_t(dx, dx);
            r2 = vec_fmadd_t(dy, dy, r2);
            
            vec_cmp_t r2_eq_0 = vec_cmp_eq_t(r2, v_0);
            vec_t logr = vec_mul_t(v_n05, vec_log_t(r2));
            logr = vec_blend_t(logr, v_dv, r2_eq_0);
            vec_storeu_t(mat_irow + j, logr);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE r2 = dx * dx + dy * dy;
            mat_irow[j] = (r2 == 0.0) ? diag_val : (-0.5 * DLOG(r2));
        }
    }
}

static void Laplace_2D_krnl_bimv_intrin_t(KRNL_BIMV_PARAM)
{
    EXTRACT_2D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE diag_val = (param_ == NULL) ? 0.0 : param_[0];
    const vec_t v_0   = vec_zero_t();
    const vec_t v_n05 = vec_set1_t(-0.5);
    const vec_t v_dv  = vec_set1_t(diag_val);
    for (int i = 0; i < n0; i += 2)
    {
        vec_t sum_v0 = vec_zero_t();
        vec_t sum_v1 = vec_zero_t();
        const vec_t x0_i0v = vec_bcast_t(x0 + i);
        const vec_t y0_i0v = vec_bcast_t(y0 + i);
        const vec_t x0_i1v = vec_bcast_t(x0 + i + 1);
        const vec_t y0_i1v = vec_bcast_t(y0 + i + 1);
        const vec_t x_in_1_i0v = vec_bcast_t(x_in_1 + i);
        const vec_t x_in_1_i1v = vec_bcast_t(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_t d0, d1, jv, r20, r21;
            vec_cmp_t r20_eq_0, r21_eq_0;
            
            jv  = vec_load_t(x1 + j);
            d0  = vec_sub_t(x0_i0v, jv);
            d1  = vec_sub_t(x0_i1v, jv);
            r20 = vec_mul_t(d0, d0);
            r21 = vec_mul_t(d1, d1);
            
            jv  = vec_load_t(y1 + j);
            d0  = vec_sub_t(y0_i0v, jv);
            d1  = vec_sub_t(y0_i1v, jv);
            r20 = vec_fmadd_t(d0, d0, r20);
            r21 = vec_fmadd_t(d1, d1, r21);
            
            d0 = vec_load_t(x_in_0 + j);
            d1 = vec_load_t(x_out_1 + j);
            
            r20_eq_0 = vec_cmp_eq_t(r20, v_0);
            r21_eq_0 = vec_cmp_eq_t(r21, v_0);
            r20 = vec_mul_t(v_n05, vec_log_t(r20));
            r21 = vec_mul_t(v_n05, vec_log_t(r21));
            r20 = vec_blend_t(r20, v_dv, r20_eq_0);
            r21 = vec_blend_t(r21, v_dv, r21_eq_0);
            
            sum_v0 = vec_fmadd_t(d0, r20, sum_v0);
            sum_v1 = vec_fmadd_t(d0, r21, sum_v1);
            
            d1 = vec_fmadd_t(x_in_1_i0v, r20, d1);
            d1 = vec_fmadd_t(x_in_1_i1v, r21, d1);
            vec_store_t(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_t(sum_v0);
        x_out_0[i+1] += vec_reduce_add_t(sum_v1);
    }
}

// ============================================================ //
// ====================   Gaussian Kernel   =================== //
// ============================================================ //

const  int  Gaussian_2D_krnl_bimv_flop = 11;

static void Gaussian_2D_eval_intrin_t(KRNL_EVAL_PARAM)
{
    EXTRACT_2D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE neg_l = -param_[0];
    const vec_t neg_l_v = vec_set1_t(neg_l);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_t x0_iv = vec_bcast_t(x0 + i);
        vec_t y0_iv = vec_bcast_t(y0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_t dx = vec_sub_t(x0_iv, vec_loadu_t(x1 + j));
            vec_t dy = vec_sub_t(y0_iv, vec_loadu_t(y1 + j));
            
            vec_t r2 = vec_mul_t(dx, dx);
            r2 = vec_fmadd_t(dy, dy, r2);
            
            r2 = vec_exp_t(vec_mul_t(neg_l_v, r2));
            
            vec_storeu_t(mat_irow + j, r2);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE r2 = dx * dx + dy * dy;
            mat_irow[j] = exp(neg_l * r2);
        }
    }
}

static void Gaussian_2D_krnl_bimv_intrin_t(KRNL_BIMV_PARAM)
{
    EXTRACT_2D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE neg_l = -param_[0];
    const vec_t neg_l_v = vec_set1_t(neg_l);
    for (int i = 0; i < n0; i += 2)
    {
        vec_t sum_v0 = vec_zero_t();
        vec_t sum_v1 = vec_zero_t();
        const vec_t x0_i0v = vec_bcast_t(x0 + i);
        const vec_t y0_i0v = vec_bcast_t(y0 + i);
        const vec_t x0_i1v = vec_bcast_t(x0 + i + 1);
        const vec_t y0_i1v = vec_bcast_t(y0 + i + 1);
        const vec_t x_in_1_i0v = vec_bcast_t(x_in_1 + i);
        const vec_t x_in_1_i1v = vec_bcast_t(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_t d0, d1, jv, r20, r21;
            
            jv  = vec_load_t(x1 + j);
            d0  = vec_sub_t(x0_i0v, jv);
            d1  = vec_sub_t(x0_i1v, jv);
            r20 = vec_mul_t(d0, d0);
            r21 = vec_mul_t(d1, d1);
            
            jv  = vec_load_t(y1 + j);
            d0  = vec_sub_t(y0_i0v, jv);
            d1  = vec_sub_t(y0_i1v, jv);
            r20 = vec_fmadd_t(d0, d0, r20);
            r21 = vec_fmadd_t(d1, d1, r21);
            
            d0 = vec_load_t(x_in_0 + j);
            d1 = vec_load_t(x_out_1 + j);
            
            r20 = vec_exp_t(vec_mul_t(neg_l_v, r20));
            r21 = vec_exp_t(vec_mul_t(neg_l_v, r21));
            
            sum_v0 = vec_fmadd_t(d0, r20, sum_v0);
            sum_v1 = vec_fmadd_t(d0, r21, sum_v1);
            
            d1 = vec_fmadd_t(x_in_1_i0v, r20, d1);
            d1 = vec_fmadd_t(x_in_1_i1v, r21, d1);
            vec_store_t(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_t(sum_v0);
        x_out_0[i+1] += vec_reduce_add_t(sum_v1);
    }
}

// ============================================================ //
// ==================   Exponential Kernel   ================== //
// ============================================================ //

const  int  Expon_2D_krnl_bimv_flop = 12;

static void Expon_2D_eval_intrin_t(KRNL_EVAL_PARAM)
{
    EXTRACT_2D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE neg_l = -param_[0];
    const vec_t neg_l_v = vec_set1_t(neg_l);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_t x0_iv = vec_bcast_t(x0 + i);
        vec_t y0_iv = vec_bcast_t(y0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_t dx = vec_sub_t(x0_iv, vec_loadu_t(x1 + j));
            vec_t dy = vec_sub_t(y0_iv, vec_loadu_t(y1 + j));
            
            vec_t r2 = vec_mul_t(dx, dx);
            r2 = vec_fmadd_t(dy, dy, r2);

            r2 = vec_mul_t(neg_l_v, vec_sqrt_t(r2));
            r2 = vec_exp_t(r2);
            
            vec_storeu_t(mat_irow + j, r2);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE r2 = dx * dx + dy * dy;
            mat_irow[j] = exp(neg_l * sqrt(r2));
        }
    }
}

static void Expon_2D_krnl_bimv_intrin_t(KRNL_BIMV_PARAM)
{
    EXTRACT_2D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE neg_l = -param_[0];
    const vec_t neg_l_v = vec_set1_t(neg_l);
    for (int i = 0; i < n0; i += 2)
    {
        vec_t sum_v0 = vec_zero_t();
        vec_t sum_v1 = vec_zero_t();
        const vec_t x0_i0v = vec_bcast_t(x0 + i);
        const vec_t y0_i0v = vec_bcast_t(y0 + i);
        const vec_t x0_i1v = vec_bcast_t(x0 + i + 1);
        const vec_t y0_i1v = vec_bcast_t(y0 + i + 1);
        const vec_t x_in_1_i0v = vec_bcast_t(x_in_1 + i);
        const vec_t x_in_1_i1v = vec_bcast_t(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_t d0, d1, jv, r20, r21;
            
            jv  = vec_load_t(x1 + j);
            d0  = vec_sub_t(x0_i0v, jv);
            d1  = vec_sub_t(x0_i1v, jv);
            r20 = vec_mul_t(d0, d0);
            r21 = vec_mul_t(d1, d1);
            
            jv  = vec_load_t(y1 + j);
            d0  = vec_sub_t(y0_i0v, jv);
            d1  = vec_sub_t(y0_i1v, jv);
            r20 = vec_fmadd_t(d0, d0, r20);
            r21 = vec_fmadd_t(d1, d1, r21);
            
            d0 = vec_load_t(x_in_0 + j);
            d1 = vec_load_t(x_out_1 + j);
            
            r20 = vec_mul_t(neg_l_v, vec_sqrt_t(r20));
            r21 = vec_mul_t(neg_l_v, vec_sqrt_t(r21));
            r20 = vec_exp_t(r20);
            r21 = vec_exp_t(r21);
            
            sum_v0 = vec_fmadd_t(d0, r20, sum_v0);
            sum_v1 = vec_fmadd_t(d0, r21, sum_v1);
            
            d1 = vec_fmadd_t(x_in_1_i0v, r20, d1);
            d1 = vec_fmadd_t(x_in_1_i1v, r21, d1);
            vec_store_t(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_t(sum_v0);
        x_out_0[i+1] += vec_reduce_add_t(sum_v1);
    }
}

// ============================================================ //
// ===================   Matern 3/2 Kernel   ================== //
// ============================================================ //

const  int  Matern32_2D_krnl_bimv_flop = 14;

#define NSQRT3 -1.7320508075688772

static void Matern32_2D_eval_intrin_t(KRNL_EVAL_PARAM)
{
    EXTRACT_2D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE nsqrt3_l = NSQRT3 * param_[0];
    const vec_t nsqrt3_l_v = vec_set1_t(nsqrt3_l);
    const vec_t v_1 = vec_set1_t(1.0);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_t x0_iv = vec_bcast_t(x0 + i);
        vec_t y0_iv = vec_bcast_t(y0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_t dx = vec_sub_t(x0_iv, vec_loadu_t(x1 + j));
            vec_t dy = vec_sub_t(y0_iv, vec_loadu_t(y1 + j));
            
            vec_t r = vec_mul_t(dx, dx);
            r = vec_fmadd_t(dy, dy, r);
            r = vec_sqrt_t(r);
            r = vec_mul_t(r, nsqrt3_l_v);
            r = vec_mul_t(vec_sub_t(v_1, r), vec_exp_t(r));
            
            vec_storeu_t(mat_irow + j, r);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE r  = sqrt(dx * dx + dy * dy);
            r = r * nsqrt3_l;
            r = (1.0 - r) * exp(r);
            mat_irow[j] = r;
        }
    }
}

static void Matern32_2D_krnl_bimv_intrin_t(KRNL_BIMV_PARAM)
{
    EXTRACT_2D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE nsqrt3_l = NSQRT3 * param_[0];
    const vec_t nsqrt3_l_v = vec_set1_t(nsqrt3_l);
    const vec_t v_1 = vec_set1_t(1.0);
    for (int i = 0; i < n0; i += 2)
    {
        vec_t sum_v0 = vec_zero_t();
        vec_t sum_v1 = vec_zero_t();
        const vec_t x0_i0v = vec_bcast_t(x0 + i);
        const vec_t y0_i0v = vec_bcast_t(y0 + i);
        const vec_t x0_i1v = vec_bcast_t(x0 + i + 1);
        const vec_t y0_i1v = vec_bcast_t(y0 + i + 1);
        const vec_t x_in_1_i0v = vec_bcast_t(x_in_1 + i);
        const vec_t x_in_1_i1v = vec_bcast_t(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_t d0, d1, jv, r0, r1;
            
            jv = vec_load_t(x1 + j);
            d0 = vec_sub_t(x0_i0v, jv);
            d1 = vec_sub_t(x0_i1v, jv);
            r0 = vec_mul_t(d0, d0);
            r1 = vec_mul_t(d1, d1);
            
            jv = vec_load_t(y1 + j);
            d0 = vec_sub_t(y0_i0v, jv);
            d1 = vec_sub_t(y0_i1v, jv);
            r0 = vec_fmadd_t(d0, d0, r0);
            r1 = vec_fmadd_t(d1, d1, r1);
            
            r0 = vec_sqrt_t(r0);
            r1 = vec_sqrt_t(r1);
            
            d0 = vec_load_t(x_in_0 + j);
            d1 = vec_load_t(x_out_1 + j);
            
            r0 = vec_mul_t(r0, nsqrt3_l_v);
            r1 = vec_mul_t(r1, nsqrt3_l_v);
            r0 = vec_mul_t(vec_sub_t(v_1, r0), vec_exp_t(r0));
            r1 = vec_mul_t(vec_sub_t(v_1, r1), vec_exp_t(r1));
            
            sum_v0 = vec_fmadd_t(d0, r0, sum_v0);
            sum_v1 = vec_fmadd_t(d0, r1, sum_v1);
            
            d1 = vec_fmadd_t(x_in_1_i0v, r0, d1);
            d1 = vec_fmadd_t(x_in_1_i1v, r1, d1);
            vec_store_t(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_t(sum_v0);
        x_out_0[i+1] += vec_reduce_add_t(sum_v1);
    }
}

// ============================================================ //
// ===================   Matern 5/2 Kernel   ================== //
// ============================================================ //

const  int  Matern52_2D_krnl_bimv_flop = 17;

#define NSQRT5 -2.2360679774997896
#define _1o3    0.3333333333333333

static void Matern52_2D_eval_intrin_t(KRNL_EVAL_PARAM)
{
    EXTRACT_2D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE nsqrt5_l = NSQRT5 * param_[0];
    const vec_t nsqrt5_l_v = vec_set1_t(nsqrt5_l);
    const vec_t v_1   = vec_set1_t(1.0);
    const vec_t v_1o3 = vec_set1_t(_1o3);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_t x0_iv = vec_bcast_t(x0 + i);
        vec_t y0_iv = vec_bcast_t(y0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_t dx = vec_sub_t(x0_iv, vec_loadu_t(x1 + j));
            vec_t dy = vec_sub_t(y0_iv, vec_loadu_t(y1 + j));
            
            vec_t r = vec_mul_t(dx, dx);
            r = vec_fmadd_t(dy, dy, r);
            r = vec_sqrt_t(r);

            vec_t lk  = vec_mul_t(nsqrt5_l_v, r);
            vec_t val = vec_sub_t(v_1, lk);
            vec_t lk2 = vec_mul_t(lk, lk);
            val = vec_fmadd_t(v_1o3, lk2, val);
            val = vec_mul_t(val, vec_exp_t(lk));
            
            vec_storeu_t(mat_irow + j, val);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx  = x0_i - x1[j];
            DTYPE dy  = y0_i - y1[j];
            DTYPE r   = sqrt(dx * dx + dy * dy);
            DTYPE lk  = nsqrt5_l * r;
            DTYPE val = (1.0 - lk + _1o3 * lk * lk) * exp(lk);
            mat_irow[j] = val;
        }
    }
}

static void Matern52_2D_krnl_bimv_intrin_t(KRNL_BIMV_PARAM)
{
    EXTRACT_2D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE nsqrt5_l = NSQRT5 * param_[0];
    const vec_t nsqrt5_l_v = vec_set1_t(nsqrt5_l);
    const vec_t v_1   = vec_set1_t(1.0);
    const vec_t v_1o3 = vec_set1_t(_1o3);
    for (int i = 0; i < n0; i += 2)
    {
        vec_t sum_v0 = vec_zero_t();
        vec_t sum_v1 = vec_zero_t();
        const vec_t x0_i0v = vec_bcast_t(x0 + i);
        const vec_t y0_i0v = vec_bcast_t(y0 + i);
        const vec_t x0_i1v = vec_bcast_t(x0 + i + 1);
        const vec_t y0_i1v = vec_bcast_t(y0 + i + 1);
        const vec_t x_in_1_i0v = vec_bcast_t(x_in_1 + i);
        const vec_t x_in_1_i1v = vec_bcast_t(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_t d0, d1, jv, r0, r1, lk0, lk1, lk02, lk12, val0, val1;
            
            jv = vec_load_t(x1 + j);
            d0 = vec_sub_t(x0_i0v, jv);
            d1 = vec_sub_t(x0_i1v, jv);
            r0 = vec_mul_t(d0, d0);
            r1 = vec_mul_t(d1, d1);
            
            jv = vec_load_t(y1 + j);
            d0 = vec_sub_t(y0_i0v, jv);
            d1 = vec_sub_t(y0_i1v, jv);
            r0 = vec_fmadd_t(d0, d0, r0);
            r1 = vec_fmadd_t(d1, d1, r1);
            
            r0 = vec_sqrt_t(r0);
            r1 = vec_sqrt_t(r1);
            
            d0 = vec_load_t(x_in_0 + j);
            d1 = vec_load_t(x_out_1 + j);
            
            lk0  = vec_mul_t(nsqrt5_l_v, r0);
            val0 = vec_sub_t(v_1, lk0);
            lk02 = vec_mul_t(lk0, lk0);
            val0 = vec_fmadd_t(v_1o3, lk02, val0);
            val0 = vec_mul_t(val0, vec_exp_t(lk0));

            lk1  = vec_mul_t(nsqrt5_l_v, r1);
            val1 = vec_sub_t(v_1, lk1);
            lk12 = vec_mul_t(lk1, lk1);
            val1 = vec_fmadd_t(v_1o3, lk12, val1);
            val1 = vec_mul_t(val1, vec_exp_t(lk1));
            
            sum_v0 = vec_fmadd_t(d0, val0, sum_v0);
            sum_v1 = vec_fmadd_t(d0, val1, sum_v1);
            
            d1 = vec_fmadd_t(x_in_1_i0v, val0, d1);
            d1 = vec_fmadd_t(x_in_1_i1v, val1, d1);
            vec_store_t(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_t(sum_v0);
        x_out_0[i+1] += vec_reduce_add_t(sum_v1);
    }
}


// ============================================================ //
// ===================   Quadratic Kernel   =================== //
// ============================================================ //

const  int  Quadratic_2D_krnl_bimv_flop = 12;

static void Quadratic_2D_eval_intrin_t(KRNL_EVAL_PARAM)
{
    EXTRACT_2D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE c = param_[0];
    const DTYPE a = param_[1];
    const vec_t vec_c = vec_set1_t(c);
    const vec_t vec_a = vec_set1_t(a);
    const vec_t vec_1 = vec_set1_t(1.0);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_t x0_iv = vec_bcast_t(x0 + i);
        vec_t y0_iv = vec_bcast_t(y0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_t dx = vec_sub_t(x0_iv, vec_loadu_t(x1 + j));
            vec_t dy = vec_sub_t(y0_iv, vec_loadu_t(y1 + j));
            
            vec_t r2 = vec_mul_t(dx, dx);
            r2 = vec_fmadd_t(dy, dy, r2);
            
            r2 = vec_fmadd_t(r2, vec_c, vec_1);
            r2 = vec_pow_t(r2, vec_a);
            
            vec_storeu_t(mat_irow + j, r2);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE r2 = dx * dx + dy * dy;

            r2 = 1.0 + c * r2;
            r2 = DPOW(r2, a);
            mat_irow[j] = r2;
        }
    }
}

static void Quadratic_2D_krnl_bimv_intrin_t(KRNL_BIMV_PARAM)
{
    EXTRACT_2D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const vec_t vec_c = vec_bcast_t(param_ + 0);
    const vec_t vec_a = vec_bcast_t(param_ + 1);
    const vec_t vec_1 = vec_set1_t(1.0);
    for (int i = 0; i < n0; i += 2)
    {
        vec_t sum_v0 = vec_zero_t();
        vec_t sum_v1 = vec_zero_t();
        const vec_t x0_i0v = vec_bcast_t(x0 + i);
        const vec_t y0_i0v = vec_bcast_t(y0 + i);
        const vec_t x0_i1v = vec_bcast_t(x0 + i + 1);
        const vec_t y0_i1v = vec_bcast_t(y0 + i + 1);
        const vec_t x_in_1_i0v = vec_bcast_t(x_in_1 + i);
        const vec_t x_in_1_i1v = vec_bcast_t(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_t d0, d1, jv, r20, r21;
            
            jv  = vec_load_t(x1 + j);
            d0  = vec_sub_t(x0_i0v, jv);
            d1  = vec_sub_t(x0_i1v, jv);
            r20 = vec_mul_t(d0, d0);
            r21 = vec_mul_t(d1, d1);
            
            jv  = vec_load_t(y1 + j);
            d0  = vec_sub_t(y0_i0v, jv);
            d1  = vec_sub_t(y0_i1v, jv);
            r20 = vec_fmadd_t(d0, d0, r20);
            r21 = vec_fmadd_t(d1, d1, r21);
            
            d0 = vec_load_t(x_in_0 + j);
            d1 = vec_load_t(x_out_1 + j);
            
            r20 = vec_fmadd_t(r20, vec_c, vec_1);
            r21 = vec_fmadd_t(r21, vec_c, vec_1);

            r20 = vec_pow_t(r20, vec_a);
            r21 = vec_pow_t(r21, vec_a);
            
            sum_v0 = vec_fmadd_t(d0, r20, sum_v0);
            sum_v1 = vec_fmadd_t(d0, r21, sum_v1);
            
            d1 = vec_fmadd_t(x_in_1_i0v, r20, d1);
            d1 = vec_fmadd_t(x_in_1_i1v, r21, d1);
            vec_store_t(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_t(sum_v0);
        x_out_0[i+1] += vec_reduce_add_t(sum_v1);
    }
}

#ifdef __cplusplus
}
#endif

#endif
