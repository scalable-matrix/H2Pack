#ifndef __H2PACK_3D_KERNELS_H__
#define __H2PACK_3D_KERNELS_H__

#include <math.h>

#include "H2Pack_config.h"
#include "x86_intrin_wrapper.h" 

#ifndef KRNL_EVAL_PARAM 
#define KRNL_EVAL_PARAM \
    const DTYPE *coord0, const int ld0, const int n0, \
    const DTYPE *coord1, const int ld1, const int n1, \
    const void *param, DTYPE *restrict mat, const int ldm 
#endif

#ifndef KRNL_BIMV_PARAM
#define KRNL_BIMV_PARAM \
    const DTYPE *coord0, const int ld0, const int n0,            \
    const DTYPE *coord1, const int ld1, const int n1,            \
    const void *param, const DTYPE *x_in_0, const DTYPE *x_in_1, \
    DTYPE *restrict x_out_0, DTYPE *restrict x_out_1
#endif

#define EXTRACT_3D_COORD() \
    const DTYPE *x0 = coord0 + ld0 * 0; \
    const DTYPE *y0 = coord0 + ld0 * 1; \
    const DTYPE *z0 = coord0 + ld0 * 2; \
    const DTYPE *x1 = coord1 + ld1 * 0; \
    const DTYPE *y1 = coord1 + ld1 * 1; \
    const DTYPE *z1 = coord1 + ld1 * 2; 

// When counting bimv flops, report effective flops (1 / sqrt(x) == 2 flops)
// instead of achieved flops (1 / sqrt(x) == 1 + NEWTON_ITER * 4 flops)

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================ //
// ====================   Coulomb Kernel   ==================== //
// ============================================================ //

const  int  Coulomb_3D_krnl_bimv_flop = 14;

static void Coulomb_3D_eval_intrin_d(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const vec_d frsqrt_pf = vec_frsqrt_pf_d();
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_d x0_iv = vec_bcast_d(x0 + i);
        vec_d y0_iv = vec_bcast_d(y0 + i);
        vec_d z0_iv = vec_bcast_d(z0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_d dx = vec_sub_d(x0_iv, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(y0_iv, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(z0_iv, vec_loadu_d(z1 + j));
            
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            
            vec_d rinv = vec_mul_d(frsqrt_pf, vec_frsqrt_d(r2));
            vec_storeu_d(mat_irow + j, rinv);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        const DTYPE z0_i = z0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            mat_irow[j] = (r2 == 0.0) ? 0.0 : (1.0 / DSQRT(r2));
        }
    }
}

static void Coulomb_3D_krnl_bimv_intrin_d(KRNL_BIMV_PARAM)
{
    EXTRACT_3D_COORD();
    const vec_d frsqrt_pf = vec_frsqrt_pf_d();
    for (int i = 0; i < n0; i += 2)
    {
        vec_d sum_v0 = vec_zero_d();
        vec_d sum_v1 = vec_zero_d();
        const vec_d x0_i0v = vec_bcast_d(x0 + i);
        const vec_d y0_i0v = vec_bcast_d(y0 + i);
        const vec_d z0_i0v = vec_bcast_d(z0 + i);
        const vec_d x0_i1v = vec_bcast_d(x0 + i + 1);
        const vec_d y0_i1v = vec_bcast_d(y0 + i + 1);
        const vec_d z0_i1v = vec_bcast_d(z0 + i + 1);
        const vec_d x_in_1_i0v = vec_bcast_d(x_in_1 + i);
        const vec_d x_in_1_i1v = vec_bcast_d(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_d d0, d1, jv, r20, r21;
            
            jv  = vec_load_d(x1 + j);
            d0  = vec_sub_d(x0_i0v, jv);
            d1  = vec_sub_d(x0_i1v, jv);
            r20 = vec_mul_d(d0, d0);
            r21 = vec_mul_d(d1, d1);
            
            jv  = vec_load_d(y1 + j);
            d0  = vec_sub_d(y0_i0v, jv);
            d1  = vec_sub_d(y0_i1v, jv);
            r20 = vec_fmadd_d(d0, d0, r20);
            r21 = vec_fmadd_d(d1, d1, r21);
            
            jv  = vec_load_d(z1 + j);
            d0  = vec_sub_d(z0_i0v, jv);
            d1  = vec_sub_d(z0_i1v, jv);
            r20 = vec_fmadd_d(d0, d0, r20);
            r21 = vec_fmadd_d(d1, d1, r21);
            
            d0 = vec_load_d(x_in_0 + j);
            d1 = vec_load_d(x_out_1 + j);
            
            r20 = vec_mul_d(frsqrt_pf, vec_frsqrt_d(r20));
            r21 = vec_mul_d(frsqrt_pf, vec_frsqrt_d(r21));
            
            sum_v0 = vec_fmadd_d(d0, r20, sum_v0);
            sum_v1 = vec_fmadd_d(d0, r21, sum_v1);
            
            d1 = vec_fmadd_d(x_in_1_i0v, r20, d1);
            d1 = vec_fmadd_d(x_in_1_i1v, r21, d1);
            vec_store_d(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_d(sum_v0);
        x_out_0[i+1] += vec_reduce_add_d(sum_v1);
    }
}

// ============================================================ //
// ==================   Erf Coulomb Kernel   ================== //
// ============================================================ //

const  int  ECoulomb_3D_krnl_bimv_flop = 18;

static void ECoulomb_3D_eval_intrin_d(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const vec_d frsqrt_pf = vec_frsqrt_pf_d();
    const vec_d v_05 = vec_set1_d(0.5);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_d x0_iv = vec_bcast_d(x0 + i);
        vec_d y0_iv = vec_bcast_d(y0 + i);
        vec_d z0_iv = vec_bcast_d(z0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_d dx = vec_sub_d(x0_iv, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(y0_iv, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(z0_iv, vec_loadu_d(z1 + j));
            
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            vec_d semi_r = vec_mul_d(v_05, vec_sqrt_d(r2));
            
            vec_d rinv = vec_mul_d(frsqrt_pf, vec_frsqrt_d(r2));
            vec_d val  = vec_mul_d(rinv, vec_erf_d(semi_r));
            vec_storeu_d(mat_irow + j, val);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        const DTYPE z0_i = z0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            DTYPE r  = DSQRT(r2);
            DTYPE invr = (r2 == 0.0) ? 0.0 : (1.0 / r);
            mat_irow[j] = invr * DERF(0.5 * r);
        }
    }
}

static void ECoulomb_3D_krnl_bimv_intrin_d(KRNL_BIMV_PARAM)
{
    EXTRACT_3D_COORD();
    const vec_d frsqrt_pf = vec_frsqrt_pf_d();
    const vec_d v_05 = vec_set1_d(0.5);
    for (int i = 0; i < n0; i += 2)
    {
        vec_d sum_v0 = vec_zero_d();
        vec_d sum_v1 = vec_zero_d();
        const vec_d x0_i0v = vec_bcast_d(x0 + i);
        const vec_d y0_i0v = vec_bcast_d(y0 + i);
        const vec_d z0_i0v = vec_bcast_d(z0 + i);
        const vec_d x0_i1v = vec_bcast_d(x0 + i + 1);
        const vec_d y0_i1v = vec_bcast_d(y0 + i + 1);
        const vec_d z0_i1v = vec_bcast_d(z0 + i + 1);
        const vec_d x_in_1_i0v = vec_bcast_d(x_in_1 + i);
        const vec_d x_in_1_i1v = vec_bcast_d(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_d d0, d1, jv, r20, r21, semi_r0, semi_r1, val0, val1;
            
            jv  = vec_load_d(x1 + j);
            d0  = vec_sub_d(x0_i0v, jv);
            d1  = vec_sub_d(x0_i1v, jv);
            r20 = vec_mul_d(d0, d0);
            r21 = vec_mul_d(d1, d1);
            
            jv  = vec_load_d(y1 + j);
            d0  = vec_sub_d(y0_i0v, jv);
            d1  = vec_sub_d(y0_i1v, jv);
            r20 = vec_fmadd_d(d0, d0, r20);
            r21 = vec_fmadd_d(d1, d1, r21);
            
            jv  = vec_load_d(z1 + j);
            d0  = vec_sub_d(z0_i0v, jv);
            d1  = vec_sub_d(z0_i1v, jv);
            r20 = vec_fmadd_d(d0, d0, r20);
            r21 = vec_fmadd_d(d1, d1, r21);
            
            semi_r0 = vec_mul_d(v_05, vec_sqrt_d(r20));
            semi_r1 = vec_mul_d(v_05, vec_sqrt_d(r21));

            d0 = vec_load_d(x_in_0 + j);
            d1 = vec_load_d(x_out_1 + j);
            
            r20 = vec_mul_d(frsqrt_pf, vec_frsqrt_d(r20));
            r21 = vec_mul_d(frsqrt_pf, vec_frsqrt_d(r21));
            
            val0 = vec_mul_d(r20, vec_erf_d(semi_r0));
            val1 = vec_mul_d(r21, vec_erf_d(semi_r1));

            sum_v0 = vec_fmadd_d(d0, val0, sum_v0);
            sum_v1 = vec_fmadd_d(d0, val1, sum_v1);
            
            d1 = vec_fmadd_d(x_in_1_i0v, val0, d1);
            d1 = vec_fmadd_d(x_in_1_i1v, val1, d1);
            vec_store_d(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_d(sum_v0);
        x_out_0[i+1] += vec_reduce_add_d(sum_v1);
    }
}

// ============================================================ //
// ====================   Gaussian Kernel   =================== //
// ============================================================ //

const  int  Gaussian_3D_krnl_bimv_flop = 14;

static void Gaussian_3D_eval_intrin_d(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE l = param_[0];
    const vec_d l_v = vec_set1_d(l);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_d x0_iv = vec_bcast_d(x0 + i);
        vec_d y0_iv = vec_bcast_d(y0 + i);
        vec_d z0_iv = vec_bcast_d(z0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_d dx = vec_sub_d(x0_iv, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(y0_iv, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(z0_iv, vec_loadu_d(z1 + j));
            
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            
            r2 = vec_fnmadd_d(r2, l_v, vec_zero_d());
            r2 = vec_exp_d(r2);
            
            vec_storeu_d(mat_irow + j, r2);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        const DTYPE z0_i = z0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            mat_irow[j] = exp(-l * r2);
        }
    }
}

static void Gaussian_3D_krnl_bimv_intrin_d(KRNL_BIMV_PARAM)
{
    EXTRACT_3D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE l = param_[0];
    const vec_d l_v = vec_set1_d(l);
    for (int i = 0; i < n0; i += 2)
    {
        vec_d sum_v0 = vec_zero_d();
        vec_d sum_v1 = vec_zero_d();
        const vec_d x0_i0v = vec_bcast_d(x0 + i);
        const vec_d y0_i0v = vec_bcast_d(y0 + i);
        const vec_d z0_i0v = vec_bcast_d(z0 + i);
        const vec_d x0_i1v = vec_bcast_d(x0 + i + 1);
        const vec_d y0_i1v = vec_bcast_d(y0 + i + 1);
        const vec_d z0_i1v = vec_bcast_d(z0 + i + 1);
        const vec_d x_in_1_i0v = vec_bcast_d(x_in_1 + i);
        const vec_d x_in_1_i1v = vec_bcast_d(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_d d0, d1, jv, r20, r21;
            
            jv  = vec_load_d(x1 + j);
            d0  = vec_sub_d(x0_i0v, jv);
            d1  = vec_sub_d(x0_i1v, jv);
            r20 = vec_mul_d(d0, d0);
            r21 = vec_mul_d(d1, d1);
            
            jv  = vec_load_d(y1 + j);
            d0  = vec_sub_d(y0_i0v, jv);
            d1  = vec_sub_d(y0_i1v, jv);
            r20 = vec_fmadd_d(d0, d0, r20);
            r21 = vec_fmadd_d(d1, d1, r21);
            
            jv  = vec_load_d(z1 + j);
            d0  = vec_sub_d(z0_i0v, jv);
            d1  = vec_sub_d(z0_i1v, jv);
            r20 = vec_fmadd_d(d0, d0, r20);
            r21 = vec_fmadd_d(d1, d1, r21);
            
            d0 = vec_load_d(x_in_0 + j);
            d1 = vec_load_d(x_out_1 + j);
            
            r20 = vec_fnmadd_d(r20, l_v, vec_zero_d());
            r21 = vec_fnmadd_d(r21, l_v, vec_zero_d());
            r20 = vec_exp_d(r20);
            r21 = vec_exp_d(r21);
            
            sum_v0 = vec_fmadd_d(d0, r20, sum_v0);
            sum_v1 = vec_fmadd_d(d0, r21, sum_v1);
            
            d1 = vec_fmadd_d(x_in_1_i0v, r20, d1);
            d1 = vec_fmadd_d(x_in_1_i1v, r21, d1);
            vec_store_d(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_d(sum_v0);
        x_out_0[i+1] += vec_reduce_add_d(sum_v1);
    }
}

// ============================================================ //
// =====================   Matern Kernel   ==================== //
// ============================================================ //

const  int  Matern_3D_krnl_bimv_flop = 17;

#define NSQRT3 -1.7320508075688772

static void Matern_3D_eval_intrin_d(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE nsqrt3_l = NSQRT3 * param_[0];
    const vec_d nsqrt3_l_v = vec_set1_d(nsqrt3_l);
    const vec_d v_1 = vec_set1_d(1.0);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_d x0_iv = vec_bcast_d(x0 + i);
        vec_d y0_iv = vec_bcast_d(y0 + i);
        vec_d z0_iv = vec_bcast_d(z0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_d dx = vec_sub_d(x0_iv, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(y0_iv, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(z0_iv, vec_loadu_d(z1 + j));
            
            vec_d r = vec_mul_d(dx, dx);
            r = vec_fmadd_d(dy, dy, r);
            r = vec_fmadd_d(dz, dz, r);
            r = vec_sqrt_d(r);
            r = vec_mul_d(r, nsqrt3_l_v);
            r = vec_mul_d(vec_sub_d(v_1, r), vec_exp_d(r));
            
            vec_storeu_d(mat_irow + j, r);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        const DTYPE z0_i = z0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            DTYPE r  = sqrt(dx * dx + dy * dy + dz * dz);
            r = r * nsqrt3_l;
            r = (1.0 - r) * exp(r);
            mat_irow[j] = r;
        }
    }
}

static void Matern_3D_krnl_bimv_intrin_d(KRNL_BIMV_PARAM)
{
    EXTRACT_3D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE nsqrt3_l = NSQRT3 * param_[0];
    const vec_d nsqrt3_l_v = vec_set1_d(nsqrt3_l);
    const vec_d v_1 = vec_set1_d(1.0);
    for (int i = 0; i < n0; i += 2)
    {
        vec_d sum_v0 = vec_zero_d();
        vec_d sum_v1 = vec_zero_d();
        const vec_d x0_i0v = vec_bcast_d(x0 + i);
        const vec_d y0_i0v = vec_bcast_d(y0 + i);
        const vec_d z0_i0v = vec_bcast_d(z0 + i);
        const vec_d x0_i1v = vec_bcast_d(x0 + i + 1);
        const vec_d y0_i1v = vec_bcast_d(y0 + i + 1);
        const vec_d z0_i1v = vec_bcast_d(z0 + i + 1);
        const vec_d x_in_1_i0v = vec_bcast_d(x_in_1 + i);
        const vec_d x_in_1_i1v = vec_bcast_d(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_d d0, d1, jv, r0, r1;
            
            jv = vec_load_d(x1 + j);
            d0 = vec_sub_d(x0_i0v, jv);
            d1 = vec_sub_d(x0_i1v, jv);
            r0 = vec_mul_d(d0, d0);
            r1 = vec_mul_d(d1, d1);
            
            jv = vec_load_d(y1 + j);
            d0 = vec_sub_d(y0_i0v, jv);
            d1 = vec_sub_d(y0_i1v, jv);
            r0 = vec_fmadd_d(d0, d0, r0);
            r1 = vec_fmadd_d(d1, d1, r1);
            
            jv = vec_load_d(z1 + j);
            d0 = vec_sub_d(z0_i0v, jv);
            d1 = vec_sub_d(z0_i1v, jv);
            r0 = vec_fmadd_d(d0, d0, r0);
            r1 = vec_fmadd_d(d1, d1, r1);
            
            r0 = vec_sqrt_d(r0);
            r1 = vec_sqrt_d(r1);
            
            d0 = vec_load_d(x_in_0 + j);
            d1 = vec_load_d(x_out_1 + j);
            
            r0 = vec_mul_d(r0, nsqrt3_l_v);
            r1 = vec_mul_d(r1, nsqrt3_l_v);
            r0 = vec_mul_d(vec_sub_d(v_1, r0), vec_exp_d(r0));
            r1 = vec_mul_d(vec_sub_d(v_1, r1), vec_exp_d(r1));
            
            sum_v0 = vec_fmadd_d(d0, r0, sum_v0);
            sum_v1 = vec_fmadd_d(d0, r1, sum_v1);
            
            d1 = vec_fmadd_d(x_in_1_i0v, r0, d1);
            d1 = vec_fmadd_d(x_in_1_i1v, r1, d1);
            vec_store_d(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_d(sum_v0);
        x_out_0[i+1] += vec_reduce_add_d(sum_v1);
    }
}

// ============================================================ //
// ===================   Quadratic Kernel   =================== //
// ============================================================ //

const  int  Quadratic_3D_krnl_bimv_flop = 15;

static void Quadratic_3D_eval_intrin_d(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE c = param_[0];
    const DTYPE a = param_[1];
    const vec_d vec_c = vec_set1_d(c);
    const vec_d vec_a = vec_set1_d(a);
    const vec_d vec_1 = vec_set1_d(1.0);
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        
        vec_d x0_iv = vec_bcast_d(x0 + i);
        vec_d y0_iv = vec_bcast_d(y0 + i);
        vec_d z0_iv = vec_bcast_d(z0 + i);
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_d dx = vec_sub_d(x0_iv, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(y0_iv, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(z0_iv, vec_loadu_d(z1 + j));
            
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            
            r2 = vec_fmadd_d(r2, vec_c, vec_1);
            r2 = vec_pow_d(r2, vec_a);
            
            vec_storeu_d(mat_irow + j, r2);
        }
        
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        const DTYPE z0_i = z0[i];
        for (int j = n1_vec; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;

            r2 = 1.0 + c * r2;
            r2 = DPOW(r2, a);
            mat_irow[j] = r2;
        }
    }
}

static void Quadratic_3D_krnl_bimv_intrin_d(KRNL_BIMV_PARAM)
{
    EXTRACT_3D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const vec_d vec_c = vec_bcast_d(param_ + 0);
    const vec_d vec_a = vec_bcast_d(param_ + 1);
    const vec_d vec_1 = vec_set1_d(1.0);
    for (int i = 0; i < n0; i += 2)
    {
        vec_d sum_v0 = vec_zero_d();
        vec_d sum_v1 = vec_zero_d();
        const vec_d x0_i0v = vec_bcast_d(x0 + i);
        const vec_d y0_i0v = vec_bcast_d(y0 + i);
        const vec_d z0_i0v = vec_bcast_d(z0 + i);
        const vec_d x0_i1v = vec_bcast_d(x0 + i + 1);
        const vec_d y0_i1v = vec_bcast_d(y0 + i + 1);
        const vec_d z0_i1v = vec_bcast_d(z0 + i + 1);
        const vec_d x_in_1_i0v = vec_bcast_d(x_in_1 + i);
        const vec_d x_in_1_i1v = vec_bcast_d(x_in_1 + i + 1);
        for (int j = 0; j < n1; j += SIMD_LEN)
        {
            vec_d d0, d1, jv, r20, r21;
            
            jv  = vec_load_d(x1 + j);
            d0  = vec_sub_d(x0_i0v, jv);
            d1  = vec_sub_d(x0_i1v, jv);
            r20 = vec_mul_d(d0, d0);
            r21 = vec_mul_d(d1, d1);
            
            jv  = vec_load_d(y1 + j);
            d0  = vec_sub_d(y0_i0v, jv);
            d1  = vec_sub_d(y0_i1v, jv);
            r20 = vec_fmadd_d(d0, d0, r20);
            r21 = vec_fmadd_d(d1, d1, r21);
            
            jv  = vec_load_d(z1 + j);
            d0  = vec_sub_d(z0_i0v, jv);
            d1  = vec_sub_d(z0_i1v, jv);
            r20 = vec_fmadd_d(d0, d0, r20);
            r21 = vec_fmadd_d(d1, d1, r21);
            
            d0 = vec_load_d(x_in_0 + j);
            d1 = vec_load_d(x_out_1 + j);
            
            r20 = vec_fmadd_d(r20, vec_c, vec_1);
            r21 = vec_fmadd_d(r21, vec_c, vec_1);

            r20 = vec_pow_d(r20, vec_a);
            r21 = vec_pow_d(r21, vec_a);
            
            sum_v0 = vec_fmadd_d(d0, r20, sum_v0);
            sum_v1 = vec_fmadd_d(d0, r21, sum_v1);
            
            d1 = vec_fmadd_d(x_in_1_i0v, r20, d1);
            d1 = vec_fmadd_d(x_in_1_i1v, r21, d1);
            vec_store_d(x_out_1 + j, d1);
        }
        x_out_0[i]   += vec_reduce_add_d(sum_v0);
        x_out_0[i+1] += vec_reduce_add_d(sum_v1);
    }
}

// ============================================================ //
// =====================   Stokes Kernel   ==================== //
// ============================================================ //

#define CALC_STOKES_CONST() \
    const DTYPE *param_ = (DTYPE*) param;           \
    const DTYPE eta = param_[0];                    \
    const DTYPE a   = param_[1];                    \
    const DTYPE C   = 1.0 / (6.0 * M_PI * a * eta); \
    const DTYPE Ca3o4 = C * a * 0.75;               

const  int  Stokes_krnl_bimv_flop = 48;

static void Stokes_eval_std(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    CALC_STOKES_CONST();
    for (int i = 0; i < n0; i++)
    {
        DTYPE txs = x0[i];
        DTYPE tys = y0[i];
        DTYPE tzs = z0[i];
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = txs - x1[j];
            DTYPE dy = tys - y1[j];
            DTYPE dz = tzs - z1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            
            DTYPE inv_r, t1;
            if (r2 == 0.0)
            {
                inv_r = 0.0;
                t1 = C;
            } else {
                inv_r = 1.0 / sqrt(r2);
                t1 = inv_r * Ca3o4;
            }
            
            dx *= inv_r;
            dy *= inv_r;
            dz *= inv_r;
            
            int base = 3 * i * ldm + 3 * j;
            DTYPE tmp;
            #define krnl(k, l) mat[base + k * ldm + l]
            tmp = t1 * dx;
            krnl(0, 0) = tmp * dx + t1;
            krnl(0, 1) = tmp * dy;
            krnl(0, 2) = tmp * dz;
            tmp = t1 * dy;
            krnl(1, 0) = tmp * dx;
            krnl(1, 1) = tmp * dy + t1;
            krnl(1, 2) = tmp * dz;
            tmp = t1 * dz;
            krnl(2, 0) = tmp * dx;
            krnl(2, 1) = tmp * dy;
            krnl(2, 2) = tmp * dz + t1;
            #undef krnl
        }
    }
}

static void Stokes_krnl_bimv_intrin_d(KRNL_BIMV_PARAM)
{
    EXTRACT_3D_COORD();
    CALC_STOKES_CONST();
    for (int i = 0; i < n0; i++)
    {
        vec_d txv = vec_bcast_d(x0 + i);
        vec_d tyv = vec_bcast_d(y0 + i);
        vec_d tzv = vec_bcast_d(z0 + i);
        vec_d x_in_1_i0 = vec_bcast_d(x_in_1 + i + 0 * ld0);
        vec_d x_in_1_i1 = vec_bcast_d(x_in_1 + i + 1 * ld0);
        vec_d x_in_1_i2 = vec_bcast_d(x_in_1 + i + 2 * ld0);
        vec_d xo0_0 = vec_zero_d();
        vec_d xo0_1 = vec_zero_d();
        vec_d xo0_2 = vec_zero_d();
        vec_d frsqrt_pf = vec_frsqrt_pf_d();
        for (int j = 0; j < n1; j += SIMD_LEN_D)
        {
            vec_d dx = vec_sub_d(txv, vec_load_d(x1 + j));
            vec_d dy = vec_sub_d(tyv, vec_load_d(y1 + j));
            vec_d dz = vec_sub_d(tzv, vec_load_d(z1 + j));
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            vec_d inv_r = vec_mul_d(vec_frsqrt_d(r2), frsqrt_pf);
            
            dx = vec_mul_d(dx, inv_r);
            dy = vec_mul_d(dy, inv_r);
            dz = vec_mul_d(dz, inv_r);
            
            vec_cmp_d r2_eq_0 = vec_cmp_eq_d(r2, vec_zero_d());
            vec_d tmp0 = vec_set1_d(C);
            vec_d tmp1 = vec_mul_d(inv_r, vec_set1_d(Ca3o4));
            vec_d t1 = vec_blend_d(tmp1, tmp0, r2_eq_0);
            
            vec_d x_in_0_j0 = vec_load_d(x_in_0 + j + ld1 * 0);
            vec_d x_in_0_j1 = vec_load_d(x_in_0 + j + ld1 * 1);
            vec_d x_in_0_j2 = vec_load_d(x_in_0 + j + ld1 * 2);
            
            tmp0 = vec_mul_d(x_in_0_j0, dx);
            tmp0 = vec_fmadd_d(x_in_0_j1, dy, tmp0);
            tmp0 = vec_fmadd_d(x_in_0_j2, dz, tmp0);
            
            tmp1 = vec_mul_d(x_in_1_i0, dx);
            tmp1 = vec_fmadd_d(x_in_1_i1, dy, tmp1);
            tmp1 = vec_fmadd_d(x_in_1_i2, dz, tmp1);
            
            xo0_0 = vec_fmadd_d(t1, vec_fmadd_d(dx, tmp0, x_in_0_j0), xo0_0);
            xo0_1 = vec_fmadd_d(t1, vec_fmadd_d(dy, tmp0, x_in_0_j1), xo0_1);
            xo0_2 = vec_fmadd_d(t1, vec_fmadd_d(dz, tmp0, x_in_0_j2), xo0_2);

            DTYPE *x_out_1_0 = x_out_1 + j + 0 * ld1;
            DTYPE *x_out_1_1 = x_out_1 + j + 1 * ld1;
            DTYPE *x_out_1_2 = x_out_1 + j + 2 * ld1;
            
            vec_d xo1_0 = vec_load_d(x_out_1_0);
            vec_d xo1_1 = vec_load_d(x_out_1_1);
            vec_d xo1_2 = vec_load_d(x_out_1_2);
            
            xo1_0 = vec_fmadd_d(t1, vec_fmadd_d(dx, tmp1, x_in_1_i0), xo1_0);
            xo1_1 = vec_fmadd_d(t1, vec_fmadd_d(dy, tmp1, x_in_1_i1), xo1_1);
            xo1_2 = vec_fmadd_d(t1, vec_fmadd_d(dz, tmp1, x_in_1_i2), xo1_2);
            
            vec_store_d(x_out_1_0, xo1_0);
            vec_store_d(x_out_1_1, xo1_1);
            vec_store_d(x_out_1_2, xo1_2);
        }
        x_out_0[i + 0 * ld0] += vec_reduce_add_d(xo0_0);
        x_out_0[i + 1 * ld0] += vec_reduce_add_d(xo0_1);
        x_out_0[i + 2 * ld0] += vec_reduce_add_d(xo0_2);
    }
}

// ============================================================ //
// ======================   RPY Kernel   ====================== //
// ============================================================ //

const  int  RPY_krnl_bimv_flop = 82;

static void RPY_eval_std(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    // Radii
    const DTYPE *a0 = coord0 + ld0 * 3; 
    const DTYPE *a1 = coord1 + ld1 * 3; 
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE eta = param_[0];
    const DTYPE C   = 1.0 / (6.0 * M_PI * eta);
    for (int i = 0; i < n0; i++)
    {
        DTYPE tx = x0[i];
        DTYPE ty = y0[i];
        DTYPE tz = z0[i];
        DTYPE ta = a0[i];
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = tx - x1[j];
            DTYPE dy = ty - y1[j];
            DTYPE dz = tz - z1[j];
            DTYPE sa = a1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            DTYPE r  = DSQRT(r2);
            DTYPE inv_r  = (r == 0.0) ? 0.0 : 1.0 / r;
            DTYPE inv_r2 = inv_r * inv_r;

            dx *= inv_r;
            dy *= inv_r;
            dz *= inv_r;
            
            DTYPE t1, t2, tmp0, tmp1, tmp2;
            DTYPE ta_p_sa = ta + sa;
            DTYPE ta_m_sa = ta - sa;
            if (r > ta_p_sa)
            {
                tmp0 = C * 0.75 * inv_r;
                tmp1 = (ta * ta + sa * sa) * inv_r2;
                t1 = tmp0 * (1.0 + tmp1 / 3.0);
                t2 = tmp0 * (1.0 - tmp1);
            } else if (r > fabs(ta_m_sa)) {
                tmp0 = ta_m_sa * ta_m_sa;
                tmp1 = (inv_r2 * inv_r * C) / (ta * sa * 32.0);
                t1 = tmp0 + 3.0 * r2;
                t1 = tmp1 * (16.0 * r2 * r * ta_p_sa - t1 * t1);
                t2 = tmp0 - r2;
                t2 = tmp1 * 3.0 * t2 * t2;
            } else {
                t1 = C / (ta > sa ? ta : sa);
                t2 = 0.0;
            }
            
            DTYPE *krnl_ptr = mat + 3 * i * ldm + 3 * j;
            tmp0 = t2 * dx;
            tmp1 = t2 * dy;
            tmp2 = t2 * dz;
            krnl_ptr[0 * ldm + 0] = tmp0 * dx + t1;
            krnl_ptr[0 * ldm + 1] = tmp0 * dy;
            krnl_ptr[0 * ldm + 2] = tmp0 * dz;
            krnl_ptr[1 * ldm + 0] = tmp1 * dx; 
            krnl_ptr[1 * ldm + 1] = tmp1 * dy + t1;
            krnl_ptr[1 * ldm + 2] = tmp1 * dz;
            krnl_ptr[2 * ldm + 0] = tmp2 * dx;
            krnl_ptr[2 * ldm + 1] = tmp2 * dy;
            krnl_ptr[2 * ldm + 2] = tmp2 * dz + t1;
        }
    }
}

static void RPY_krnl_bimv_intrin_d(KRNL_BIMV_PARAM)
{
    EXTRACT_3D_COORD();
    // Radii
    const DTYPE *a0 = coord0 + ld0 * 3; 
    const DTYPE *a1 = coord1 + ld1 * 3; 
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE eta = param_[0];
    const DTYPE C = 1.0 / (6.0 * M_PI * eta);
    const vec_d vC    = vec_set1_d(C);
    const vec_d vC3o4 = vec_set1_d(C * 0.75);
    const vec_d v1    = vec_set1_d(1.0);
    const vec_d v3    = vec_set1_d(3.0);
    const vec_d v16   = vec_set1_d(16.0);
    const vec_d v32   = vec_set1_d(32.0);
    const vec_d v1o3  = vec_set1_d(1.0 / 3.0);
    for (int i = 0; i < n0; i++)
    {
        vec_d txv = vec_bcast_d(x0 + i);
        vec_d tyv = vec_bcast_d(y0 + i);
        vec_d tzv = vec_bcast_d(z0 + i);
        vec_d ta  = vec_bcast_d(a0 + i);
        vec_d x_in_1_i0 = vec_bcast_d(x_in_1 + i + 0 * ld0);
        vec_d x_in_1_i1 = vec_bcast_d(x_in_1 + i + 1 * ld0);
        vec_d x_in_1_i2 = vec_bcast_d(x_in_1 + i + 2 * ld0);
        vec_d xo0_0 = vec_zero_d();
        vec_d xo0_1 = vec_zero_d();
        vec_d xo0_2 = vec_zero_d();
        vec_d frsqrt_pf = vec_frsqrt_pf_d();
        for (int j = 0; j < n1; j += SIMD_LEN_D)
        {
            vec_d dx = vec_sub_d(txv, vec_load_d(x1 + j));
            vec_d dy = vec_sub_d(tyv, vec_load_d(y1 + j));
            vec_d dz = vec_sub_d(tzv, vec_load_d(z1 + j));
            vec_d sa = vec_load_d(a1 + j);
            vec_d r2 = vec_mul_d(dx, dx);
            r2 = vec_fmadd_d(dy, dy, r2);
            r2 = vec_fmadd_d(dz, dz, r2);
            vec_d inv_r  = vec_mul_d(vec_frsqrt_d(r2), frsqrt_pf);
            vec_d inv_r2 = vec_mul_d(inv_r, inv_r);
            vec_d r      = vec_mul_d(inv_r, r2);
            
            dx = vec_mul_d(dx, inv_r);
            dy = vec_mul_d(dy, inv_r);
            dz = vec_mul_d(dz, inv_r);

            vec_d tmp0, tmp1, t1, t2;
            vec_d t1_0, t2_0, t1_1, t2_1, t1_2, t2_2;
            vec_d ta_p_sa = vec_add_d(ta, sa);
            vec_d ta_m_sa = vec_max_d(vec_sub_d(ta, sa), vec_sub_d(sa, ta));
            // r > ta + sa
            tmp0 = vec_mul_d(vC3o4, inv_r);
            tmp1 = vec_mul_d(vec_fmadd_d(sa, sa, vec_mul_d(ta, ta)), inv_r2);
            t1_0 = vec_mul_d(tmp0, vec_fmadd_d(v1o3, tmp1, v1));
            t2_0 = vec_mul_d(tmp0, vec_sub_d(v1, tmp1));
            // ta + sa >= r > abs(ta - sa)
            tmp0 = vec_mul_d(ta_m_sa, ta_m_sa);
            tmp1 = vec_div_d(vec_mul_d(vec_mul_d(vC, inv_r2), inv_r), vec_mul_d(vec_mul_d(ta, sa), v32));
            t1_1 = vec_fmadd_d(v3, r2, tmp0);
            t1_1 = vec_mul_d(t1_1, t1_1);
            t1_1 = vec_fmsub_d(vec_mul_d(v16, r2), vec_mul_d(r, ta_p_sa), t1_1);
            t1_1 = vec_mul_d(tmp1, t1_1);
            t2_1 = vec_sub_d(tmp0, r2);
            t2_1 = vec_mul_d(t2_1, t2_1);
            t2_1 = vec_mul_d(vec_mul_d(tmp1, v3), t2_1);
            // r <= abs(ta - sa)
            t1_2 = vec_div_d(vC, vec_max_d(ta, sa));
            t2_2 = vec_set1_d(0.0);

            vec_cmp_d r_gt_ta_p_da = vec_cmp_gt_d(r, ta_p_sa);
            vec_cmp_d r_le_ta_m_da = vec_cmp_le_d(r, ta_m_sa);
            t1 = vec_blend_d(t1_1, t1_0, r_gt_ta_p_da);
            t1 = vec_blend_d(t1,   t1_2, r_le_ta_m_da);
            t2 = vec_blend_d(t2_1, t2_0, r_gt_ta_p_da);
            t2 = vec_blend_d(t2,   t2_2, r_le_ta_m_da);
            
            vec_d x_in_0_j0 = vec_load_d(x_in_0 + j + ld1 * 0);
            vec_d x_in_0_j1 = vec_load_d(x_in_0 + j + ld1 * 1);
            vec_d x_in_0_j2 = vec_load_d(x_in_0 + j + ld1 * 2);
            
            tmp0 = vec_mul_d(x_in_0_j0, dx);
            tmp0 = vec_fmadd_d(x_in_0_j1, dy, tmp0);
            tmp0 = vec_fmadd_d(x_in_0_j2, dz, tmp0);
            tmp0 = vec_mul_d(tmp0, t2);
            
            tmp1 = vec_mul_d(x_in_1_i0, dx);
            tmp1 = vec_fmadd_d(x_in_1_i1, dy, tmp1);
            tmp1 = vec_fmadd_d(x_in_1_i2, dz, tmp1);
            tmp1 = vec_mul_d(tmp1, t2);
            
            DTYPE *x_out_1_0 = x_out_1 + j + 0 * ld1;
            DTYPE *x_out_1_1 = x_out_1 + j + 1 * ld1;
            DTYPE *x_out_1_2 = x_out_1 + j + 2 * ld1;
            
            vec_d xo1_0 = vec_load_d(x_out_1_0);
            vec_d xo1_1 = vec_load_d(x_out_1_1);
            vec_d xo1_2 = vec_load_d(x_out_1_2);

            xo0_0 = vec_fmadd_d(dx, tmp0, xo0_0);
            xo0_1 = vec_fmadd_d(dy, tmp0, xo0_1);
            xo0_2 = vec_fmadd_d(dz, tmp0, xo0_2);
            xo0_0 = vec_fmadd_d(t1, x_in_0_j0, xo0_0);
            xo0_1 = vec_fmadd_d(t1, x_in_0_j1, xo0_1);
            xo0_2 = vec_fmadd_d(t1, x_in_0_j2, xo0_2);
            
            xo1_0 = vec_fmadd_d(dx, tmp1, xo1_0);
            xo1_1 = vec_fmadd_d(dy, tmp1, xo1_1);
            xo1_2 = vec_fmadd_d(dz, tmp1, xo1_2);
            xo1_0 = vec_fmadd_d(t1, x_in_1_i0, xo1_0);
            xo1_1 = vec_fmadd_d(t1, x_in_1_i1, xo1_1);
            xo1_2 = vec_fmadd_d(t1, x_in_1_i2, xo1_2);
            
            vec_store_d(x_out_1_0, xo1_0);
            vec_store_d(x_out_1_1, xo1_1);
            vec_store_d(x_out_1_2, xo1_2);
        }
        x_out_0[i + 0 * ld0] += vec_reduce_add_d(xo0_0);
        x_out_0[i + 1 * ld0] += vec_reduce_add_d(xo0_1);
        x_out_0[i + 2 * ld0] += vec_reduce_add_d(xo0_2);
    }
}

#ifdef __cplusplus
}
#endif

#endif
