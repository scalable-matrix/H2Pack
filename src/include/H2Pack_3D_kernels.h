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

#ifndef KRNL_MV_PARAM
#define KRNL_MV_PARAM \
    const DTYPE *coord0, const int ld0, const int n0,            \
    const DTYPE *coord1, const int ld1, const int n1,            \
    const void *param, const DTYPE *x_in, DTYPE *restrict x_out
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
        DTYPE tx = x0[i];
        DTYPE ty = y0[i];
        DTYPE tz = z0[i];
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = tx - x1[j];
            DTYPE dy = ty - y1[j];
            DTYPE dz = tz - z1[j];
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
const  int  RPY_krnl_mv_flop   = 70;

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

static void RPY_Ewald_init_workbuf(const DTYPE L, const DTYPE xi, const int nr, const int nk, DTYPE **workbuf_)
{
    const int r_size =  (2*nr+1) * (2*nr+1) * (2*nr+1);
    const int k_size = ((2*nk+1) * (2*nk+1) * (2*nk+1) - 1) / 2;
    DTYPE *workbuf = (DTYPE*) malloc(sizeof(DTYPE) * (3 * r_size + 6 * k_size));
    DTYPE *rx_shift_arr = workbuf;
    DTYPE *ry_shift_arr = rx_shift_arr + r_size;
    DTYPE *rz_shift_arr = ry_shift_arr + r_size;
    DTYPE *k_arr        = rz_shift_arr + r_size;
    DTYPE *kinv_arr     = k_arr        + k_size;
    DTYPE *m2_arr       = kinv_arr     + k_size;
    DTYPE *kx_arr       = m2_arr       + k_size;
    DTYPE *ky_arr       = kx_arr       + k_size;
    DTYPE *kz_arr       = ky_arr       + k_size;

    const DTYPE V_inv     = 1.0 / (L * L * L);
    const DTYPE _2_PI_o_L = 2.0 * M_PI / L;

    int idx = 0;
    for (int ix = -nr; ix <= nr; ix++)
    for (int iy = -nr; iy <= nr; iy++)
    for (int iz = -nr; iz <= nr; iz++)
    {
        DTYPE k  = _2_PI_o_L * DSQRT((DTYPE)(ix*ix + iy*iy + iz*iz));
        DTYPE k2 = k * k;
        DTYPE k2_o_xi2 = k2 / (xi * xi);
        DTYPE m2 = 2.0 * V_inv * (1.0 + 0.25 * k2_o_xi2 + 0.125 * k2_o_xi2 * k2_o_xi2) * 6.0 * M_PI / k2 * DEXP(-0.25 * k2_o_xi2);
        rx_shift_arr[idx] = L * ix;
        ry_shift_arr[idx] = L * iy;
        rz_shift_arr[idx] = L * iz;
        if (idx < k_size)
        {
            k_arr[idx]    = k;
            kinv_arr[idx] = 1.0 / k;
            m2_arr[idx]   = m2;
            kx_arr[idx]   = _2_PI_o_L * ix;
            ky_arr[idx]   = _2_PI_o_L * iy;
            kz_arr[idx]   = _2_PI_o_L * iz;
        }
        idx++;
    }

    *workbuf_ = workbuf;
}

static void RPY_Ewald_eval_std(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    // Radii
    const DTYPE *a0 = coord0 + ld0 * 3; 
    const DTYPE *a1 = coord1 + ld1 * 3; 
    // Other parameters
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE L  = param_[0];
    const DTYPE xi = param_[1];
    const int   nr = DROUND(param_[2]);
    const int   nk = DROUND(param_[3]);
    DTYPE *workbuf;
    memcpy(&workbuf, param_ + 4, sizeof(DTYPE*));

    const DTYPE xi2 = xi  * xi;
    const DTYPE xi3 = xi2 * xi;
    const DTYPE xi5 = xi3 * xi2;
    const DTYPE xi7 = xi5 * xi2;

    const DTYPE V_inv        = 1.0 / (L * L * L);
    const DTYPE _2_PI_o_L    = 2.0 * M_PI / L;
    const DTYPE _40_o_3_xi2  = 40.0 / 3.0 * xi * xi;
    const DTYPE xi_o_sqrt_PI = xi  / DSQRT(M_PI);
    const DTYPE inv_sqrt_PI  = 1.0 / DSQRT(M_PI);

    const int r_size =  (2*nr+1) * (2*nr+1) * (2*nr+1);
    const int k_size = ((2*nk+1) * (2*nk+1) * (2*nk+1) - 1) / 2;
    DTYPE *rx_shift_arr = workbuf;
    DTYPE *ry_shift_arr = rx_shift_arr + r_size;
    DTYPE *rz_shift_arr = ry_shift_arr + r_size;
    DTYPE *k_arr        = rz_shift_arr + r_size;
    DTYPE *kinv_arr     = k_arr        + k_size;
    DTYPE *m2_arr       = kinv_arr     + k_size;
    DTYPE *kx_arr       = m2_arr       + k_size;
    DTYPE *ky_arr       = kx_arr       + k_size;
    DTYPE *kz_arr       = ky_arr       + k_size;

    for (int i = 0; i < n0; i++)
    {
        DTYPE x_i = x0[i];
        DTYPE y_i = y0[i];
        DTYPE z_i = z0[i];
        DTYPE a_i = a0[i];
        DTYPE self_t = 1.0 / a_i - (6.0 - _40_o_3_xi2 * a_i * a_i) * xi_o_sqrt_PI;

        for (int j = 0; j < n1; j++)
        {
            DTYPE a00 = 0.0;
            DTYPE a10 = 0.0, a11 = 0.0;
            DTYPE a20 = 0.0, a21 = 0.0, a22 = 0.0;

            DTYPE dx = x_i - x1[j];
            DTYPE dy = y_i - y1[j];
            DTYPE dz = z_i - z1[j];
            DTYPE a_j = a1[j];

            DTYPE a3 = 0.5 * (a_i * a_i + a_j * a_j);

            // 1. Real-space sum
            #pragma omp simd
            for (int idx_r = 0; idx_r < r_size; idx_r++)
            {
                DTYPE rvec_x = dx + rx_shift_arr[idx_r];
                DTYPE rvec_y = dy + ry_shift_arr[idx_r];
                DTYPE rvec_z = dz + rz_shift_arr[idx_r];

                DTYPE r2    = rvec_x * rvec_x + rvec_y * rvec_y + rvec_z * rvec_z;
                DTYPE r4    = r2 * r2;
                DTYPE r     = DSQRT(r2);
                DTYPE rinv  = 1.0 / r;
                DTYPE rinv2 = rinv * rinv;
                DTYPE rinv3 = rinv * rinv2;

                DTYPE erfc_xi_r = DERFC(xi * r);
                DTYPE pi_exp    = inv_sqrt_PI * DEXP(-xi2 * r2);
                DTYPE tmp0 = 0.75 * rinv;
                DTYPE tmp1 = 0.5  * rinv3 * a3;
                DTYPE tmp2 = 4.0  * xi7 * a3 * r4;
                DTYPE tmp3 = 3.0  * xi3 * r2;
                DTYPE tmp4 = 4.0  * xi5 * a3 * r2;
                DTYPE tmp5 = 2.0  * xi3 * a3;
                DTYPE tmp6 = xi * a3 * rinv2;
                DTYPE m11 = (tmp0 +     tmp1) * erfc_xi_r + ( tmp2 + tmp3 - 5.0*tmp4 - 4.5*xi + 7.0*tmp5 +     tmp6) * pi_exp;
                DTYPE m12 = (tmp0 - 3.0*tmp1) * erfc_xi_r + (-tmp2 - tmp3 + 4.0*tmp4 + 1.5*xi -     tmp5 - 3.0*tmp6) * pi_exp;

                if (r2 == 0)
                {
                    m11    = 0.0;
                    m12    = 0.0;
                    rvec_x = 0.0;
                    rvec_y = 0.0;
                    rvec_z = 0.0;
                } else {
                    rvec_x *= rinv;
                    rvec_y *= rinv;
                    rvec_z *= rinv;
                }
                a00 += m12 * rvec_x * rvec_x + m11;
                a10 += m12 * rvec_x * rvec_y;
                a20 += m12 * rvec_x * rvec_z;
                a11 += m12 * rvec_y * rvec_y + m11;
                a21 += m12 * rvec_y * rvec_z;
                a22 += m12 * rvec_z * rvec_z + m11;
            }  // End of idx_r loop

            // 2. Reciprocal-space sum
            #pragma omp simd
            for (int idx_k = 0; idx_k < k_size; idx_k++)
            {
                DTYPE k      = k_arr[idx_k];
                DTYPE kinv   = kinv_arr[idx_k];
                DTYPE m2     = m2_arr[idx_k];
                DTYPE kvec_x = kx_arr[idx_k];
                DTYPE kvec_y = ky_arr[idx_k];
                DTYPE kvec_z = kz_arr[idx_k];
                DTYPE t      = m2 * DCOS(kvec_x * dx + kvec_y * dy + kvec_z * dz) * (1.0 - a3 * k * k / 3.0);

                kvec_x *= kinv;
                kvec_y *= kinv;
                kvec_z *= kinv;
                a00 += t * (1.0 - kvec_x * kvec_x);
                a10 += t *      - kvec_x * kvec_y;
                a20 += t *      - kvec_x * kvec_z;
                a11 += t * (1.0 - kvec_y * kvec_y);
                a21 += t *      - kvec_y * kvec_z;
                a22 += t * (1.0 - kvec_z * kvec_z);
            }  // End of idx_k loop

            DTYPE r2 = dx * dx + dy * dy + dz * dz;

            // 3. Overlap correction (i and j are different particles)
            if (r2 >= 1e-15 * 1e-15)
            {
                DTYPE rvec_x = DFMOD(dx + 2 * L, L);
                DTYPE rvec_y = DFMOD(dy + 2 * L, L);
                DTYPE rvec_z = DFMOD(dz + 2 * L, L);
                
                rvec_x = (rvec_x > 0.5 * L) ? rvec_x - L : rvec_x;
                rvec_y = (rvec_y > 0.5 * L) ? rvec_y - L : rvec_y;
                rvec_z = (rvec_z > 0.5 * L) ? rvec_z - L : rvec_z;
                
                DTYPE r2    = rvec_x * rvec_x + rvec_y * rvec_y + rvec_z * rvec_z;
                DTYPE r     = DSQRT(r2);
                DTYPE r3    = r2 * r;
                DTYPE rinv  = 1.0 / r;
                DTYPE rinv3 = rinv * rinv * rinv;
                
                rvec_x *= rinv;
                rvec_y *= rinv;
                rvec_z *= rinv;
                
                DTYPE t1, t2;
                DTYPE tmp0 = (a_i * a_i + a_j * a_j) / r2;
                DTYPE tmp1 = 0.75 * rinv * (1.0 + tmp0 / 3.0);
                DTYPE tmp2 = 0.75 * rinv * (1.0 - tmp0);
                DTYPE diff_aij = a_i - a_j;

                if (r > a_i + a_j) 
                {
                    // So t1 and t2 will be 0 
                    t1 = tmp1;
                    t2 = tmp2;
                }
                else if (r > DABS(diff_aij))
                {
                    DTYPE tmp3 = rinv3 / (32.0 * a_i * a_j);
                    t1 = diff_aij * diff_aij + 3.0 * r2;
                    t1 = (16.0 * r3 * (a_i + a_j) - t1 * t1) * tmp3;
                    t2 = diff_aij * diff_aij - r2;
                    t2 = 3.0 * t2 * t2 * tmp3;
                }
                else
                {
                    t1 = 1.0 / (a_i > a_j ? a_i : a_j);
                    t2 = 0;
                }
                
                t1 -= tmp1;
                t2 -= tmp2;
                a00 += t2 * rvec_x * rvec_x + t1;
                a10 += t2 * rvec_x * rvec_y;
                a20 += t2 * rvec_x * rvec_z;
                a11 += t2 * rvec_y * rvec_y + t1;
                a21 += t2 * rvec_y * rvec_z;
                a22 += t2 * rvec_z * rvec_z + t1;
            }  // End of "if (j > i)"

            // 4. Self part (i and j are the same particle)
            if (r2 < 1e-15 * 1e-15)
            {
                a00 += self_t;
                a11 += self_t;
                a22 += self_t;
            }

            // 5. Write global matrix block
            DTYPE *mat_blk = mat + (i * 3) * ldm + (j * 3);
            mat_blk[0 * ldm + 0] = a00;
            mat_blk[0 * ldm + 1] = a10;
            mat_blk[0 * ldm + 2] = a20;
            mat_blk[1 * ldm + 0] = a10;
            mat_blk[1 * ldm + 1] = a11;
            mat_blk[1 * ldm + 2] = a21;
            mat_blk[2 * ldm + 0] = a20;
            mat_blk[2 * ldm + 1] = a21;
            mat_blk[2 * ldm + 2] = a22;
        }  // End of j loop
    }  // End of i loop 
}

static void RPY_krnl_mv_intrin_d(KRNL_MV_PARAM)
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
        vec_d x_out_i0 = vec_zero_d();
        vec_d x_out_i1 = vec_zero_d();
        vec_d x_out_i2 = vec_zero_d();
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
            
            vec_d x_in_j0 = vec_load_d(x_in + j + ld1 * 0);
            vec_d x_in_j1 = vec_load_d(x_in + j + ld1 * 1);
            vec_d x_in_j2 = vec_load_d(x_in + j + ld1 * 2);
            
            tmp0 = vec_mul_d(x_in_j0, dx);
            tmp0 = vec_fmadd_d(x_in_j1, dy, tmp0);
            tmp0 = vec_fmadd_d(x_in_j2, dz, tmp0);
            tmp0 = vec_mul_d(tmp0, t2);

            x_out_i0 = vec_fmadd_d(dx, tmp0, x_out_i0);
            x_out_i1 = vec_fmadd_d(dy, tmp0, x_out_i1);
            x_out_i2 = vec_fmadd_d(dz, tmp0, x_out_i2);
            x_out_i0 = vec_fmadd_d(t1, x_in_j0, x_out_i0);
            x_out_i1 = vec_fmadd_d(t1, x_in_j1, x_out_i1);
            x_out_i2 = vec_fmadd_d(t1, x_in_j2, x_out_i2);
        }
        x_out[i + 0 * ld0] += vec_reduce_add_d(x_out_i0);
        x_out[i + 1 * ld0] += vec_reduce_add_d(x_out_i1);
        x_out[i + 2 * ld0] += vec_reduce_add_d(x_out_i2);
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
