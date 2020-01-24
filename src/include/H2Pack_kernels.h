#ifndef __H2PACK_KERNELS_H__
#define __H2PACK_KERNELS_H__

#include <math.h>

#include "H2Pack_config.h"
#include "x86_intrin_wrapper.h" 

#define KRNL_EVAL_PARAM \
    const DTYPE *coord0, const int ld0, const int n0, \
    const DTYPE *coord1, const int ld1, const int n1, \
    DTYPE *mat, const int ldm 

#define KRNL_SYMMV_PARAM \
    const DTYPE *coord0, const int ld0, const int n0, \
    const DTYPE *coord1, const int ld1, const int n1, \
    const DTYPE *x_in_0, const DTYPE *x_in_1,         \
    DTYPE *x_out_0, DTYPE *x_out_1

#define EXTRACT_3D_COORD() \
    const DTYPE *x0 = coord0 + ld0 * 0; \
    const DTYPE *y0 = coord0 + ld0 * 1; \
    const DTYPE *z0 = coord0 + ld0 * 2; \
    const DTYPE *x1 = coord1 + ld1 * 0; \
    const DTYPE *y1 = coord1 + ld1 * 1; \
    const DTYPE *z1 = coord1 + ld1 * 2; 

// When counting symmv flops, report effective flops (1 / sqrt(x) == 2 flops)
// instead of achieved flops (1 / sqrt(x) == 1 + NEWTON_ITER * 4 flops)

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================ //
// ====================   Coulomb Kernel   ==================== //
// ============================================================ //

const  int  Coulomb_3d_krnl_symmv_flop = 14;

static void Coulomb_3d_eval_intrin_d(KRNL_EVAL_PARAM)
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

static void Coulomb_3d_krnl_symmv_intrin_d(KRNL_SYMMV_PARAM)
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
// ====================   Gaussian Kernel   =================== //
// ============================================================ //

const  int  Gaussian_3d_krnl_symmv_flop = 13;

static void Gaussian_3d_eval_intrin_d(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
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
            
            r2 = vec_sub_d(vec_zero_d(), r2);
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
            mat_irow[j] = exp(-r2);
        }
    }
}

static void Gaussian_3d_krnl_symmv_intrin_d(KRNL_SYMMV_PARAM)
{
    EXTRACT_3D_COORD();
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
            
            r20 = vec_sub_d(vec_zero_d(), r20);
            r21 = vec_sub_d(vec_zero_d(), r21);
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

const  int  Matern_3d_krnl_symmv_flop = 17;

#define NSQRT3 -1.7320508075688772

static void Matern_3d_eval_intrin_d(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
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
            r = vec_mul_d(r, vec_set1_d(NSQRT3));
            r = vec_mul_d(vec_sub_d(vec_set1_d(1.0), r), vec_exp_d(r));
            
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
            r = r * NSQRT3;
            r = (1.0 - r) * exp(r);
            mat_irow[j] = r;
        }
    }
}

static void Matern_3d_krnl_symmv_intrin_d(KRNL_SYMMV_PARAM)
{
    EXTRACT_3D_COORD();
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
            
            r0 = vec_mul_d(r0, vec_set1_d(NSQRT3));
            r1 = vec_mul_d(r1, vec_set1_d(NSQRT3));
            r0 = vec_mul_d(vec_sub_d(vec_set1_d(1.0), r0), vec_exp_d(r0));
            r1 = vec_mul_d(vec_sub_d(vec_set1_d(1.0), r1), vec_exp_d(r1));
            
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
// =====================   Stokes Kernel   ==================== //
// ============================================================ //

#define CALC_STOKES_CONST() \
    const double eta = 1.0;                          \
    const double a   = 1.0;                          \
    const double C   = 1.0 / (6.0 * M_PI * a * eta); \
    const double Ca3o4 = C * a * 0.75;               

const  int  Stokes_krnl_symmv_flop = 48;

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

static void Stokes_krnl_symmv_intrin_d(KRNL_SYMMV_PARAM)
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

#define CALC_RPY_CONST() \
    const DTYPE a   = 1.0;                          \
    const DTYPE eta = 1.0;                          \
    const DTYPE C   = 1.0 / (6.0 * M_PI * a * eta); \
    const DTYPE aa  = a * a;                        \
    const DTYPE a2  = 2.0 * a;                      \
    const DTYPE aa2 = aa * 2.0;                     \
    const DTYPE aa_2o3   = aa2 / 3.0;               \
    const DTYPE C_075    = C * 0.75;                \
    const DTYPE C_9o32oa = C * 9.0 / 32.0 / a;      \
    const DTYPE C_3o32oa = C * 3.0 / 32.0 / a;

const  int  RPY_krnl_symmv_flop = 62;

static void RPY_eval_std(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    CALC_RPY_CONST();
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
            DTYPE r  = DSQRT(r2);
            DTYPE inv_r  = (r == 0.0) ? 0.0 : 1.0 / r;

            dx *= inv_r;
            dy *= inv_r;
            dz *= inv_r;
            
            DTYPE t1, t2;
            if (r < a2)
            {
                t1 = C - C_9o32oa * r;
                t2 =     C_3o32oa * r;
            } else {
                DTYPE inv_r2 = inv_r * inv_r;
                t1 = C_075 * inv_r * (1 + aa_2o3 * inv_r2);
                t2 = C_075 * inv_r * (1 - aa2    * inv_r2); 
            }
            
            int base = 3 * i * ldm + 3 * j;
            DTYPE tmp;
            #define krnl(k, l) mat[base + k * ldm + l]
            tmp = t2 * dx;
            krnl(0, 0) = tmp * dx + t1;
            krnl(0, 1) = tmp * dy;
            krnl(0, 2) = tmp * dz;
            tmp = t2 * dy;
            krnl(1, 0) = tmp * dx;
            krnl(1, 1) = tmp * dy + t1;
            krnl(1, 2) = tmp * dz;
            tmp = t2 * dz;
            krnl(2, 0) = tmp * dx;
            krnl(2, 1) = tmp * dy;
            krnl(2, 2) = tmp * dz + t1;
            #undef krnl
        }
    }
}

static void RPY_krnl_symmv_intrin_d(KRNL_SYMMV_PARAM)
{
    EXTRACT_3D_COORD();
    CALC_RPY_CONST();
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
            vec_d r = vec_sqrt_d(r2);
            vec_d inv_r = vec_mul_d(vec_frsqrt_d(r2), frsqrt_pf);
            
            dx = vec_mul_d(dx, inv_r);
            dy = vec_mul_d(dy, inv_r);
            dz = vec_mul_d(dz, inv_r);
            
            vec_cmp_d r_lt_a2 = vec_cmp_lt_d(r, vec_set1_d(a2));
            vec_d C_075_o_r = vec_mul_d(vec_set1_d(C_075), inv_r);
            vec_d inv_r2 = vec_mul_d(inv_r, inv_r);
            
            vec_d tmp0, tmp1, tmp2, t1, t2;
            tmp0 = vec_fnmadd_d(vec_set1_d(C_9o32oa), r, vec_set1_d(C));
            tmp1 = vec_fmadd_d(vec_set1_d(aa_2o3), inv_r2, vec_set1_d(1));
            tmp1 = vec_mul_d(C_075_o_r, tmp1);
            t1   = vec_blend_d(tmp1, tmp0, r_lt_a2);
            
            tmp0 = vec_mul_d(vec_set1_d(C_3o32oa), r);
            tmp1 = vec_fnmadd_d(vec_set1_d(aa2), inv_r2, vec_set1_d(1));
            tmp1 = vec_mul_d(C_075_o_r, tmp1);
            t2   = vec_blend_d(tmp1, tmp0, r_lt_a2);
            
            vec_d x_in_0_j0 = vec_load_d(x_in_0 + j + ld1 * 0);
            vec_d x_in_0_j1 = vec_load_d(x_in_0 + j + ld1 * 1);
            vec_d x_in_0_j2 = vec_load_d(x_in_0 + j + ld1 * 2);
            
            #define xo1_0 tmp0
            #define xo1_1 tmp1
            #define xo1_2 tmp2
            #define k0    C_075_o_r
            #define k1    inv_r2
            k0 = vec_mul_d(x_in_0_j0, dx);
            k0 = vec_fmadd_d(x_in_0_j1, dy, k0);
            k0 = vec_fmadd_d(x_in_0_j2, dz, k0);
            k0 = vec_mul_d(t2, k0);
            k1 = vec_mul_d(x_in_1_i0, dx);
            k1 = vec_fmadd_d(x_in_1_i1, dy, k1);
            k1 = vec_fmadd_d(x_in_1_i2, dz, k1);
            k1 = vec_mul_d(t2, k1);
            
            xo0_0 = vec_fmadd_d(dx, k0, xo0_0);
            xo0_1 = vec_fmadd_d(dy, k0, xo0_1);
            xo0_2 = vec_fmadd_d(dz, k0, xo0_2);
            xo0_0 = vec_fmadd_d(t1, x_in_0_j0, xo0_0);
            xo0_1 = vec_fmadd_d(t1, x_in_0_j1, xo0_1);
            xo0_2 = vec_fmadd_d(t1, x_in_0_j2, xo0_2);
            
            DTYPE *x_out_1_0 = x_out_1 + j + 0 * ld1;
            DTYPE *x_out_1_1 = x_out_1 + j + 1 * ld1;
            DTYPE *x_out_1_2 = x_out_1 + j + 2 * ld1;
            
            xo1_0 = vec_load_d(x_out_1_0);
            xo1_1 = vec_load_d(x_out_1_1);
            xo1_2 = vec_load_d(x_out_1_2);
            
            xo1_0 = vec_fmadd_d(dx, k1, xo1_0);
            xo1_1 = vec_fmadd_d(dy, k1, xo1_1);
            xo1_2 = vec_fmadd_d(dz, k1, xo1_2);
            xo1_0 = vec_fmadd_d(t1, x_in_1_i0, xo1_0);
            xo1_1 = vec_fmadd_d(t1, x_in_1_i1, xo1_1);
            xo1_2 = vec_fmadd_d(t1, x_in_1_i2, xo1_2);

            vec_store_d(x_out_1_0, xo1_0);
            vec_store_d(x_out_1_1, xo1_1);
            vec_store_d(x_out_1_2, xo1_2);
            #undef xo1_0
            #undef xo1_1
            #undef xo1_2
            #undef k0
            #undef k1
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
