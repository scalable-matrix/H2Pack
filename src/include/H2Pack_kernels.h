#ifndef __H2PACK_KERNELS_H__
#define __H2PACK_KERNELS_H__

#include <math.h>

#include "H2Pack_config.h"
#include "x86_intrin_wrapper.h" 

#define EVAL_KRNL_PARAM \
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

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================ //
// ====================   Coulomb Kernel   ==================== //
// ============================================================ //

#if DTYPE_SIZE == DOUBLE_SIZE
#define Coulomb_3d_eval_intrin   Coulomb_3d_eval_intrin_d
#define Coulomb_3d_matvec_intrin Coulomb_3d_matvec_intrin_d
#endif

// Report the effective instead of achieved FLOP
// 3: dx, dy, dz; 5: r2 = dx^2 + dy^2 + dz^2; 2: 1/sqrt(r2); 4: matvec
const int Coulomb_3d_krnl_symmv_flop = (3 + 5 + 2 + 4);

static void Coulomb_3d_eval_intrin_d(EVAL_KRNL_PARAM)
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
// ======================   RPY Kernel   ====================== //
// ============================================================ //

const DTYPE RPY_a   = 1.0;
const DTYPE RPY_eta = 1.0;
#define CALC_RPY_CONST() \
    const DTYPE C   = 1.0 / (6.0 * M_PI * RPY_a * RPY_eta); \
    const DTYPE aa  = RPY_a * RPY_a;                        \
    const DTYPE a2  = 2.0 * RPY_a;                          \
    const DTYPE aa2 = aa * 2.0;                             \
    const DTYPE aa_2o3   = aa2 / 3.0;                       \
    const DTYPE C_075    = C * 0.75;                        \
    const DTYPE C_9o32oa = C * 9.0 / 32.0 / RPY_a;          \
    const DTYPE C_3o32oa = C * 3.0 / 32.0 / RPY_a;

// Report the effective instead of achieved FLOP
const int RPY_krnl_symmv_flop = 62;

static void RPY_eval_std(EVAL_KRNL_PARAM)
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
            DTYPE inv_r  = (r < 1e-15) ? 0.0 : 1.0 / r;
            DTYPE inv_r2 = inv_r * inv_r;
            
            dx *= inv_r;
            dy *= inv_r;
            dz *= inv_r;
            
            DTYPE t1, t2;
            if (r < a2)
            {
                t1 = C - C_9o32oa * r;
                t2 =     C_3o32oa * r;
            } else {
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
            vec_d dx = vec_sub_d(txv, vec_loadu_d(x1 + j));
            vec_d dy = vec_sub_d(tyv, vec_loadu_d(y1 + j));
            vec_d dz = vec_sub_d(tzv, vec_loadu_d(z1 + j));
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
            
            vec_d x_in_0_j0 = vec_loadu_d(x_in_0 + j);
            vec_d x_in_0_j1 = vec_loadu_d(x_in_0 + j + ld1);
            vec_d x_in_0_j2 = vec_loadu_d(x_in_0 + j + ld1 * 2);
            
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
            
            xo1_0 = vec_mul_d(dx, k1);
            xo1_1 = vec_mul_d(dy, k1);
            xo1_2 = vec_mul_d(dz, k1);
            xo1_0 = vec_fmadd_d(t1, x_in_1_i0, xo1_0);
            xo1_1 = vec_fmadd_d(t1, x_in_1_i1, xo1_1);
            xo1_2 = vec_fmadd_d(t1, x_in_1_i2, xo1_2);

            DTYPE *x_out_1_0 = x_out_1 + j + 0 * ld1;
            DTYPE *x_out_1_1 = x_out_1 + j + 1 * ld1;
            DTYPE *x_out_1_2 = x_out_1 + j + 2 * ld1;
            vec_storeu_d(x_out_1_0, vec_add_d(xo1_0, vec_loadu_d(x_out_1_0)));
            vec_storeu_d(x_out_1_1, vec_add_d(xo1_1, vec_loadu_d(x_out_1_1)));
            vec_storeu_d(x_out_1_2, vec_add_d(xo1_2, vec_loadu_d(x_out_1_2)));
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
