#ifndef __H2PACK_KERNELS_H__
#define __H2PACK_KERNELS_H__

#include <math.h>

#include "H2Pack_config.h"
#include "x86_intrin_wrapper.h" 

#if DTYPE_SIZE == DOUBLE_SIZE
#define SIMD_LEN SIMD_LEN_D
#endif

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

#if DTYPE_SIZE == FLOAT_SIZE
#define Coulomb_3d_eval_intrin   Coulomb_3d_eval_intrin_s
#define Coulomb_3d_matvec_intrin Coulomb_3d_matvec_intrin_s
#endif

static void Coulomb_3d_eval_std(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *mat, const int ldm
)
{
    const DTYPE *x0 = coord0 + ld0 * 0;
    const DTYPE *y0 = coord0 + ld0 * 1;
    const DTYPE *z0 = coord0 + ld0 * 2;
    const DTYPE *x1 = coord1 + ld1 * 0;
    const DTYPE *y1 = coord1 + ld1 * 1;
    const DTYPE *z1 = coord1 + ld1 * 2;
    for (int i = 0; i < n0; i++)
    {
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        const DTYPE z0_i = z0[i];
        DTYPE *mat_irow = mat + i * ldm;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            mat_irow[j] = (r2 == 0.0) ? 0.0 : (1.0 / DSQRT(r2));
        }
    }
}

static void Coulomb_3d_matvec_std(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const double *x_in_0, const double *x_in_1, 
    double *x_out_0, double *x_out_1
)
{
    const DTYPE *x0 = coord0 + ld0 * 0;
    const DTYPE *y0 = coord0 + ld0 * 1;
    const DTYPE *z0 = coord0 + ld0 * 2;
    const DTYPE *x1 = coord1 + ld1 * 0;
    const DTYPE *y1 = coord1 + ld1 * 1;
    const DTYPE *z1 = coord1 + ld1 * 2;
    if (x_in_1 == NULL)
    {
        for (int i = 0; i < n0; i++)
        {
            const DTYPE x0_i = x0[i];
            const DTYPE y0_i = y0[i];
            const DTYPE z0_i = z0[i];
            DTYPE sum = 0.0;
            #pragma omp simd
            for (int j = 0; j < n1; j++)
            {
                DTYPE dx = x0_i - x1[j];
                DTYPE dy = y0_i - y1[j];
                DTYPE dz = z0_i - z1[j];
                DTYPE r2 = dx * dx + dy * dy + dz * dz;
                sum += (r2 == 0.0) ? 0.0 : (x_in_0[j] / DSQRT(r2));
            }
            x_out_0[i] += sum;
        }
    } else {
        for (int i = 0; i < n0; i++)
        {
            const DTYPE x0_i = x0[i];
            const DTYPE y0_i = y0[i];
            const DTYPE z0_i = z0[i];
            const DTYPE x_in_1_i = x_in_1[i];
            DTYPE sum = 0.0;
            #pragma omp simd
            for (int j = 0; j < n1; j++)
            {
                DTYPE dx = x0_i - x1[j];
                DTYPE dy = y0_i - y1[j];
                DTYPE dz = z0_i - z1[j];
                DTYPE r2 = dx * dx + dy * dy + dz * dz;
                DTYPE inv_d = (r2 == 0.0) ? 0.0 : (1.0 / DSQRT(r2));
                sum += x_in_0[j] * inv_d;
                x_out_1[j] += x_in_1_i * inv_d;
            }
            x_out_0[i] += sum;
        }
    }
}

static void Coulomb_3d_eval_intrin_d(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    double *mat, const int ldm
)
{
    const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;
    const vec_d frsqrt_pf = vec_frsqrt_pf_d();
    for (int i = 0; i < n0; i++)
    {
        double *mat_irow = mat + i * ldm;
        
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

static void Coulomb_3d_matvec_intrin_d(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in_0, const double *x_in_1, 
    double *x_out_0, double *x_out_1
)
{
    const int n0_vec = (n0 / SIMD_LEN) * SIMD_LEN;
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;
    const vec_d frsqrt_pf = vec_frsqrt_pf_d();
    if (x_in_1 == NULL)
    {
        for (int i = 0; i < n0_vec; i += SIMD_LEN)
        {
            vec_d tx = vec_loadu_d(x0 + i);
            vec_d ty = vec_loadu_d(y0 + i);
            vec_d tz = vec_loadu_d(z0 + i);
            vec_d tv = vec_zero_d();
            for (int j = 0; j < n1; j++)
            {
                vec_d dx = vec_sub_d(tx, vec_bcast_d(x1 + j));
                vec_d dy = vec_sub_d(ty, vec_bcast_d(y1 + j));
                vec_d dz = vec_sub_d(tz, vec_bcast_d(z1 + j));
                
                vec_d r2 = vec_mul_d(dx, dx);
                r2 = vec_fmadd_d(dy, dy, r2);
                r2 = vec_fmadd_d(dz, dz, r2);
                
                vec_d sv = vec_mul_d(vec_bcast_d(x_in_0 + j), frsqrt_pf);
                vec_d rinv = vec_frsqrt_d(r2);
                tv = vec_fmadd_d(rinv, sv, tv);
            }
            vec_d ov0 = vec_loadu_d(x_out_0 + i);
            vec_storeu_d(x_out_0 + i, vec_add_d(ov0, tv));
        }
        Coulomb_3d_matvec_std(
            coord0 + n0_vec, ld0, n0 - n0_vec,
            coord1, ld1, n1,
            x_in_0, NULL, x_out_0 + n0_vec, NULL
        );
    } else {
        for (int i = 0; i < n0_vec; i += SIMD_LEN)
        {
            vec_d tx = vec_loadu_d(x0 + i);
            vec_d ty = vec_loadu_d(y0 + i);
            vec_d tz = vec_loadu_d(z0 + i);
            vec_d tv = vec_zero_d();
            for (int j = 0; j < n1; j++)
            {
                vec_d dx = vec_sub_d(tx, vec_bcast_d(x1 + j));
                vec_d dy = vec_sub_d(ty, vec_bcast_d(y1 + j));
                vec_d dz = vec_sub_d(tz, vec_bcast_d(z1 + j));
                
                vec_d r2 = vec_mul_d(dx, dx);
                r2 = vec_fmadd_d(dy, dy, r2);
                r2 = vec_fmadd_d(dz, dz, r2);
                
                vec_d sv = vec_mul_d(vec_bcast_d(x_in_0 + j), frsqrt_pf);
                vec_d rinv = vec_frsqrt_d(r2);
                tv = vec_fmadd_d(rinv, sv, tv);
            }
            vec_d ov0 = vec_loadu_d(x_out_0 + i);
            vec_storeu_d(x_out_0 + i, vec_add_d(ov0, tv));
        }
        Coulomb_3d_matvec_std(
            coord0 + n0_vec, ld0, n0 - n0_vec,
            coord1, ld1, n1,
            x_in_0, NULL, x_out_0 + n0_vec, NULL
        );
        
        const int n1_vec = (n1 / SIMD_LEN) * SIMD_LEN;
        for (int j = 0; j < n1_vec; j += SIMD_LEN)
        {
            vec_d tx = vec_loadu_d(x1 + j);
            vec_d ty = vec_loadu_d(y1 + j);
            vec_d tz = vec_loadu_d(z1 + j);
            vec_d tv = vec_zero_d();
            for (int i = 0; i < n0; i++)
            {
                vec_d dx = vec_sub_d(tx, vec_bcast_d(x0 + i));
                vec_d dy = vec_sub_d(ty, vec_bcast_d(y0 + i));
                vec_d dz = vec_sub_d(tz, vec_bcast_d(z0 + i));
                
                vec_d r2 = vec_mul_d(dx, dx);
                r2 = vec_fmadd_d(dy, dy, r2);
                r2 = vec_fmadd_d(dz, dz, r2);
                
                vec_d sv = vec_mul_d(vec_bcast_d(x_in_1 + i), frsqrt_pf);
                vec_d rinv = vec_frsqrt_d(r2);
                tv = vec_fmadd_d(rinv, sv, tv);
            }
            vec_d ov1 = vec_loadu_d(x_out_1 + j);
            vec_storeu_d(x_out_1 + j, vec_add_d(ov1, tv));
        }
        Coulomb_3d_matvec_std(
            coord1 + n1_vec, ld1, n1 - n1_vec,
            coord0, ld0, n0,
            x_in_1, NULL, x_out_1 + n1_vec, NULL
        );
    }
}

// ============================================================ //
// ======================   RPY Kernel   ====================== //
// ============================================================ //

static void RPY_3d_eval_std(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *mat, const int ldm
)
{
    const DTYPE a = 1.0, eta = 1.0;
    const DTYPE C   = 1.0 / (6.0 * M_PI * a * eta);
    const DTYPE aa  = a * a;
    const DTYPE a2  = 2.0 * a;
    const DTYPE aa2 = aa * 2.0;
    const DTYPE aa_2o3   = aa2 / 3.0;
    const DTYPE C_075    = C * 0.75;
    const DTYPE C_9o32oa = C * 9.0 / 32.0 / a;
    const DTYPE C_3o32oa = C * 3.0 / 32.0 / a;
    for (int i = 0; i < n0; i++)
    {
        DTYPE x0 = coord0[i];
        DTYPE y0 = coord0[i + ld0];
        DTYPE z0 = coord0[i + ld0 * 2];
        for (int j = 0; j < n1; j++)
        {
            DTYPE r0 = x0 - coord1[j];
            DTYPE r1 = y0 - coord1[j + ld1];
            DTYPE r2 = z0 - coord1[j + ld1 * 2];
            DTYPE s2 = r0 * r0 + r1 * r1 + r2 * r2;
            DTYPE s  = DSQRT(s2);
            int base = 3 * i * ldm + 3 * j;
            #define krnl(k, l) mat[base + k * ldm + l]
            if (s < 1e-15)
            {
                krnl(0, 0) = C;
                krnl(0, 1) = 0;
                krnl(0, 2) = 0;
                krnl(1, 0) = 0;
                krnl(1, 1) = C;
                krnl(1, 2) = 0;
                krnl(2, 0) = 0;
                krnl(2, 1) = 0;
                krnl(2, 2) = C;
                continue;
            }
            
            DTYPE inv_s = 1.0 / s;
            r0 *= inv_s;
            r1 *= inv_s;
            r2 *= inv_s;
            DTYPE t1, t2;
            if (s < a2)
            {
                t1 = C - C_9o32oa * s;
                t2 = C_3o32oa * s;
            } else {
                t1 = C_075 / s * (1 + aa_2o3 / s2);
                t2 = C_075 / s * (1 - aa2 / s2); 
            }
            krnl(0, 0) = t2 * r0 * r0 + t1;
            krnl(0, 1) = t2 * r0 * r1;
            krnl(0, 2) = t2 * r0 * r2;
            krnl(1, 0) = t2 * r1 * r0;
            krnl(1, 1) = t2 * r1 * r1 + t1;
            krnl(1, 2) = t2 * r1 * r2;
            krnl(2, 0) = t2 * r2 * r0;
            krnl(2, 1) = t2 * r2 * r1;
            krnl(2, 2) = t2 * r2 * r2 + t1;
            #undef krnl
        }
    }
}

static void RPY_3d_matvec_std(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const DTYPE *x_in_0, const DTYPE *x_in_1, 
    DTYPE *x_out_0, DTYPE *x_out_1
)
{
    const DTYPE a = 1.0, eta = 1.0;
    const DTYPE C   = 1.0 / (6.0 * M_PI * a * eta);
    const DTYPE aa  = a * a;
    const DTYPE a2  = 2.0 * a;
    const DTYPE aa2 = aa * 2.0;
    const DTYPE aa_2o3   = aa2 / 3.0;
    const DTYPE C_075    = C * 0.75;
    const DTYPE C_9o32oa = C * 9.0 / 32.0 / a;
    const DTYPE C_3o32oa = C * 3.0 / 32.0 / a;

    if (x_in_1 == NULL)
    {
        for (int i = 0; i < n0; i++)
        {
            DTYPE x0 = coord0[i];
            DTYPE y0 = coord0[i + ld0];
            DTYPE z0 = coord0[i + ld0 * 2];
            DTYPE res[3] = {0.0, 0.0, 0.0};
            for (int j = 0; j < n1; j++)
            {
                DTYPE r0 = x0 - coord1[j];
                DTYPE r1 = y0 - coord1[j + ld1];
                DTYPE r2 = z0 - coord1[j + ld1 * 2];
                DTYPE s2 = r0 * r0 + r1 * r1 + r2 * r2;
                DTYPE s  = DSQRT(s2);
                DTYPE x_in_0_j[3];
                x_in_0_j[0] = x_in_0[j * 3 + 0];
                x_in_0_j[1] = x_in_0[j * 3 + 1];
                x_in_0_j[2] = x_in_0[j * 3 + 2];

                if (s < 1e-15)
                {
                    res[0] += C * x_in_0_j[0];
                    res[1] += C * x_in_0_j[1];
                    res[2] += C * x_in_0_j[2];
                    continue;
                }
                
                DTYPE inv_s = 1.0 / s;
                r0 *= inv_s;
                r1 *= inv_s;
                r2 *= inv_s;
                DTYPE t1, t2;
                if (s < a2)
                {
                    t1 = C - C_9o32oa * s;
                    t2 = C_3o32oa * s;
                } else {
                    t1 = C_075 / s * (1 + aa_2o3 / s2);
                    t2 = C_075 / s * (1 - aa2 / s2); 
                }

                res[0] += (t2 * r0 * r0 + t1) * x_in_0_j[0];
                res[0] += (t2 * r0 * r1)      * x_in_0_j[1];
                res[0] += (t2 * r0 * r2)      * x_in_0_j[2];
                res[1] += (t2 * r1 * r0)      * x_in_0_j[0];
                res[1] += (t2 * r1 * r1 + t1) * x_in_0_j[1];
                res[1] += (t2 * r1 * r2)      * x_in_0_j[2];
                res[2] += (t2 * r2 * r0)      * x_in_0_j[0];
                res[2] += (t2 * r2 * r1)      * x_in_0_j[1];
                res[2] += (t2 * r2 * r2 + t1) * x_in_0_j[2];
            }
            x_out_0[i * 3 + 0] += res[0];
            x_out_0[i * 3 + 1] += res[1];
            x_out_0[i * 3 + 2] += res[2];
        }
    } else {
        for (int i = 0; i < n0; i++)
        {
            DTYPE x0 = coord0[i];
            DTYPE y0 = coord0[i + ld0];
            DTYPE z0 = coord0[i + ld0 * 2];
            DTYPE res[3] = {0.0, 0.0, 0.0};
            DTYPE x_in_1_i[3];
            int i3 = i * 3;
            x_in_1_i[0] = x_in_1[i3 + 0];
            x_in_1_i[1] = x_in_1[i3 + 1];
            x_in_1_i[2] = x_in_1[i3 + 2];

            for (int j = 0; j < n1; j++)
            {
                DTYPE r0 = x0 - coord1[j];
                DTYPE r1 = y0 - coord1[j + ld1];
                DTYPE r2 = z0 - coord1[j + ld1 * 2];
                DTYPE s2 = r0 * r0 + r1 * r1 + r2 * r2;
                DTYPE s  = DSQRT(s2);
                DTYPE x_in_0_j[3];
                int j3 = j * 3;
                x_in_0_j[0] = x_in_0[j3 + 0];
                x_in_0_j[1] = x_in_0[j3 + 1];
                x_in_0_j[2] = x_in_0[j3 + 2];

                if (s < 1e-15)
                {
                    res[0] += C * x_in_0_j[0];
                    res[1] += C * x_in_0_j[1];
                    res[2] += C * x_in_0_j[2];
                    x_out_1[j3 + 0] += C * x_in_1_i[0];
                    x_out_1[j3 + 1] += C * x_in_1_i[1];
                    x_out_1[j3 + 2] += C * x_in_1_i[2];
                    continue;
                }
                
                DTYPE inv_s = 1.0 / s;
                r0 *= inv_s;
                r1 *= inv_s;
                r2 *= inv_s;
                DTYPE t1, t2;
                if (s < a2)
                {
                    t1 = C - C_9o32oa * s;
                    t2 = C_3o32oa * s;
                } else {
                    t1 = C_075 / s * (1 + aa_2o3 / s2);
                    t2 = C_075 / s * (1 - aa2 / s2); 
                }

                res[0] += (t2 * r0 * r0 + t1) * x_in_0_j[0];
                res[0] += (t2 * r0 * r1)      * x_in_0_j[1];
                res[0] += (t2 * r0 * r2)      * x_in_0_j[2];
                res[1] += (t2 * r1 * r0)      * x_in_0_j[0];
                res[1] += (t2 * r1 * r1 + t1) * x_in_0_j[1];
                res[1] += (t2 * r1 * r2)      * x_in_0_j[2];
                res[2] += (t2 * r2 * r0)      * x_in_0_j[0];
                res[2] += (t2 * r2 * r1)      * x_in_0_j[1];
                res[2] += (t2 * r2 * r2 + t1) * x_in_0_j[2];

                x_out_1[j3 + 0] += (t2 * r0 * r0 + t1) * x_in_1_i[0];
                x_out_1[j3 + 0] += (t2 * r1 * r0)      * x_in_1_i[1];
                x_out_1[j3 + 0] += (t2 * r2 * r0)      * x_in_1_i[2];
                x_out_1[j3 + 1] += (t2 * r0 * r1)      * x_in_1_i[0];
                x_out_1[j3 + 1] += (t2 * r1 * r1 + t1) * x_in_1_i[1];
                x_out_1[j3 + 1] += (t2 * r2 * r1)      * x_in_1_i[2];
                x_out_1[j3 + 2] += (t2 * r0 * r2)      * x_in_1_i[0];
                x_out_1[j3 + 2] += (t2 * r1 * r2)      * x_in_1_i[1];
                x_out_1[j3 + 2] += (t2 * r2 * r2 + t1) * x_in_1_i[2];
            }
            x_out_0[i3 + 0] += res[0];
            x_out_0[i3 + 1] += res[1];
            x_out_0[i3 + 2] += res[2];
        }
    }
}


#ifdef __cplusplus
}
#endif

#endif
