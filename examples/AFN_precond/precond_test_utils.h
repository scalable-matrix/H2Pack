#ifndef __PRECOND_TEST_UTILS_H__
#define __PRECOND_TEST_UTILS_H__

#include <stdio.h>
#include "H2Pack_typedef.h"
#include "H2Pack_kernels.h"
#include "H2Pack.h"

static DTYPE shift_ = 0.0;
static int n_ = 0;

static void H2Pack_matvec(const void *h2pack_, const DTYPE *b, DTYPE *x)
{
    H2Pack_p h2pack = (H2Pack_p) h2pack_;
    H2P_matvec(h2pack, b, x);
    #pragma omp simd
    for (int i = 0; i < h2pack->krnl_mat_size; i++) x[i] += shift_ * b[i];
}

static void select_kernel(
    const int kid, const int pt_dim, const DTYPE kp, const DTYPE mu, const int npt, 
    kernel_eval_fptr *krnl_eval_, kernel_bimv_fptr *krnl_bimv_, int *krnl_bimv_flops_
)
{
    shift_ = mu;
    n_ = npt;
    kernel_eval_fptr krnl_eval = NULL;
    kernel_bimv_fptr krnl_bimv = NULL;
    int krnl_bimv_flops = 0;
    switch (kid)
    {
        case 1:
        {
            if (pt_dim == 3)
            {
                krnl_eval = Gaussian_3D_eval_intrin_t;
                krnl_bimv = Gaussian_3D_krnl_bimv_intrin_t;
                krnl_bimv_flops = Gaussian_3D_krnl_bimv_flop;
            } else {
                krnl_eval = Gaussian_2D_eval_intrin_t;
                krnl_bimv = Gaussian_2D_krnl_bimv_intrin_t;
                krnl_bimv_flops = Gaussian_2D_krnl_bimv_flop;
            }
            printf("Test kernel: Gaussian    k(x, y) = exp(-l * |x-y|^2), l = %.4f\n", kp);
            break;
        }
        case 2:
        {
            if (pt_dim == 3)
            {
                krnl_eval = Expon_3D_eval_intrin_t;
                krnl_bimv = Expon_3D_krnl_bimv_intrin_t;
                krnl_bimv_flops = Expon_3D_krnl_bimv_flop;
            } else {
                krnl_eval = Expon_2D_eval_intrin_t;
                krnl_bimv = Expon_2D_krnl_bimv_intrin_t;
                krnl_bimv_flops = Expon_2D_krnl_bimv_flop;
            }
            printf("Test kernel: Exponential k(x, y) = exp(-l * |x-y|), l = %.4f\n", kp);
            break;
        }
        case 3:
        {
            if (pt_dim == 3)
            {
                krnl_eval = Matern32_3D_eval_intrin_t;
                krnl_bimv = Matern32_3D_krnl_bimv_intrin_t;
                krnl_bimv_flops = Matern32_3D_krnl_bimv_flop;
            } else {
                krnl_eval = Matern32_2D_eval_intrin_t;
                krnl_bimv = Matern32_2D_krnl_bimv_intrin_t;
                krnl_bimv_flops = Matern32_2D_krnl_bimv_flop;
            }
            printf("Test kernel: 3/2 Matern  k(x, y) = (1 + l*k) * exp(-l*k), k = sqrt(3) * |x-y|, l = %.4f\n", kp);
            break;
        }
        case 4:
        {
            if (pt_dim == 3)
            {
                krnl_eval = Matern52_3D_eval_intrin_t;
                krnl_bimv = Matern52_3D_krnl_bimv_intrin_t;
                krnl_bimv_flops = Matern32_3D_krnl_bimv_flop;
            } else {
                krnl_eval = Matern52_2D_eval_intrin_t;
                krnl_bimv = Matern52_2D_krnl_bimv_intrin_t;
                krnl_bimv_flops = Matern32_2D_krnl_bimv_flop;
            }
            printf("Test kernel: 5/2 Matern  k(x, y) = (1 + l*k + l^2*k^2/3) * exp(-l*k), l = %.4f\n", kp);
            break;
        }
    } 
    *krnl_eval_ = krnl_eval;
    *krnl_bimv_ = krnl_bimv;
    *krnl_bimv_flops_ = krnl_bimv_flops;
}

#endif
