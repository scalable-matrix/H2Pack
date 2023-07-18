#ifndef __PRECOND_TEST_UTILS_H__
#define __PRECOND_TEST_UTILS_H__

#include <stdio.h>
#include "H2Pack_typedef.h"
#include "H2Pack_kernels.h"
#include "H2Pack.h"
#include "../PCG/pcg.h"

static DTYPE shift_ = 0.0;
static int n_ = 0;

static void H2Pack_matvec_diagshift(const void *h2pack_, const DTYPE *b, DTYPE *x)
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

static void H2mat_build(
    const int npt, const int pt_dim, DTYPE *coord, DTYPE reltol, kernel_eval_fptr krnl_eval, 
    kernel_bimv_fptr krnl_bimv, int krnl_bimv_flops, void *krnl_param, H2Pack_p *h2mat_
)
{
    double st, et;
    H2Pack_p h2mat = NULL;
    int krnl_dim = 1, BD_JIT = 1;
    H2P_dense_mat_p *pp = NULL;
    printf("Building H2 representation with reltol = %.4e for kernel matrix...\n", reltol);
    H2P_init(&h2mat, pt_dim, krnl_dim, QR_REL_NRM, &reltol);
    H2P_calc_enclosing_box(pt_dim, npt, coord, NULL, &h2mat->root_enbox);
    H2P_partition_points(h2mat, npt, coord, 0, 0);
    st = get_wtime_sec();
    H2P_generate_proxy_point_ID_file(h2mat, krnl_param, krnl_eval, NULL, &pp);
    et = get_wtime_sec();
    printf("H2Pack proxy point selection time = %.3f s\n", et - st);
    st = get_wtime_sec();
    H2P_build(h2mat, pp, BD_JIT, krnl_param, krnl_eval, krnl_bimv, krnl_bimv_flops);
    et = get_wtime_sec();
    printf("H2Pack build time = %.3f s\n", et - st);
    H2P_print_statistic(h2mat);
    H2P_dense_mat_destroy(pp);
    *h2mat_ = h2mat;
    printf("\n");
}

static void test_PCG(
    matvec_fptr Ax, void *Ax_param, matvec_fptr invMx, void *invMx_param,
    const int n, const int max_iter, const DTYPE CG_reltol
)
{
    DTYPE relres;
    int flag, iter, pcg_print_level = 1;
    DTYPE *x = malloc(sizeof(DTYPE) * n);
    DTYPE *b = malloc(sizeof(DTYPE) * n);
    srand(126);  // Match with Tianshi's code
    for (int i = 0; i < n; i++)
    {
        b[i] = (rand() / (DTYPE) RAND_MAX) - 0.5;
        x[i] = 0.0;
    }
    pcg(
        n, CG_reltol, max_iter, 
        Ax, Ax_param, b, invMx, invMx_param, x,
        &flag, &relres, &iter, NULL, pcg_print_level
    );
    free(x);
    free(b);
}

#endif
