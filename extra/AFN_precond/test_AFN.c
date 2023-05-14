#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_kernels.h"
#include "AFN_precond.h"
#include "pcg.h"

static DTYPE shift_ = 0.0;

void H2Pack_matvec(const void *h2pack_, const DTYPE *b, DTYPE *x)
{
    H2Pack_p h2pack = (H2Pack_p) h2pack_;
    H2P_matvec(h2pack, b, x);
    #pragma omp simd
    for (int i = 0; i < h2pack->krnl_mat_size; i++) x[i] += shift_ * b[i];
}

void AFN_precond_apply_(const void *precond_, const DTYPE *b, DTYPE *x)
{
    AFN_precond_p AFN_precond = (AFN_precond_p) precond_;
    AFN_precond_apply(AFN_precond, b, x);
}

int main(int argc, char **argv)
{
    int npt, pt_dim, max_k, ss_npt, fsai_npt;
    DTYPE mu, l, *coord = NULL;
    if (argc < 9)
    {
        printf("Usage: %s npt pt_dim l mu max_k ss_npt fsai_npt coord_bin\n", argv[0]);
        printf("  - npt       [int]    : Number of points\n");
        printf("  - pt_dim    [int]    : Point dimension\n");
        printf("  - l         [double] : Gaussian kernel parameter\n");
        printf("  - mu        [double] : Kernel matrix diagonal shift\n");
        printf("  - max_k     [int]    : Maximum global low-rank approximation rank\n");
        printf("  - ss_npt    [int]    : Number of points in the sample set\n");
        printf("  - fsai_npt  [int]    : Maximum number of nonzeros in each row of the FSAI matrix\n");
        printf("  - coord_bin [str]    : Binary file containing the coordinates, size pt_dim * npt,\n");
        printf("                         row major, each column is a point\n");
        return 255;
    } else {
        npt      = atoi(argv[1]);
        pt_dim   = atoi(argv[2]);
        l        = atof(argv[3]);
        mu       = atof(argv[4]);
        max_k    = atoi(argv[5]);
        ss_npt   = atoi(argv[6]);
        fsai_npt = atoi(argv[7]);
        coord    = (DTYPE*) malloc(sizeof(DTYPE) * npt * pt_dim);
        FILE *inf = fopen(argv[8], "rb");
        fread(coord, sizeof(DTYPE), npt * pt_dim, inf);
        fclose(inf);
    }
    printf("Test kernel: k(x, y) = exp(- %.2f * ||x-y||_2^2), diagonal shift = %.4e\n", l, mu);
    printf("%d points in %d-D, sample set size = %d\n", npt, pt_dim, ss_npt);
    printf("Maximum global low-rank approximation rank = %d\n", max_k);
    printf("Maximum number of nonzeros in each row of the FSAI matrix = %d\n", fsai_npt);
    printf("\n");
    shift_ = mu;

    kernel_eval_fptr krnl_eval = Gaussian_3D_eval_intrin_t;
    kernel_bimv_fptr krnl_bimv = Gaussian_3D_krnl_bimv_intrin_t;
    int krnl_bimv_flops = Gaussian_3D_krnl_bimv_flop;
    void *krnl_param = &l;

    // Build H2 representation
    double st, et;
    st = get_wtime_sec();
    H2Pack_p h2mat = NULL;
    int krnl_dim = 1, BD_JIT = 1;
    DTYPE reltol = 1e-8;
    H2P_dense_mat_p *pp = NULL;
    H2P_init(&h2mat, pt_dim, krnl_dim, QR_REL_NRM, &reltol);
    H2P_calc_enclosing_box(pt_dim, npt, coord, NULL, &h2mat->root_enbox);
    H2P_partition_points(h2mat, npt, coord, 0, 0);
    H2P_generate_proxy_point_ID_file(h2mat, krnl_param, krnl_eval, NULL, &pp);
    H2P_build(h2mat, pp, BD_JIT, krnl_param, krnl_eval, krnl_bimv, krnl_bimv_flops);
    et = get_wtime_sec();
    printf("H2Pack proxy point selection and build time: %.3f s\n", et - st);

    // Build AFN preconditioner
    st = get_wtime_sec();
    AFN_precond_p AFN_precond = NULL;
    AFN_precond_build(krnl_eval, krnl_param, npt, pt_dim, coord, mu, max_k, ss_npt, fsai_npt, &AFN_precond);
    et = get_wtime_sec();
    printf("AFN_precond build time: %.3lf s\n", et - st);

    // PCG test
    DTYPE CG_reltol = 1e-4, relres;
    int max_iter = 400, flag, iter;
    DTYPE *x = malloc(sizeof(DTYPE) * npt);
    DTYPE *b = malloc(sizeof(DTYPE) * npt);
    for (int i = 0; i < npt; i++)
    {
        b[i] = drand48() - 0.5;
        x[i] = 0.0;
    }
    printf("\nStarting PCG solve with AFN preconditioner...\n");
    st = get_wtime_sec();
    pcg(
        npt, CG_reltol, max_iter, 
        H2Pack_matvec, h2mat, b, AFN_precond_apply_, AFN_precond, x,
        &flag, &relres, &iter, NULL
    );
    et = get_wtime_sec();
    printf("PCG stopped after %d iterations, relres = %e, used time = %.2lf s\n\n", iter, relres, et - st);

    // Print AFN preconditioner statistics and clean up
    AFN_precond_print_stat(AFN_precond);
    AFN_precond_destroy(&AFN_precond);
    H2P_destroy(&h2mat);

    free(coord);
    free(x);
    free(b);
    return 0;
}
