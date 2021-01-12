#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "H2Pack.h"

#include "pcg.h"
#include "block_jacobi_precond.h"
#include "LRD_precond.h"
#include "FSAI_precond.h"

static DTYPE shift_;

void H2Pack_matvec(const void *h2pack_, const DTYPE *b, DTYPE *x)
{
    H2Pack_p h2pack = (H2Pack_p) h2pack_;
    H2P_matvec(h2pack, b, x);
    #pragma omp simd
    for (int i = 0; i < h2pack->krnl_mat_size; i++) x[i] += shift_ * b[i];
}

void block_jacobi_precond(const void *precond_, const DTYPE *b, DTYPE *x)
{
    block_jacobi_precond_p precond = (block_jacobi_precond_p) precond_;
    block_jacobi_precond_apply(precond, b, x);
}

void LRD_precond(const void *precond_, const DTYPE *b, DTYPE *x)
{
    LRD_precond_p precond = (LRD_precond_p) precond_;
    LRD_precond_apply(precond, b, x);
}

void FSAI_precond(const void *precond_, const DTYPE *b, DTYPE *x)
{
    FSAI_precond_p precond = (FSAI_precond_p) precond_;
    FSAI_precond_apply(precond, b, x);
}

void HSS_ULV_Chol_precond(const void *hssmat_, const DTYPE *b, DTYPE *x)
{
    H2Pack_p hssmat = (H2Pack_p) hssmat_;
    H2P_HSS_ULV_Cholesky_solve(hssmat, 3, b, x);
}

// Test preconditioned conjugate gradient solver with different preconditioner
void pcg_tests(
    const int krnl_mat_size, H2Pack_p h2mat, H2Pack_p hssmat, const DTYPE shift, 
    const int max_rank, const int max_iter, const DTYPE CG_tol, const int method
)
{
    DTYPE *x = malloc(sizeof(DTYPE) * krnl_mat_size);
    DTYPE *y = malloc(sizeof(DTYPE) * krnl_mat_size);
    assert(x != NULL && y != NULL);


    //  Random right hand side vector
    srand48(2);
    for (int i = 0; i < krnl_mat_size; i++) y[i] = 0.5 - drand48();

    int flag, iter;
    DTYPE relres;
    double st, et;

    shift_ = shift;

    if (method == 0 || method == 1)
    {
        printf("\nStarting PCG solve without preconditioner...\n");
        memset(x, 0, sizeof(DTYPE) * krnl_mat_size);
        st = get_wtime_sec();
        pcg(
            krnl_mat_size, CG_tol, max_iter, 
            H2Pack_matvec, h2mat, y, NULL, NULL, x,
            &flag, &relres, &iter, NULL
        );
        et = get_wtime_sec();
        printf("PCG stopped after %d iterations, relres = %e, used time = %.2lf sec\n", iter, relres, et - st);
    }

    if (method == 0 || method == 2)
    {
        printf("\nConstructing block Jacobi preconditioner...\n");
        block_jacobi_precond_p bj_precond;
        H2P_build_block_jacobi_precond(h2mat, shift, &bj_precond);
        printf("Starting PCG solve with block Jacobi preconditioner...\n");
        memset(x, 0, sizeof(DTYPE) * krnl_mat_size);
        st = get_wtime_sec();
        pcg(
            krnl_mat_size, CG_tol, max_iter, 
            H2Pack_matvec, h2mat, y, block_jacobi_precond, bj_precond, x,
            &flag, &relres, &iter, NULL
        );
        et = get_wtime_sec();
        printf("PCG stopped after %d iterations, relres = %e, used time = %.2lf sec\n", iter, relres, et - st);
        block_jacobi_precond_print_stat(bj_precond);
        block_jacobi_precond_destroy(&bj_precond);
    }

    if (method == 0 || method == 3)
    {
        printf("\nConstructing LRD preconditioner...\n");
        LRD_precond_p lrd_precond;
        H2P_build_LRD_precond(h2mat, max_rank, shift, &lrd_precond);
        printf("Starting PCG solve with LRD preconditioner...\n");
        memset(x, 0, sizeof(DTYPE) * krnl_mat_size);
        st = get_wtime_sec();
        pcg(
            krnl_mat_size, CG_tol, max_iter, 
            H2Pack_matvec, h2mat, y, LRD_precond, lrd_precond, x,
            &flag, &relres, &iter, NULL
        );
        et = get_wtime_sec();
        printf("PCG stopped after %d iterations, relres = %e, used time = %.2lf sec\n", iter, relres, et - st);
        LRD_precond_print_stat(lrd_precond);
        LRD_precond_destroy(&lrd_precond);
    }

    if (method == 0 || method == 4)
    {
        printf("\nConstructing FSAI preconditioner...\n");
        FSAI_precond_p fsai_precond;
        H2P_build_FSAI_precond(h2mat, max_rank, shift, &fsai_precond);
        printf("Starting PCG solve with FSAI preconditioner...\n");
        memset(x, 0, sizeof(DTYPE) * krnl_mat_size);
        st = get_wtime_sec();
        pcg(
            krnl_mat_size, CG_tol, max_iter, 
            H2Pack_matvec, h2mat, y, FSAI_precond, fsai_precond, x,
            &flag, &relres, &iter, NULL
        );
        et = get_wtime_sec();
        printf("PCG stopped after %d iterations, relres = %e, used time = %.2lf sec\n", iter, relres, et - st);
        FSAI_precond_print_stat(fsai_precond);
        FSAI_precond_destroy(&fsai_precond);
    }

    if (method == 0 || method == 5)
    {
        printf("\nStarting PCG solve with SPDHSS preconditioner...\n");
        memset(x, 0, sizeof(DTYPE) * krnl_mat_size);
        st = get_wtime_sec();
        pcg(
            krnl_mat_size, CG_tol, max_iter, 
            H2Pack_matvec, h2mat, y, HSS_ULV_Chol_precond, hssmat, x,
            &flag, &relres, &iter, NULL
        );
        et = get_wtime_sec();
        printf("PCG stopped after %d iterations, relres = %e, used time = %.2lf sec\n", iter, relres, et - st);
    }

    free(x);
    free(y);
}
