#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack.h"
#include "LRD_precond.h"

// Evaluate a kernel matrix with OpenMP parallelization
extern void H2P_eval_kernel_matrix_OMP(
    const void *krnl_param, kernel_eval_fptr krnl_eval, const int krnl_dim, 
    H2P_dense_mat_t x_coord, H2P_dense_mat_t y_coord, H2P_dense_mat_t kernel_mat
);

// Construct a LRD_precond from a H2Pack structure using Nystrom method with random sampling
void H2P_build_LRD_precond(H2Pack_t h2pack, const int rank, const DTYPE shift, LRD_precond_t *precond_)
{
    LRD_precond_t precond = (LRD_precond_t) malloc(sizeof(LRD_precond_s));
    assert(precond != NULL);

    int n_point  = h2pack->n_point;
    int n_thread = h2pack->n_thread;
    int pt_dim   = h2pack->pt_dim;
    int mat_size = h2pack->krnl_mat_size;
    int krnl_dim = h2pack->krnl_dim;
    int nrow     = rank * krnl_dim;

    int *flag = (int*) malloc(sizeof(int) * n_point);
    ASSERT_PRINTF(flag != NULL, "Failed to allocate work array of size %d for LRD preconditioner\n", n_point);
    memset(flag, 0, sizeof(int) * n_point);
    H2P_int_vec_t   skel_idx;
    H2P_dense_mat_t coord_all, coord_skel;
    H2P_int_vec_init(&skel_idx, rank);
    H2P_dense_mat_init(&coord_all,  pt_dim, n_point);
    H2P_dense_mat_init(&coord_skel, pt_dim, n_point);
    memcpy(coord_all->data,  h2pack->coord, sizeof(DTYPE) * pt_dim * n_point);
    memcpy(coord_skel->data, h2pack->coord, sizeof(DTYPE) * pt_dim * n_point);
    int cnt = 0;
    while (cnt < rank)
    {
        int idx = rand() % n_point;
        if (flag[idx] == 0)
        {
            flag[idx] = 1;
            skel_idx->data[cnt] = idx;
            cnt++;
        }
    }
    skel_idx->length = rank;
    H2P_dense_mat_select_columns(coord_skel, skel_idx);
    
    H2P_dense_mat_t L, Ut, tmp;
    H2P_dense_mat_init(&L,   nrow, nrow);
    H2P_dense_mat_init(&Ut,  nrow, mat_size);
    H2P_dense_mat_init(&tmp, nrow, nrow);
    // L   = kernel({coord(idx, :), coord(idx, :)});
    // Ut  = kernel({coord(idx, :), coord});
    H2P_eval_kernel_matrix_OMP(h2pack->krnl_param, h2pack->krnl_eval, krnl_dim, coord_skel, coord_skel, L);
    H2P_eval_kernel_matrix_OMP(h2pack->krnl_param, h2pack->krnl_eval, krnl_dim, coord_skel, coord_all,  Ut);
    // L   = chol(L, 'lower');
    // Ut  = linsolve(L, Ut, struct('LT', true));
    LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', nrow, L->data, L->ld);
    CBLAS_TRSM(
        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
        nrow, mat_size, 1.0, L->data, L->ld, Ut->data, Ut->ld
    );
    // tmp = Ut * Ut' + shift * eye(lr_rank);
    CBLAS_GEMM(
        CblasRowMajor, CblasNoTrans, CblasTrans, nrow, nrow, mat_size,
        1.0, Ut->data, Ut->ld, Ut->data, Ut->ld, 0.0, tmp->data, tmp->ld
    );
    for (int i = 0; i < nrow; i++)
        tmp->data[i * tmp->ld + i] += shift;
    // tmp = chol(tmp, 'lower');
    // Ut  = linsolve(tmp, Ut, struct('LT', true));
    LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', nrow, tmp->data, tmp->ld);
    CBLAS_TRSM(
        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
        nrow, mat_size, 1.0, tmp->data, tmp->ld, Ut->data, Ut->ld
    );

    DTYPE *Ut_ = (DTYPE*) malloc(sizeof(DTYPE) * nrow * mat_size);
    DTYPE *workbuf = (DTYPE*) malloc(sizeof(DTYPE) * nrow);
    ASSERT_PRINTF(
        Ut_ != NULL && workbuf != NULL, 
        "Failed to allocate matrix of size %d * %d for LRD preconditioner\n", nrow, mat_size+1
    );
    memcpy(Ut_, Ut->data, sizeof(DTYPE) * nrow * mat_size);

    H2P_dense_mat_destroy(coord_all);
    H2P_dense_mat_destroy(coord_skel);
    H2P_dense_mat_destroy(L);
    H2P_dense_mat_destroy(Ut);
    H2P_dense_mat_destroy(tmp);
    H2P_int_vec_destroy(skel_idx);
    free(coord_all);
    free(coord_skel);
    free(skel_idx);
    precond->mat_size = mat_size;
    precond->rank     = nrow;
    precond->shift    = shift;
    precond->Ut       = Ut_;
    precond->workbuf  = workbuf;
    *precond_ = precond;
}

// Apply LRD preconditioner, x := M_{BJP}^{-1} * b
void apply_LRD_precond(LRD_precond_t precond, const DTYPE *b, DTYPE *x)
{
    int   mat_size  = precond->mat_size;
    int   rank      = precond->rank;
    DTYPE shift     = precond->shift;
    DTYPE *Ut       = precond->Ut;
    DTYPE *workbuf  = precond->workbuf;
    // x = 1 / shift * (b - Ut' * (Ut * b));
    memcpy(x, b, sizeof(DTYPE) * mat_size);
    CBLAS_GEMV(CblasRowMajor, CblasNoTrans, rank, mat_size,  1.0, Ut, mat_size,       b, 1, 0.0, workbuf, 1);
    CBLAS_GEMV(CblasRowMajor, CblasTrans,   rank, mat_size, -1.0, Ut, mat_size, workbuf, 1, 1.0, x,       1);
    DTYPE inv_shift = 1.0 / shift;
    #pragma omp simd
    for (int i = 0; i < mat_size; i++) x[i] *= inv_shift;
}

// Destroy a LRD_precond structure
void free_LRD_precond(LRD_precond_t precond)
{
    free(precond->Ut);
    free(precond->workbuf);
    free(precond);
}
