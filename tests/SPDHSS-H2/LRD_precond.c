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
    H2P_dense_mat_p x_coord, H2P_dense_mat_p y_coord, H2P_dense_mat_p kernel_mat
);

// Construct a LRD_precond from a H2Pack structure using Nystrom method with random sampling
void H2P_build_LRD_precond(H2Pack_p h2pack, const int rank, const DTYPE shift, LRD_precond_p *precond_)
{
    LRD_precond_p precond = (LRD_precond_p) malloc(sizeof(LRD_precond_s));
    assert(precond != NULL);

    int n_point  = h2pack->n_point;
    int n_thread = h2pack->n_thread;
    int pt_dim   = h2pack->pt_dim;
    int mat_size = h2pack->krnl_mat_size;
    int krnl_dim = h2pack->krnl_dim;
    int nrow     = rank * krnl_dim;

    double st = get_wtime_sec();

    int *flag = (int*) malloc(sizeof(int) * n_point);
    ASSERT_PRINTF(flag != NULL, "Failed to allocate work array of size %d for LRD preconditioner\n", n_point);
    memset(flag, 0, sizeof(int) * n_point);
    H2P_int_vec_p   skel_idx;
    H2P_dense_mat_p coord_all, coord_skel;
    H2P_int_vec_init(&skel_idx, rank);
    H2P_dense_mat_init(&coord_all,  pt_dim, n_point);
    H2P_dense_mat_init(&coord_skel, pt_dim, n_point);
    memcpy(coord_all->data,  h2pack->coord, sizeof(DTYPE) * pt_dim * n_point);
    memcpy(coord_skel->data, h2pack->coord, sizeof(DTYPE) * pt_dim * n_point);
    // Not working?
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
    //for (int i = 0; i < rank; i++) skel_idx->data[i] = i * n_point / rank;
    skel_idx->length = rank;
    H2P_dense_mat_select_columns(coord_skel, skel_idx);
    
    int info;
    H2P_dense_mat_p L, Ut, tmp;
    H2P_dense_mat_init(&L,   nrow, nrow);
    H2P_dense_mat_init(&Ut,  nrow, mat_size);
    H2P_dense_mat_init(&tmp, nrow, mat_size);
    // L   = kernel({coord(idx, :), coord(idx, :)});
    // Ut  = kernel({coord(idx, :), coord});
    H2P_eval_kernel_matrix_OMP(h2pack->krnl_param, h2pack->krnl_eval, krnl_dim, coord_skel, coord_skel, L);
    H2P_eval_kernel_matrix_OMP(h2pack->krnl_param, h2pack->krnl_eval, krnl_dim, coord_skel, coord_all,  Ut);
    // [V, D] = eig(S);
    // Ut  = inv(D) * V' * Ut;
    info = LAPACK_SYEVD(LAPACK_ROW_MAJOR, 'V', 'L', nrow, L->data, L->ld, tmp->data);
    ASSERT_PRINTF(info == 0, "Eigen decomposition for S matrix failed and returned %d, matrix size = %d\n", info, nrow);
    DTYPE *V = L->data, *D = tmp->data;
    DTYPE max_diag = 0.0;
    for (int i = 0; i < nrow; i++) 
    {
        if (D[i] < 0.0) WARNING_PRINTF("S matrix %d-th eigenvalue = %e < 0!\n", i+1, D[i]);
        if (D[i] > max_diag) max_diag = D[i];
    }
    for (int i = 0; i < nrow; i++)
        D[i] = (D[i] >= 1e-10 * max_diag) ? D[i] = 1.0 / sqrt(D[i]) : 0.0;
    #pragma omp parallel for
    for (int i = 0; i < nrow; i++)
    {
        DTYPE *V_i = V + i * nrow;
        #pragma omp simd
        for (int j = 0; j < nrow; j++) V_i[j] *= D[j];
    }
    CBLAS_GEMM(
        CblasRowMajor, CblasTrans, CblasNoTrans, nrow, mat_size, nrow,
        1.0, V, nrow, Ut->data, mat_size, 0.0, tmp->data, mat_size
    );
    #pragma omp parallel for
    for (int i = 0; i < nrow * mat_size; i++) Ut->data[i] = tmp->data[i];
    // tmp = Ut * Ut' + shift * eye(lr_rank);
    H2P_dense_mat_resize(tmp, nrow, nrow);
    CBLAS_GEMM(
        CblasRowMajor, CblasNoTrans, CblasTrans, nrow, nrow, mat_size,
        1.0, Ut->data, Ut->ld, Ut->data, Ut->ld, 0.0, tmp->data, tmp->ld
    );
    for (int i = 0; i < nrow; i++)
        tmp->data[i * tmp->ld + i] += shift;
    // tmp = chol(tmp, 'lower');
    // Ut  = linsolve(tmp, Ut, struct('LT', true));
    info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', nrow, tmp->data, tmp->ld);
    ASSERT_PRINTF(info == 0, "Cholesky decomposition failed and return %d, matrix size = %d\n", info, nrow);
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

    H2P_dense_mat_destroy(&coord_all);
    H2P_dense_mat_destroy(&coord_skel);
    H2P_dense_mat_destroy(&L);
    H2P_dense_mat_destroy(&Ut);
    H2P_dense_mat_destroy(&tmp);
    H2P_int_vec_destroy(&skel_idx);
    free(coord_all);
    free(coord_skel);
    free(skel_idx);

    double et = get_wtime_sec();

    precond->mat_size = mat_size;
    precond->rank     = nrow;
    precond->shift    = shift;
    precond->Ut       = Ut_;
    precond->workbuf  = workbuf;
    precond->t_build  = et - st;
    precond->t_apply  = 0.0;
    precond->n_apply  = 0;
    precond->mem_MB   = (double) (sizeof(DTYPE) * nrow * (mat_size + 1)) / 1048576.0;
    *precond_ = precond;
}

// Apply LRD preconditioner, x := M_{LRD}^{-1} * b
void LRD_precond_apply(LRD_precond_p precond, const DTYPE *b, DTYPE *x)
{
    if (precond == NULL) return;
    int   mat_size  = precond->mat_size;
    int   rank      = precond->rank;
    DTYPE shift     = precond->shift;
    DTYPE *Ut       = precond->Ut;
    DTYPE *workbuf  = precond->workbuf;

    double st = get_wtime_sec();
    // x = 1 / shift * (b - Ut' * (Ut * b));
    memcpy(x, b, sizeof(DTYPE) * mat_size);
    CBLAS_GEMV(CblasRowMajor, CblasNoTrans, rank, mat_size,  1.0, Ut, mat_size,       b, 1, 0.0, workbuf, 1);
    CBLAS_GEMV(CblasRowMajor, CblasTrans,   rank, mat_size, -1.0, Ut, mat_size, workbuf, 1, 1.0, x,       1);
    DTYPE inv_shift = 1.0 / shift;
    #pragma omp simd
    for (int i = 0; i < mat_size; i++) x[i] *= inv_shift;
    double et = get_wtime_sec();
    precond->t_apply += et - st;
    precond->n_apply++;
}

// Destroy a LRD_precond structure
void LRD_precond_destroy(LRD_precond_p *precond_)
{
    LRD_precond_p precond = *precond_;
    if (precond == NULL) return;
    free(precond->Ut);
    free(precond->workbuf);
    free(precond);
    *precond_ = NULL;
}

// Print statistic info of a FSAI_precond structure
void LRD_precond_print_stat(LRD_precond_p precond)
{
    if (precond == NULL) return;
    printf(
        "LRD precond used memory = %.2lf MB, build time = %.3lf sec, apply avg time = %.3lf sec\n", 
        precond->mem_MB, precond->t_build, precond->t_apply / (double) precond->n_apply
    );
}
