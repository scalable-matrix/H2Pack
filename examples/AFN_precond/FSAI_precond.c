#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "FSAI_precond.h"

// In AFN_precond.c
void FSAI_precond_build_(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int fsai_npt,
    const int n, const int pt_dim, const DTYPE *coord, const int ldc, 
    const int *coord0_idx, const int n1, const DTYPE *P, const DTYPE mu, void *h2mat, 
    int **G_rowptr_,  int **G_colidx_,  DTYPE **G_val_, 
    int **GT_rowptr_, int **GT_colidx_, DTYPE **GT_val_,
    double *t_knn_, double *t_fsai_, double *t_csr_
);
void FSAI_precond_apply_(
    const int *G_rowptr, const int *G_colidx, const DTYPE *G_val,
    const int *GT_rowptr, const int *GT_colidx, const DTYPE *GT_val,
    const int n, const DTYPE *x, DTYPE *y, DTYPE *t
);

// Build a Factoized Sparse Approximate Inverse (FSAI) preconditioner 
// for a kernel matrix K(X, X) + mu * I - P * P^T, where P is a low rank matrix
void FSAI_precond_build(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int fsai_npt,
    const int n, const int pt_dim, const DTYPE *coord, const int ldc, 
    const DTYPE mu, void *h2mat, FSAI_precond_p *FSAI_precond_
)
{
    FSAI_precond_p FSAI_precond = (FSAI_precond_p) malloc(sizeof(FSAI_precond_s));
    memset(FSAI_precond, 0, sizeof(FSAI_precond_s));

    int *coord0_idx = (int *) malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) coord0_idx[i] = i;
    FSAI_precond->n = n;
    FSAI_precond->t = (DTYPE *) malloc(sizeof(DTYPE) * n);
    FSAI_precond_build_(
        krnl_eval, krnl_param, fsai_npt, 
        n, pt_dim, coord, ldc, 
        coord0_idx, 0, NULL, mu, h2mat,
        &FSAI_precond->G_rowptr,  &FSAI_precond->G_colidx,  &FSAI_precond->G_val,
        &FSAI_precond->GT_rowptr, &FSAI_precond->GT_colidx, &FSAI_precond->GT_val,
        &FSAI_precond->t_knn, &FSAI_precond->t_fsai, &FSAI_precond->t_csr
    );

    free(coord0_idx);
    *FSAI_precond_ = FSAI_precond;
}

// Apply a FSAI preconditioner to a vector
void FSAI_precond_apply(FSAI_precond_p FSAI_precond, const DTYPE *x, DTYPE *y)
{
    FSAI_precond_apply_(
        FSAI_precond->G_rowptr,  FSAI_precond->G_colidx,  FSAI_precond->G_val,
        FSAI_precond->GT_rowptr, FSAI_precond->GT_colidx, FSAI_precond->GT_val,
        FSAI_precond->n, x, y, FSAI_precond->t
    );
}

// Destroy an initialized FSAI_precond struct
void FSAI_precond_destroy(FSAI_precond_p *FSAI_precond_)
{
    FSAI_precond_p p = *FSAI_precond_;
    if (p == NULL) return;
    free(p->t);
    free(p->G_rowptr);
    free(p->G_colidx);
    free(p->G_val);
    free(p->GT_rowptr);
    free(p->GT_colidx);
    free(p->GT_val);
    free(p);
}