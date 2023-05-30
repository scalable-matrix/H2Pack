#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "Nys_precond.h"
#include "H2Pack_utils.h"

// In AFN_precond.c
void Nys_precond_build_(
    const DTYPE mu, const int n1, const int n2, DTYPE *K11, 
    DTYPE *K12, DTYPE **nys_M_, DTYPE **nys_U_
);
void Nys_precond_apply_(
    const int n1, const int n, const DTYPE *nys_M, const DTYPE *nys_U, 
    const DTYPE *x, DTYPE *y, DTYPE *t
);

// Build a randomize Nystrom preconditioner for a kernel matrix
void Nys_precond_build(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const DTYPE mu, const int nys_k, Nys_precond_p *Nys_precond_
)
{
    Nys_precond_p Nys_precond = (Nys_precond_p) malloc(sizeof(Nys_precond_s));
    memset(Nys_precond, 0, sizeof(Nys_precond_s));

    // 1. Randomly select nys_k points from npt points
    int n = npt, n1 = nys_k, n2 = npt - nys_k;
    int *perm = (int *) malloc(sizeof(int) * n);
    uint8_t *flag = (uint8_t *) malloc(sizeof(uint8_t) * n);
    DTYPE *coord_perm = (DTYPE *) malloc(sizeof(DTYPE) * npt * pt_dim);
    H2P_rand_sample(npt, nys_k, perm, flag);
    memset(flag, 0, sizeof(uint8_t) * n);
    for (int i = 0; i < n1; i++) flag[perm[i]] = 1;
    int idx = n1;
    for (int i = 0; i < n; i++)
        if (flag[i] == 0) perm[idx++] = i;
    H2P_gather_matrix_columns(coord, npt, coord_perm, npt, pt_dim, perm, npt);

    // 2. Build K11 and K12 blocks
    DTYPE *coord_n1 = coord_perm;
    DTYPE *coord_n2 = coord_perm + n1;
    DTYPE *K11 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    DTYPE *K12 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n2);
    int n_thread = omp_get_max_threads();
    ASSERT_PRINTF(
        K11 != NULL && K12 != NULL,
        "Failed to allocate Nystrom preconditioner K11/K12 buffers\n"
    );
    H2P_eval_kernel_matrix_OMP(
        krnl_eval, krnl_param, 
        coord_n1, n, n1, coord_n1, n, n1, 
        K11, n1, n_thread
    );
    H2P_eval_kernel_matrix_OMP(
        krnl_eval, krnl_param, 
        coord_n1, n, n1, coord_n2, n, n2, 
        K12, n2, n_thread
    );
    free(coord_perm);
    free(flag);

    // 3. Build U and M matrices
    Nys_precond->n    = n;
    Nys_precond->n1   = n1;
    Nys_precond->perm = perm;
    Nys_precond->t    = (DTYPE*) malloc(sizeof(DTYPE) * n);
    Nys_precond->px   = (DTYPE*) malloc(sizeof(DTYPE) * n);
    Nys_precond->py   = (DTYPE*) malloc(sizeof(DTYPE) * n);
    Nys_precond_build_(mu, n1, n2, K11, K12, &Nys_precond->M, &Nys_precond->U);
    *Nys_precond_ = Nys_precond;
}

// Apply a Nystrom preconditioner to a vector
void Nys_precond_apply(Nys_precond_p Nys_precond, const DTYPE *x, DTYPE *y)
{
    int n = Nys_precond->n, n1 = Nys_precond->n1;
    int *perm = Nys_precond->perm;
    DTYPE *px = Nys_precond->px, *py = Nys_precond->py, *t1 = Nys_precond->t;
    DTYPE *M = Nys_precond->M, *U = Nys_precond->U;
    for (int i = 0; i < n; i++) px[i] = x[perm[i]];
    Nys_precond_apply_(n1, n, M, U, px, py, t1);
    for (int i = 0; i < n; i++) y[perm[i]] = py[i];
}

// Destroy an initialized Nys_precond struct
void Nys_precond_destroy(Nys_precond_p *Nys_precond_)
{
    Nys_precond_p p = *Nys_precond_;
    if (p == NULL) return;
    free(p->perm);
    free(p->M);
    free(p->U);
    free(p->t);
    free(p);
}
