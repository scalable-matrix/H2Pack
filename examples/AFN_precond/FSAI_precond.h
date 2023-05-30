#ifndef __FSAI_PRECOND_H__
#define __FSAI_PRECOND_H__

#include "H2Pack.h"

struct FSAI_precond
{
    int n;              // Size of the kernel matrix, == number of points
    int   *G_rowptr;    // Size n2 + 1, AFN G matrix CSR row_ptr array
    int   *GT_rowptr;   // Size n2 + 1, AFN G^T matrix CSR row_ptr array
    int   *G_colidx;    // Size nnz, AFN G matrix CSR col_idx array
    int   *GT_colidx;   // Size nnz, AFN G^T matrix CSR col_idx array
    DTYPE *G_val;       // Size nnz, AFN G matrix CSR values array
    DTYPE *GT_val;      // Size nnz, AFN G^T matrix CSR values array
    DTYPE *t;           // Size n, intermediate vectors in Nystrom_precond_apply

    // Timers
    double t_knn, t_fsai, t_csr;
};
typedef struct FSAI_precond  FSAI_precond_s;
typedef struct FSAI_precond* FSAI_precond_p;

#ifdef __cplusplus
extern "C" {
#endif

// Build a Factoized Sparse Approximate Inverse (FSAI) preconditioner 
// for a kernel matrix K(X, X) + mu * I - P * P^T, where P is a low rank matrix
// Input parameters:
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   krnl_param : Pointer to kernel function parameter array
//   fsai_npt   : Maximum number of nonzeros in each row of the FSAI matrix
//   n, pt_dim  : Number of points and point dimension
//   coord      : Size pt_dim * ldc, row major, each column is a point coordinate
//   ldc        : Leading dimension of coord, >= n
//   mu         : Diagonal shift
//   h2mat      : Optional, pointer to an initialized H2Pack struct, used for FSAI KNN search
// Output parameter:
//   *FSAI_precond_  : Pointer to an initialized FSAI preconditioner
void FSAI_precond_build(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int fsai_npt,
    const int n, const int pt_dim, const DTYPE *coord, const int ldc, 
    const DTYPE mu, void *h2mat, FSAI_precond_p *FSAI_precond_
);

// Apply a FSAI preconditioner to a vector
void FSAI_precond_apply(FSAI_precond_p FSAI_precond, const DTYPE *x, DTYPE *y);

// Destroy an initialized FSAI_precond struct
void FSAI_precond_destroy(FSAI_precond_p *FSAI_precond_);

#ifdef __cplusplus
}
#endif

#endif
