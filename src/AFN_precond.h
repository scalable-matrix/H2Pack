#ifndef __AFN_PRECOND_H__
#define __AFN_PRECOND_H__

// Adaptive Factorized Nystrom preconditioner, ref: https://arxiv.org/pdf/2304.05460.pdf

#include "H2Pack_typedef.h"

struct AFN_precond
{
    int   is_nys, is_afn;   // Whether to use Nystrom ot AFN
    int   n;                // Size of the kernel matrix, == number of points (does not support krnl_dim > 1 yet)
    int   n1;               // Size of K11 block (== global low-rank approximation rank)
    int   n2;               // == n - n1
    int   *perm;            // Permutation array, size n
    DTYPE *px, *py;         // Size n, permuted x and y in AFN_precond_apply
    DTYPE *t1, *t2;         // Size n, intermediate vectors in AFN_precond_apply
    DTYPE *nys_U;           // Size n * n1, row major, Nystrom basis
    DTYPE *nys_M;           // Size n1, Nystrom eigenvalues + diagonal shift, then scaled
    int   *afn_G_rowptr;    // Size n2 + 1, AFN G matrix CSR row_ptr array
    int   *afn_GT_rowptr;   // Size n2 + 1, AFN G^T matrix CSR row_ptr array
    int   *afn_G_colidx;    // Size nnz, AFN G matrix CSR col_idx array
    int   *afn_GT_colidx;   // Size nnz, AFN G^T matrix CSR col_idx array
    DTYPE *afn_G_val;       // Size nnz, AFN G matrix CSR values array
    DTYPE *afn_GT_val;      // Size nnz, AFN G^T matrix CSR values array
    DTYPE *afn_invK11;      // Size n1 * n1, row major, AFN K11^{-1} matrix
    DTYPE *afn_K12;         // Size n1 * n2, row major, AFN K12 matrix

    // Timers for profiling
    int n_apply;
    double t_build, t_apply, t_rankest, t_fps, t_K11K12, t_nys;
    double t_afn, t_afn_mat, t_afn_knn, t_afn_fsai, t_afn_csr;
};
typedef struct AFN_precond  AFN_precond_s;
typedef struct AFN_precond *AFN_precond_p;

#ifdef __cplusplus
extern "C" {
#endif

// Construct an AFN preconditioner for a kernel matrix
// Input parameters:
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   krnl_param : Pointer to kernel function parameter array
//   npt        : Number of points in coord
//   pt_dim     : Dimension of each point
//   coord      : Matrix, size pt_dim-by-npt, coordinates of points
//   mu         : Scalar, diagonal shift of the kernel matrix
//   max_k      : Maximum global low-rank approximation rank 
//   ss_npt     : Number of points in the sampling set
//   fsai_npt   : Maximum number of nonzeros in each row of the FSAI matrix
// Output parameter:
//   AFN_precond_ : Pointer to an initialized AFN_precond struct
void AFN_precond_build(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const DTYPE mu, const int max_k, const int ss_npt,
    const int fsai_npt, AFN_precond_p *AFN_precond_
);

// Destroy an initialized AFN_precond struct
void AFN_precond_destroy(AFN_precond_p *AFN_precond_);

// Apply an AFN preconditioner to a vector
// Input parameters:
//   AFN_precond : Pointer to an initialized AFN_precond struct
//   x           : Input vector, size n
// Output parameter:
//   y : Output vector, size n
void AFN_precond_apply(AFN_precond_p AFN_precond, const DTYPE *x, DTYPE *y);

// Print statistics of an AFN_precond struct
void AFN_precond_print_stat(AFN_precond_p AFN_precond);

#ifdef __cplusplus
}
#endif

#endif  // End of "#ifndef __AFN_PRECOND_H__"
