#ifndef __NYS_PRECOND_H__
#define __NYS_PRECOND_H__

#include "H2Pack.h"

struct Nys_precond
{
    int n;          // Size of the kernel matrix, == number of points  
    int n1;         // Size of K11 block (== Nystrom approximation rank)
    int *perm;      // Permutation array, size n
    DTYPE *t;       // Size n, intermediate vectors in Nystrom_precond_apply
    DTYPE *px, *py; // Size n, permuted x and y in Nystrom_precond_apply
    DTYPE *U;       // Size n * n1, row major, Nystrom basis
    DTYPE *M;       // Size n1, Nystrom eigenvalues + diagonal shift, then scaled
};
typedef struct Nys_precond  Nys_precond_s;
typedef struct Nys_precond* Nys_precond_p;

#ifdef __cplusplus
extern "C" {
#endif

// Build a randomize Nystrom preconditioner for a kernel matrix
// Input parameters:
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   krnl_param : Pointer to kernel function parameter array
//   npt        : Number of points in coord
//   pt_dim     : Dimension of each point
//   coord      : Matrix, size pt_dim-by-npt, coordinates of points
//   mu         : Scalar, diagonal shift of the kernel matrix
//   nys_k      : Nystrom approximation rank
// Output parameter:
//   Nys_precond_ : Pointer to an initialized Nys_precond struct
void Nys_precond_build(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const DTYPE mu, const int nys_k, Nys_precond_p *Nys_precond_
);

// Apply a Nystrom preconditioner to a vector
void Nys_precond_apply(Nys_precond_p Nys_precond, const DTYPE *x, DTYPE *y);

// Destroy an initialized Nys_precond struct
void Nys_precond_destroy(Nys_precond_p *Nys_precond_);

#ifdef __cplusplus
}
#endif

#endif
