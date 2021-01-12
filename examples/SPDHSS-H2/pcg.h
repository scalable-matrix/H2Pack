#ifndef __PCG_H__
#define __PCG_H__

#ifndef DTYPE
#define DTYPE double
#define DSQRT sqrt
#endif

#ifdef __cplusplus
extern "C" {
#endif

// b := A * x
typedef void (*matvec_fptr) (const void *param, const DTYPE *x, DTYPE *b);

// Left preconditioned Conjugate Gradient for solving A * x = b
// Reference: Iterative Methods for Sparse Linear System (2nd Edition), algorithm 9.1
// Input parameters:
//   n           : Size of the matrix
//   tol         : Residual vector norm tolerance
//   max_iter    : Maximum number of iterations
//   Ax          : Function pointer for calculating A * x
//   Ax_param    : Pointer to Ax function parameters
//   b           : Size n, right-hand size vector
//   invMx       : Function pointer for applying preconditioner M^{-1} * r, 
//                 NULL pointer means no preconditioning
//   invMx_param : Pointer to invMx function parameters
//   x           : Size n, initial guess vector
// Output parameters:
//   x        : Size n, solution vector
//   *flag_   : 0 == converged, 1 == not converged
//   *relres_ : Residual vector relative 2-norm at last step
//   *iter_   : Number of iterations performed
//   res_vec  : Size >= max_iter, Residual vector relative 2-norms at each iteration, 
//              NULL pointer means these values will not be recorded
void pcg(
    const int n, const DTYPE tol, const int max_iter, 
    const matvec_fptr Ax,    const void *Ax_param,    const DTYPE *b, 
    const matvec_fptr invMx, const void *invMx_param, DTYPE *x, 
    int *flag_, DTYPE *relres_, int *iter_, DTYPE *res_vec
);

#ifdef __cplusplus
}
#endif

#endif
