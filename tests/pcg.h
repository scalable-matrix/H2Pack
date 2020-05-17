
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
)
{
    size_t vec_msize = sizeof(DTYPE) * n;
    DTYPE *r = (DTYPE*) malloc(vec_msize);
    DTYPE *z = (DTYPE*) malloc(vec_msize);
    DTYPE *p = (DTYPE*) malloc(vec_msize);
    DTYPE *s = (DTYPE*) malloc(vec_msize);
    assert(r != NULL && z != NULL && p != NULL && s != NULL);

    // r = b - A * x;
    Ax(Ax_param, x, r);
    #pragma omp simd
    for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
    // z = M \ r;
    if (invMx != NULL) invMx(invMx_param, r, z);
    else memcpy(z, r, vec_msize);
    // p = z;
    memcpy(p, z, vec_msize);
    // rho = r' * z;
    DTYPE rho = 0.0;
    #pragma omp simd
    for (int i = 0; i < n; i++) rho += r[i] * z[i];
    // b_2norm = norm(b, 2);
    // r_2norm = norm(r, 2);
    // rn_stop = b_2norm * tol;
    DTYPE b_2norm = 0.0, r_2norm = 0.0, rn_stop;
    #pragma omp simd
    for (int i = 0; i < n; i++)
    {
        b_2norm += b[i] * b[i];
        r_2norm += r[i] * r[i];
    }
    b_2norm = DSQRT(b_2norm);
    r_2norm = DSQRT(r_2norm);
    rn_stop = b_2norm * tol;

    int iter = 0;
    DTYPE alpha, beta, rho0, tmp;
    while (iter < max_iter && r_2norm > rn_stop)
    {
        // s = A * p;
        Ax(Ax_param, p, s);
        // alpha = rho / (p' * s);
        tmp = 0.0;
        #pragma omp simd
        for (int i = 0; i < n; i++) tmp += p[i] * s[i];
        alpha = rho / tmp;

        // x = x + alpha * p;
        // r = r - alpha * s;
        #pragma omp simd
        for (int i = 0; i < n; i++) 
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * s[i];
        }

        // z = M \ r;
        if (invMx != NULL) invMx(invMx_param, r, z);
        else memcpy(z, r, vec_msize);
        // rho0 = rho;
        // rho  = r' * z;
        rho0 = rho;
        rho  = 0.0;
        #pragma omp simd
        for (int i = 0; i < n; i++) rho += r[i] * z[i];

        // beta = rho / rho0;
        // p = z + beta * p;
        beta = rho / rho0;
        #pragma omp simd
        for (int i = 0; i < n; i++) p[i] = z[i] + beta * p[i];

        // r_2norm = norm(r, 2);
        // resvec(iter) = r_2norm;
        // iter = iter + 1;
        r_2norm = 0.0;
        #pragma omp simd
        for (int i = 0; i < n; i++) r_2norm += r[i] * r[i];
        r_2norm = DSQRT(r_2norm);
        if (res_vec != NULL) res_vec[iter] = r_2norm;
        iter++;

        //printf("%e\n", r_2norm / b_2norm);
    }  // End of while
    *flag_   = (r_2norm <= rn_stop) ? 0 : 1;
    *relres_ = r_2norm / b_2norm;
    *iter_   = iter;

    free(r);
    free(z);
    free(p);
    free(s);
}

