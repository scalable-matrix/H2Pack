#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "pcg.h"

// Left preconditioned Conjugate Gradient for solving A * x = b
void pcg(
    const int n, const DTYPE tol, const int max_iter, 
    const matvec_fptr Ax,    const void *Ax_param,    const DTYPE *b, 
    const matvec_fptr invMx, const void *invMx_param, DTYPE *x, 
    int *flag_, DTYPE *relres_, int *iter_, DTYPE *res_vec, int print_level
)
{
    size_t vec_msize = sizeof(DTYPE) * n;
    DTYPE *r = (DTYPE*) malloc(vec_msize);
    DTYPE *z = (DTYPE*) malloc(vec_msize);
    DTYPE *p = (DTYPE*) malloc(vec_msize);
    DTYPE *s = (DTYPE*) malloc(vec_msize);
    assert(r != NULL && z != NULL && p != NULL && s != NULL);

    double st = omp_get_wtime();

    // r = b - A * x;
    Ax(Ax_param, x, r);
    #pragma omp simd
    for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
    
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

    if (print_level > 0)
    {        
        printf("\nPCG: ||b||_2 = %e, initial ||r||_2 = %e, stopping ||r||_2 = %e\n", b_2norm, r_2norm, rn_stop);
        printf("PCG: Max number of iterations: %d\n", max_iter);
        printf("Iter      Residual norm   Relative res.     \n");
    }

    int iter = 0;
    DTYPE alpha, beta, rho0, tmp, rho = 1.0;
    while (iter < max_iter && r_2norm > rn_stop)
    {
        // z = M \ r;
        if (invMx != NULL) invMx(invMx_param, r, z);
        else memcpy(z, r, vec_msize);

        // rho0 = rho;
        // rho  = r' * z;
        // beta = rho / rho0;
        rho0 = rho;
        rho  = 0.0;
        #pragma omp simd
        for (int i = 0; i < n; i++) rho += r[i] * z[i];
        beta = rho / rho0;

        // p = z + beta * p; or p = z;
        if (iter == 0) memcpy(p, z, vec_msize);
        else
        {
            #pragma omp simd
            for (int i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
        }

        // s = A * p;
        // alpha = rho / (p' * s);
        Ax(Ax_param, p, s);
        tmp = 0.0;
        #pragma omp simd
        for (int i = 0; i < n; i++) tmp += p[i] * s[i];
        alpha = rho / tmp;

        // x = x + alpha * p;
        // r = r - alpha * s;
        r_2norm = 0.0;
        #pragma omp simd
        for (int i = 0; i < n; i++) 
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * s[i];
            r_2norm += r[i] * r[i];
        }
        r_2norm = DSQRT(r_2norm);
        if (res_vec != NULL) res_vec[iter] = r_2norm;
        iter++;

        if (print_level > 0) printf("%4d      %5.4e      %5.4e\n", iter, r_2norm, r_2norm / b_2norm);
    }  // End of while
    *flag_   = (r_2norm <= rn_stop) ? 0 : 1;
    *relres_ = r_2norm / b_2norm;
    *iter_   = iter;

    double et = omp_get_wtime();
    if (print_level > 0)
    {
        if (*flag_ == 0) printf("PCG: Converged in %d iterations, %.2f seconds\n\n", iter, et - st);
        else printf("PCG: Reached maximum number of iterations, %.2f seconds\n\n", et - st);
    }

    free(r);
    free(z);
    free(p);
    free(s);
}
