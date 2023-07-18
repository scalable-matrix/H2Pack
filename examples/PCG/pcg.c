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

    double st, st0, et, et0;
    double t_Ax = 0, t_invMx = 0, t_vec = 0;
    st0 = omp_get_wtime();

    // r = b - A * x;
    st = omp_get_wtime();
    Ax(Ax_param, x, r);
    et = omp_get_wtime();
    t_Ax += et - st;

    st = omp_get_wtime();
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
    et = omp_get_wtime();
    t_vec += et - st;

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
        st = omp_get_wtime();
        if (invMx != NULL) invMx(invMx_param, r, z);
        else memcpy(z, r, vec_msize);
        et = omp_get_wtime();
        t_invMx += et - st;

        // rho0 = rho;
        // rho  = r' * z;
        // beta = rho / rho0;
        st = omp_get_wtime();
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
        et = omp_get_wtime();
        t_vec += et - st;

        // s = A * p;
        // alpha = rho / (p' * s);
        st = omp_get_wtime();
        Ax(Ax_param, p, s);
        et = omp_get_wtime();
        t_Ax += et - st;

        st = omp_get_wtime();
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
        et = omp_get_wtime();
        t_vec += et - st;

        if (print_level > 0) printf("%4d      %5.4e      %5.4e\n", iter, r_2norm, r_2norm / b_2norm);
    }  // End of while
    *flag_   = (r_2norm <= rn_stop) ? 0 : 1;
    *relres_ = r_2norm / b_2norm;
    *iter_   = iter;
    et0 = omp_get_wtime();

    // Sanity check
    Ax(Ax_param, x, r);
    r_2norm = 0.0;
    #pragma omp simd
    for (int i = 0; i < n; i++)
    {
        r[i] = b[i] - r[i];
        r_2norm += r[i] * r[i];
    }
    r_2norm = DSQRT(r_2norm);

    if (print_level > 0)
    {
        printf("PCG: Final relres = %e\n", r_2norm / b_2norm);
        if (*flag_ == 0) printf("PCG: Converged in %d iterations, %.2f seconds\n", iter, et0 - st0);
        else printf("PCG: Reached maximum number of iterations, %.2f seconds\n", et0 - st0);
        printf("PCG: time for Ax, invMx, vector operations: %.2f, %.2f, %.2f seconds\n\n", t_Ax, t_invMx, t_vec);
    }

    free(r);
    free(z);
    free(p);
    free(s);
}
