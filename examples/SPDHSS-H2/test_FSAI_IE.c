#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack.h"
#include "FSAI_precond.h"
#include "../AFN_precond/IE_diag_quad.h"
#include "../AFN_precond/precond_test_utils.h"

// This file follows the test settings in the FLAM library rskelf/test/{ie_cube1.m, ie_square1.m}
// Solve an intergral equation (IE): a_i * u_i + b_i * \sum_{j=1}^n K(x_i, x_j) * c_j * u_j = f_i,
// with setting a(x_i) == 0, b(x_i) == c(x_i) == 1, K(x, y) is the Laplace kernel. 
// This test setting is the same as Example 5 in paper DOI:10.1002/cpa.21577

static DTYPE k_scale;  // Scaling factor for the kernel function
static void H2Pack_matvec_scale(const void *h2pack_, const DTYPE *b, DTYPE *x)
{
    H2Pack_p h2pack = (H2Pack_p) h2pack_;
    H2P_matvec(h2pack, b, x);
    #pragma omp simd
    for (int i = 0; i < h2pack->krnl_mat_size; i++) x[i] *= k_scale;
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    int npt, pt_dim, dim_n;
    int fsai_npt, fast_knn, max_iter;
    DTYPE mu = 0.0, dv, solve_tol, h, *coord = NULL, *krnl_param = NULL;
    kernel_eval_fptr krnl_eval = NULL;
    kernel_bimv_fptr krnl_bimv = NULL;
    int krnl_bimv_flops = 0;
    if (argc < 7)
    {
        printf("Usage: %s pt_dim dim_n fsai_npt fast_knn solve_tol max_iter\n", argv[0]);
        printf("  - pt_dim       [int]    : Point dimension, 2 or 3\n");
        printf("  - dim_n        [int]    : Number of discretization points in each dimension\n");
        printf("  - fsai_npt     [int]    : FSAI nonzeros per row\n");
        printf("  - fast_knn     [0 or 1] : If FSAI should use fast approximated KNN instead of exact KNN\n");
        printf("  - solve_tol    [double] : PCG relative residual tolerance\n");
        printf("  - max_iter     [int]    : PCG maximum iteration\n");
        return 255;
    } 
    pt_dim       = atoi(argv[1]);
    dim_n        = atoi(argv[2]);
    fsai_npt     = atoi(argv[3]);
    fast_knn     = atoi(argv[4]);
    solve_tol    = atof(argv[5]);
    max_iter     = atoi(argv[6]);
    printf("Point set: %d^%d equal-space points in [0, 1]^%d\n", dim_n, pt_dim, pt_dim);
    printf("Laplace kernel, K(x, y) = ");
    if (pt_dim == 2)
    {
        krnl_eval = Laplace_2D_eval_intrin_t;
        krnl_bimv = Laplace_2D_krnl_bimv_intrin_t;
        krnl_bimv_flops = Laplace_2D_krnl_bimv_flop;
        // Laplace_2D computes -log(|x - y|), so the scaling factor is 1 / (2 * pi)
        k_scale = 1.0 / (2.0 * M_PI);
        dv = diag_quad_2d[dim_n - 1];
        printf("-1 / (2 * pi) * log(|x - y|), K(x, x) = %e\n", dv);
    } else {
        krnl_eval = Coulomb_3D_eval_intrin_t;
        krnl_bimv = Coulomb_3D_krnl_bimv_intrin_t;
        krnl_bimv_flops = Coulomb_3D_krnl_bimv_flop;
        // Coulomb_3D computes 1 / |x - y|, so the scaling factor is 1 / (4 * pi)
        k_scale = 1.0 / (4.0 * M_PI);
        dv = diag_quad_3d[dim_n - 1];
        printf("1 / (4 * pi * |x - y|), K(x, x) = %e\n", dv);
    }
    printf("Linear system to solve: K(X, X) * x = b\n");
    printf("PCG relative residual tolerance = %.2e, max iterations = %d\n", solve_tol, max_iter);
    printf("\nFSAI nonzeros per row = %d\n", fsai_npt);
    printf("\nFast KNN for FSAI sparsity pattern = %s\n", fast_knn ? "Yes" : "No");

    // Generate equal-space grid
    h = 1.0 / (DTYPE) dim_n;
    npt = (pt_dim == 2) ? dim_n * dim_n : dim_n * dim_n * dim_n;
    n_ = npt;
    coord = (DTYPE*) malloc(sizeof(DTYPE) * pt_dim * npt);
    if (pt_dim == 2)
    {
        for (int i = 0; i < dim_n; i++)
        {
            for (int j = 0; j < dim_n; j++)
            {
                int idx = i * dim_n + j;
                coord[0 * npt + idx] = h * (i + 1);
                coord[1 * npt + idx] = h * (j + 1);
            }
        }
    } else {
        for (int i = 0; i < dim_n; i++)
        {
            for (int j = 0; j < dim_n; j++)
            {
                for (int k = 0; k < dim_n; k++)
                {
                    int idx = i * dim_n * dim_n + j * dim_n + k;
                    coord[0 * npt + idx] = h * (i + 1);
                    coord[1 * npt + idx] = h * (j + 1);
                    coord[2 * npt + idx] = h * (k + 1);
                }
            }
        }
    }

    // Scale the kernel matrix for area-weighted point interaction (what's this?)
    k_scale = k_scale / (DTYPE) npt;  
    // Since the diagonal value will also be scaled by k_scale, we need to scale it back
    dv = dv / k_scale;
    krnl_param = &dv;

    // Build H2 matrix
    double st, et;
    H2Pack_p h2mat = NULL;
    DTYPE h2_reltol = (solve_tol < 1e-8) ? solve_tol : 1e-8;
    H2mat_build(npt, pt_dim, coord, h2_reltol, krnl_eval, krnl_bimv, krnl_bimv_flops, krnl_param, &h2mat);

    // Build FSAI preconditioner
    printf("Building FSAI preconditioner...\n");
    FSAI_precond_p FSAI_precond = NULL;
    H2P_build_FSAI_precond(h2mat, fsai_npt, mu, &FSAI_precond);
    int nnz_upper = fsai_npt * (fsai_npt + 1) / 2 + fsai_npt * (npt - fsai_npt);
    DEBUG_PRINTF("FSAI G matrix nnz = %d, nnz upper bound = %d\n", FSAI_precond->G->nnz, nnz_upper);
    printf("\n\n");

    // PCG test
    printf("\nTesting FSAI preconditioner...\n");
    test_PCG(
        H2Pack_matvec_scale, (void *) h2mat, 
        (matvec_fptr) FSAI_precond_apply, (void *) FSAI_precond, 
        npt, max_iter, solve_tol
    );

    // Clean up
    free(coord);
    H2P_destroy(&h2mat);
    FSAI_precond_destroy(&FSAI_precond);
    return 0;
}
