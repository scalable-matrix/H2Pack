#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "FSAI_precond.h"
#include "../AFN_precond/precond_test_utils.h"
#include "../PCG/pcg.h"

int main(int argc, char **argv)
{
    // Parse command line arguments
    int kid, npt, pt_dim, fsai_npt, fast_knn;
    DTYPE mu, kp, *coord = NULL;
    void *krnl_param = &kp;
    kernel_eval_fptr krnl_eval = NULL;
    kernel_bimv_fptr krnl_bimv = NULL;
    int krnl_bimv_flops = 0;
    if (argc < 8)
    {
        printf("Usage: %s kid kp mu npt pt_dim fsai_npt fast_knn coord_bin\n", argv[0]);
        printf("  - kid       [int]    : Kernel function ID\n");
        printf("                         1 - Gaussian    k(x, y) = exp(-l * |x-y|^2)\n");
        printf("                         2 - Exponential k(x, y) = exp(-l * |x-y|)\n");
        printf("                         3 - 3/2 Matern  k(x, y) = (1 + l*k) * exp(-l*k), k = sqrt(3) * |x-y|\n");
        printf("                         4 - 5/2 Matern  k(x, y) = (1 + l*k + l^2*k^2/3) * exp(-l*k), k = sqrt(5) * |x-y|\n");
        printf("  - kp        [double] : Kernel function parameter (l)\n");
        printf("  - mu        [double] : Kernel matrix diagonal shift\n");
        printf("  - npt       [int]    : Number of points\n");
        printf("  - pt_dim    [int]    : Point dimension\n");
        printf("  - fsai_npt  [int]    : Maximum number of nonzeros in each row of the AFN FSAI matrix\n");
        printf("  - fast_knn  [0 or 1] : If AFN FSAI should use fast approximated KNN instead of exact KNN\n");
        printf("  - coord_bin [str]    : (Optional) Binary file containing the coordinates, size pt_dim * npt,\n");
        printf("                         row major, each column is a point coordinate\n");
        return 255;
    } 
    kid      = atoi(argv[1]);
    kp       = atof(argv[2]);
    mu       = atof(argv[3]);
    npt      = atoi(argv[4]);
    pt_dim   = atoi(argv[5]);
    fsai_npt = atoi(argv[6]);
    fast_knn = atoi(argv[7]);
    coord    = (DTYPE*) malloc(sizeof(DTYPE) * npt * pt_dim);
    if (kid < 1 || kid > 4) kid = 1;
    if (pt_dim < 2 || pt_dim > 3) pt_dim = 3;
    if (argc >= 9)
    {
        FILE *inf = fopen(argv[8], "rb");
        fread(coord, sizeof(DTYPE), npt * pt_dim, inf);
        fclose(inf);
    } else {
        srand(814);  // Match with Tianshi's code
        DTYPE scale = DPOW((DTYPE) npt, 1.0 / (DTYPE) pt_dim);
        for (int i = 0; i < npt * pt_dim; i++) coord[i] = scale * (rand() / (DTYPE)(RAND_MAX));
    }
    select_kernel(kid, pt_dim, kp, mu, npt, &krnl_eval, &krnl_bimv, &krnl_bimv_flops);
    printf("Point set: %d points in %d-D\n", npt, pt_dim);
    printf("Linear system to solve: (K(X, X) + %.4f * I) * x = b\n", mu);
    printf("\nFSAI preconditioner parameters:\n");
    printf("- Maximum FSAI matrix nonzeros per row = %d\n", fsai_npt);
    printf("- Fast KNN for FSAI sparsity pattern   = %s\n", fast_knn ? "Yes" : "No");
    printf("\n");
    
    // Build H2 matrix
    double st, et;
    H2Pack_p h2mat = NULL;
    DTYPE h2_reltol = 1e-8;
    H2mat_build(npt, pt_dim, coord, h2_reltol, krnl_eval, krnl_bimv, krnl_bimv_flops, krnl_param, &h2mat);

    // Build FSAI preconditioner
    printf("Building FSAI preconditioner...\n");
    FSAI_precond_p FSAI_precond = NULL;
    H2P_build_FSAI_precond(h2mat, fsai_npt, mu, &FSAI_precond);
    int nnz_upper = fsai_npt * (fsai_npt + 1) / 2 + fsai_npt * (npt - fsai_npt);
    DEBUG_PRINTF("FSAI G matrix nnz = %d, nnz upper bound = %d\n", FSAI_precond->G->nnz, nnz_upper);

    // PCG test
    DTYPE CG_reltol = 1e-4;
    int max_iter = 400;
    test_PCG(
        H2Pack_matvec_diagshift, (void *) h2mat, 
        (matvec_fptr) FSAI_precond_apply, (void *) FSAI_precond, 
        npt, max_iter, CG_reltol
    );
    FSAI_precond_print_stat(FSAI_precond);

    // Clean up
    printf("\n");
    free(coord);
    H2P_destroy(&h2mat);
    FSAI_precond_destroy(&FSAI_precond);
    return 0;
}
