#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "precond_test_utils.h"
#include "AFN_precond.h"
#include "../PCG/pcg.h"

int main(int argc, char **argv)
{
    // Parse command line arguments
    int kid, npt, pt_dim, max_k, ss_npt, fsai_npt, fast_knn;
    DTYPE mu, kp, *coord = NULL;
    void *krnl_param = &kp;
    kernel_eval_fptr krnl_eval = NULL;
    kernel_bimv_fptr krnl_bimv = NULL;
    int krnl_bimv_flops = 0;
    if (argc < 10)
    {
        printf("Usage: %s kid kp mu npt pt_dim max_k ss_npt fsai_npt fast_knn coord_bin\n", argv[0]);
        printf("  - kid       [int]    : Kernel function ID\n");
        printf("                         1 - Gaussian    k(x, y) = exp(-l * |x-y|^2)\n");
        printf("                         2 - Exponential k(x, y) = exp(-l * |x-y|)\n");
        printf("                         3 - 3/2 Matern  k(x, y) = (1 + l*k) * exp(-l*k), k = sqrt(3) * |x-y|\n");
        printf("                         4 - 5/2 Matern  k(x, y) = (1 + l*k + l^2*k^2/3) * exp(-l*k), k = sqrt(5) * |x-y|\n");
        printf("  - kp        [double] : Kernel function parameter (l)\n");
        printf("  - mu        [double] : Kernel matrix diagonal shift\n");
        printf("  - npt       [int]    : Number of points\n");
        printf("  - pt_dim    [int]    : Point dimension\n");
        printf("  - max_k     [int]    : Maximum global low-rank approximation rank\n");
        printf("  - ss_npt    [int]    : Number of points in the sample set\n");
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
    max_k    = atoi(argv[6]);
    ss_npt   = atoi(argv[7]);
    fsai_npt = atoi(argv[8]);
    fast_knn = atoi(argv[9]);
    coord    = (DTYPE*) malloc(sizeof(DTYPE) * npt * pt_dim);
    if (kid < 1 || kid > 4) kid = 1;
    if (pt_dim < 2 || pt_dim > 3) pt_dim = 3;
    if (argc >= 11)
    {
        FILE *inf = fopen(argv[10], "rb");
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
    printf("\nAFN preconditioner parameters:\n");
    printf("- Maximum Nystrom approximation rank     = %d\n", max_k);
    printf("- Maximum Rank estimation sampled points = %d\n", ss_npt);
    printf("- Maximum FSAI matrix nonzeros per row   = %d\n", fsai_npt);
    printf("- Fast KNN for FSAI sparsity pattern     = %s\n", fast_knn ? "Yes" : "No");
    printf("\n");
    
    // Build H2 matrix
    double st, et;
    H2Pack_p h2mat = NULL;
    DTYPE h2_reltol = 1e-8;
    H2mat_build(npt, pt_dim, coord, h2_reltol, krnl_eval, krnl_bimv, krnl_bimv_flops, krnl_param, &h2mat);

    // Build AFN preconditioner
    printf("Building AFN preconditioner...\n");
    st = get_wtime_sec();
    AFN_precond_p AFN_precond = NULL;
    void *h2mat_ = (fast_knn) ? (void *) h2mat : NULL;
    AFN_precond_build(krnl_eval, krnl_param, npt, pt_dim, coord, mu, max_k, ss_npt, fsai_npt, h2mat_, &AFN_precond);
    et = get_wtime_sec();
    printf("AFN_precond build time = %.3lf s\n", et - st);
    printf("AFN estimated kernel matrix rank = %d, ", AFN_precond->est_rank);
    printf("will use %s\n", (AFN_precond->est_rank >= max_k) ? "AFN" : "Nystrom");

    // PCG test
    DTYPE CG_reltol = 1e-4;
    int max_iter = 400;
    test_PCG(
        H2Pack_matvec_diagshift, (void *) h2mat, 
        (matvec_fptr) AFN_precond_apply, (void *) AFN_precond, 
        npt, max_iter, CG_reltol
    );

    // Print AFN preconditioner statistics and clean up
    AFN_precond_print_stat(AFN_precond);
    printf("\n");
    free(coord);
    H2P_destroy(&h2mat);
    AFN_precond_destroy(&AFN_precond);
    return 0;
}
