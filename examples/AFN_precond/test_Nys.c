#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "Nys_precond.h"
#include "precond_test_utils.h"
#include "../PCG/pcg.h"

int main(int argc, char **argv)
{
    // Parse command line arguments
    int kid, npt, pt_dim, nys_k;
    DTYPE mu, kp, *coord = NULL;
    void *krnl_param = &kp;
    kernel_eval_fptr krnl_eval = NULL;
    kernel_bimv_fptr krnl_bimv = NULL;
    int krnl_bimv_flops = 0;
    if (argc < 7)
    {
        printf("Usage: %s kid kp mu npt pt_dim nys_k coord_bin\n", argv[0]);
        printf("  - kid       [int]    : Kernel function ID\n");
        printf("                         1 - Gaussian    k(x, y) = exp(-l * |x-y|^2)\n");
        printf("                         2 - Exponential k(x, y) = exp(-l * |x-y|)\n");
        printf("                         3 - 3/2 Matern  k(x, y) = (1 + l*k) * exp(-l*k), k = sqrt(3) * |x-y|\n");
        printf("                         4 - 5/2 Matern  k(x, y) = (1 + l*k + l^2*k^2/3) * exp(-l*k), k = sqrt(5) * |x-y|\n");
        printf("  - kp        [double] : Kernel function parameter (l)\n");
        printf("  - mu        [double] : Kernel matrix diagonal shift\n");
        printf("  - npt       [int]    : Number of points\n");
        printf("  - pt_dim    [int]    : Point dimension\n");
        printf("  - nys_k     [int]    : Nystrom approximation rank\n");
        printf("  - coord_bin [str]    : (Optional) Binary file containing the coordinates, size pt_dim * npt,\n");
        printf("                         row major, each column is a point coordinate\n");
        return 255;
    } 
    kid      = atoi(argv[1]);
    kp       = atof(argv[2]);
    mu       = atof(argv[3]);
    npt      = atoi(argv[4]);
    pt_dim   = atoi(argv[5]);
    nys_k    = atoi(argv[6]);
    coord    = (DTYPE*) malloc(sizeof(DTYPE) * npt * pt_dim);
    if (kid < 1 || kid > 4) kid = 1;
    if (pt_dim < 2 || pt_dim > 3) pt_dim = 3;
    if (argc >= 8)
    {
        FILE *inf = fopen(argv[7], "rb");
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
    printf("Nystrom approximation rank: %d\n", nys_k);

    // Build H2 matrix
    double st, et;
    H2Pack_p h2mat = NULL;
    H2mat_build(npt, pt_dim, coord, krnl_eval, krnl_bimv, krnl_bimv_flops, krnl_param, &h2mat);

    // Build Nystrom preconditioner
    printf("Building randomize Nystrom preconditioner...\n");
    st = get_wtime_sec();
    Nys_precond_p Nys_precond = NULL;
    Nys_precond_build(krnl_eval, krnl_param, npt, pt_dim, coord, mu, nys_k, &Nys_precond);
    et = get_wtime_sec();
    printf("Nys_precond build time = %.3lf s\n", et - st);

    // PCG test
    DTYPE CG_reltol = 1e-4;
    int max_iter = 400;
    test_PCG(
        H2Pack_matvec, (void *) h2mat, 
        (matvec_fptr) Nys_precond_apply, (void *) Nys_precond, 
        npt, max_iter, CG_reltol
    );

    // Clean up
    printf("\n");
    free(coord);
    H2P_destroy(&h2mat);
    Nys_precond_destroy(&Nys_precond);
    return 0;
}