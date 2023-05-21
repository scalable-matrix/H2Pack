#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_utils.h"
#include "precond_test_utils.h"
#include "../PCG/pcg.h"

// =============== Nystrom Preconditioner =============== //

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

// In Nys_precond.c
void Nys_precond_build_(
    const DTYPE mu, const int n1, const int n2, DTYPE *K11, 
    DTYPE *K12, DTYPE **nys_M_, DTYPE **nys_U_
);
void Nys_precond_apply_(
    const int n1, const int n, const DTYPE *nys_M, const DTYPE *nys_U, 
    const DTYPE *x, DTYPE *y, DTYPE *t
);
void AFNi_FPS(const int npt, const int pt_dim, const DTYPE *coord, const int k, int *idx);

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
)
{
    Nys_precond_p Nys_precond = (Nys_precond_p) malloc(sizeof(Nys_precond_s));
    memset(Nys_precond, 0, sizeof(Nys_precond_s));

    // 1. Randomly select nys_k points from npt points
    int n = npt, n1 = nys_k, n2 = npt - nys_k;
    int *perm = (int *) malloc(sizeof(int) * n);
    uint8_t *flag = (uint8_t *) malloc(sizeof(uint8_t) * n);
    DTYPE *coord_perm = (DTYPE *) malloc(sizeof(DTYPE) * npt * pt_dim);
    H2P_rand_sample(npt, nys_k, perm, flag);
    //AFNi_FPS(npt, pt_dim, coord, n1, perm);
    memset(flag, 0, sizeof(uint8_t) * n);
    for (int i = 0; i < n1; i++) flag[perm[i]] = 1;
    int idx = n1;
    for (int i = 0; i < n; i++)
        if (flag[i] == 0) perm[idx++] = i;
    H2P_gather_matrix_columns(coord, npt, coord_perm, npt, pt_dim, perm, npt);

    // 2. Build K11 and K12 blocks
    DTYPE *coord_n1 = coord_perm;
    DTYPE *coord_n2 = coord_perm + n1;
    DTYPE *K11 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    DTYPE *K12 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n2);
    int n_thread = omp_get_max_threads();
    ASSERT_PRINTF(
        K11 != NULL && K12 != NULL,
        "Failed to allocate Nystrom preconditioner K11/K12 buffers\n"
    );
    H2P_eval_kernel_matrix_OMP(
        krnl_eval, krnl_param, 
        coord_n1, n, n1, coord_n1, n, n1, 
        K11, n1, n_thread
    );
    H2P_eval_kernel_matrix_OMP(
        krnl_eval, krnl_param, 
        coord_n1, n, n1, coord_n2, n, n2, 
        K12, n2, n_thread
    );
    free(coord_perm);
    free(flag);

    // 3. Build U and M matrices
    Nys_precond->n    = n;
    Nys_precond->n1   = n1;
    Nys_precond->perm = perm;
    Nys_precond->t    = (DTYPE*) malloc(sizeof(DTYPE) * n);
    Nys_precond->px   = (DTYPE*) malloc(sizeof(DTYPE) * n);
    Nys_precond->py   = (DTYPE*) malloc(sizeof(DTYPE) * n);
    Nys_precond_build_(mu, n1, n2, K11, K12, &Nys_precond->M, &Nys_precond->U);
    *Nys_precond_ = Nys_precond;
}

// Apply a Nystrom preconditioner to a vector
void Nys_precond_apply(Nys_precond_p Nys_precond, const DTYPE *x, DTYPE *y)
{
    int n = Nys_precond->n, n1 = Nys_precond->n1;
    int *perm = Nys_precond->perm;
    DTYPE *px = Nys_precond->px, *py = Nys_precond->py, *t1 = Nys_precond->t;
    DTYPE *M = Nys_precond->M, *U = Nys_precond->U;
    for (int i = 0; i < n; i++) px[i] = x[perm[i]];
    Nys_precond_apply_(n1, n, M, U, px, py, t1);
    for (int i = 0; i < n; i++) y[perm[i]] = py[i];
}

// Destroy an initialized Nys_precond struct
void Nys_precond_destroy(Nys_precond_p *Nys_precond_)
{
    Nys_precond_p p = *Nys_precond_;
    if (p == NULL) return;
    free(p->perm);
    free(p->M);
    free(p->U);
    free(p->t);
    free(p);
}

// ============= Nystrom Preconditioner End ============= //

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