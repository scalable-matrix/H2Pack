#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "precond_test_utils.h"
#include "../PCG/pcg.h"

// ================= FSAI Preconditioner ================ //

struct FSAI_precond
{
    int n;              // Size of the kernel matrix, == number of points
    int   *G_rowptr;    // Size n2 + 1, AFN G matrix CSR row_ptr array
    int   *GT_rowptr;   // Size n2 + 1, AFN G^T matrix CSR row_ptr array
    int   *G_colidx;    // Size nnz, AFN G matrix CSR col_idx array
    int   *GT_colidx;   // Size nnz, AFN G^T matrix CSR col_idx array
    DTYPE *G_val;       // Size nnz, AFN G matrix CSR values array
    DTYPE *GT_val;      // Size nnz, AFN G^T matrix CSR values array
    DTYPE *t;           // Size n, intermediate vectors in Nystrom_precond_apply

    // Timers
    double t_knn, t_fsai, t_csr;
};
typedef struct FSAI_precond  FSAI_precond_s;
typedef struct FSAI_precond* FSAI_precond_p;

// In Nys_precond.c
extern void FSAI_precond_build_(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int fsai_npt,
    const int n, const int pt_dim, const DTYPE *coord, const int ldc, 
    const int *coord0_idx, const int n1, const DTYPE *P, const DTYPE mu, void *h2mat, 
    int **G_rowptr_,  int **G_colidx_,  DTYPE **G_val_, 
    int **GT_rowptr_, int **GT_colidx_, DTYPE **GT_val_,
    double *t_knn_, double *t_fsai_, double *t_csr_
);
extern void FSAI_precond_apply_(
    const int *G_rowptr, const int *G_colidx, const DTYPE *G_val,
    const int *GT_rowptr, const int *GT_colidx, const DTYPE *GT_val,
    const int n, const DTYPE *x, DTYPE *y, DTYPE *t
);

// Build a Factoized Sparse Approximate Inverse (FSAI) preconditioner 
// for a kernel matrix K(X, X) + mu * I - P * P^T, where P is a low rank matrix
// Input parameters:
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   krnl_param : Pointer to kernel function parameter array
//   fsai_npt   : Maximum number of nonzeros in each row of the FSAI matrix
//   n, pt_dim  : Number of points and point dimension
//   coord      : Size pt_dim * ldc, row major, each column is a point coordinate
//   ldc        : Leading dimension of coord, >= n
//   mu         : Diagonal shift
//   h2mat      : Optional, pointer to an initialized H2Pack struct, used for FSAI KNN search
// Output parameter:
//   *FSAI_precond_  : Pointer to an initialized FSAI preconditioner
void FSAI_precond_build(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int fsai_npt,
    const int n, const int pt_dim, const DTYPE *coord, const int ldc, 
    const DTYPE mu, void *h2mat, FSAI_precond_p *FSAI_precond_
)
{
    FSAI_precond_p FSAI_precond = (FSAI_precond_p) malloc(sizeof(FSAI_precond_s));
    memset(FSAI_precond, 0, sizeof(FSAI_precond_s));

    int *coord0_idx = (int *) malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) coord0_idx[i] = i;
    FSAI_precond->n = n;
    FSAI_precond->t = (DTYPE *) malloc(sizeof(DTYPE) * n);
    FSAI_precond_build_(
        krnl_eval, krnl_param, fsai_npt, 
        n, pt_dim, coord, ldc, 
        coord0_idx, 0, NULL, mu, h2mat,
        &FSAI_precond->G_rowptr,  &FSAI_precond->G_colidx,  &FSAI_precond->G_val,
        &FSAI_precond->GT_rowptr, &FSAI_precond->GT_colidx, &FSAI_precond->GT_val,
        &FSAI_precond->t_knn, &FSAI_precond->t_fsai, &FSAI_precond->t_csr
    );

    free(coord0_idx);
    *FSAI_precond_ = FSAI_precond;
}

// Apply a FSAI preconditioner to a vector
void FSAI_precond_apply(const void *FSAI_precond, const DTYPE *x, DTYPE *y)
{
    FSAI_precond_p p = (FSAI_precond_p) FSAI_precond;
    FSAI_precond_apply_(
        p->G_rowptr, p->G_colidx, p->G_val,
        p->GT_rowptr, p->GT_colidx, p->GT_val,
        p->n, x, y, p->t
    );
}

// Destroy an initialized FSAI_precond struct
void FSAI_precond_destroy(FSAI_precond_p *FSAI_precond_)
{
    FSAI_precond_p p = *FSAI_precond_;
    if (p == NULL) return;
    free(p->t);
    free(p->G_rowptr);
    free(p->G_colidx);
    free(p->G_val);
    free(p->GT_rowptr);
    free(p->GT_colidx);
    free(p->GT_val);
    free(p);
}

// ============== FSAI Preconditioner End =============== //

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
    if (argc >= 11)
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
    int krnl_dim = 1, BD_JIT = 1;
    DTYPE reltol = 1e-8;
    H2P_dense_mat_p *pp = NULL;
    printf("Building H2 representation with reltol = %.4e for kernel matrix...\n", reltol);
    H2P_init(&h2mat, pt_dim, krnl_dim, QR_REL_NRM, &reltol);
    H2P_calc_enclosing_box(pt_dim, npt, coord, NULL, &h2mat->root_enbox);
    H2P_partition_points(h2mat, npt, coord, 0, 0);
    st = get_wtime_sec();
    H2P_generate_proxy_point_ID_file(h2mat, krnl_param, krnl_eval, NULL, &pp);
    et = get_wtime_sec();
    printf("H2Pack proxy point selection time = %.3f s\n", et - st);
    st = get_wtime_sec();
    H2P_build(h2mat, pp, BD_JIT, krnl_param, krnl_eval, krnl_bimv, krnl_bimv_flops);
    et = get_wtime_sec();
    printf("H2Pack build time = %.3f s\n", et - st);
    H2P_print_statistic(h2mat);
    printf("\n");

    // Build AFN preconditioner
    printf("Building FSAI preconditioner...\n");
    st = get_wtime_sec();
    FSAI_precond_p FSAI_precond = NULL;
    void *h2mat_ = (fast_knn) ? (void *) h2mat : NULL;
    FSAI_precond_build(
        krnl_eval, krnl_param, fsai_npt, 
        npt, pt_dim, coord, npt, 
        mu, h2mat_, &FSAI_precond
    );
    et = get_wtime_sec();
    printf("FSAI preconditioner build time = %.3f s\n", et - st);
    printf("  * KNN search                 = %.3f s\n", FSAI_precond->t_knn);
    printf("  * FSAI COO matrix build      = %.3f s\n", FSAI_precond->t_fsai);
    printf("  * FSAI COO matrix to CSR     = %.3f s\n", FSAI_precond->t_csr);
    printf("\n");

    // PCG test
    DTYPE CG_reltol = 1e-4, relres;
    int max_iter = 400, flag, iter;
    DTYPE *x = malloc(sizeof(DTYPE) * npt);
    DTYPE *b = malloc(sizeof(DTYPE) * npt);
    srand(126);  // Match with Tianshi's code
    for (int i = 0; i < npt; i++)
    {
        b[i] = (rand() / (DTYPE) RAND_MAX) - 0.5;
        x[i] = 0.0;
    }
    int pcg_print_level = 1;
    pcg(
        npt, CG_reltol, max_iter, 
        H2Pack_matvec, h2mat, b, FSAI_precond_apply, FSAI_precond, x,
        &flag, &relres, &iter, NULL, pcg_print_level
    );

    // Clean up
    printf("\n");
    FSAI_precond_destroy(&FSAI_precond);
    free(coord);
    free(x);
    free(b);
    H2P_destroy(&h2mat);
    return 0;
}
