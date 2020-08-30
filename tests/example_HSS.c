#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_kernels.h"
#include "direct_nbody.h"


int main(int argc, char **argv)
{
    //  Timing variable
    double st, et;

    //  Point configuration, random generation
    int pt_dim = 2;
    int n_point = 80000;
    DTYPE* coord = (DTYPE*) malloc_aligned(sizeof(DTYPE) * n_point * pt_dim, 64);
    assert(coord != NULL);

    DTYPE prefac = DPOW((DTYPE) n_point, 1.0 / (DTYPE) pt_dim);
    printf("Generating random coordinates in a scaled cubic box...");
    for (int i = 0; i < n_point * pt_dim; i++)
    {
        coord[i] = (DTYPE) drand48();
        coord[i] *= prefac;
    }
 
    //  Kernel configuration
    int krnl_dim = 1;
    DTYPE krnl_param[1] = {0.1}; 
    kernel_eval_fptr krnl_eval = Gaussian_2D_eval_intrin_d;
    kernel_bimv_fptr krnl_bimv = Gaussian_2D_krnl_bimv_intrin_d;
    int krnl_bimv_flop = Gaussian_2D_krnl_bimv_flop;

    //  HSS/H2 construction configuration
    int krnl_mat_size = krnl_dim * n_point;
    DTYPE rel_tol = 1e-6;
    const int BD_JIT = 1;


    //  Initialization
    H2Pack_p h2pack;
    H2P_init(&h2pack, pt_dim, krnl_dim, QR_REL_NRM, &rel_tol);
    H2P_run_HSS(h2pack);
    
    //  Hierarchical partitioning
    H2P_partition_points(h2pack, n_point, coord, 0, 0);
    
    //  Select proxy points
    H2P_dense_mat_p *pp;
    char *pp_fname = NULL;
    st = get_wtime_sec();
    H2P_generate_proxy_point_ID_file(
        h2pack, krnl_param, krnl_eval, pp_fname, &pp
    );
    et = get_wtime_sec();
    printf("H2Pack generate proxy points used %.3lf (s)\n", et - st);
    
    //  Construct HSS matrix representation
    H2P_build(
        h2pack, pp, BD_JIT, krnl_param, krnl_eval, krnl_bimv, krnl_bimv_flop
    );
    
    //  Check multiplication errors
    int n_check_pt = 20000, check_pt_s;
    if (n_check_pt >= n_point)
    {
        n_check_pt = n_point;
        check_pt_s = 0;
    } else {
        srand(time(NULL));
        check_pt_s = rand() % (n_point - n_check_pt);
    }
    printf("Calculating direct n-body reference result for points %d -> %d\n", check_pt_s, check_pt_s + n_check_pt - 1);
    
    DTYPE *x0, *x1, *y0, *y1;
    x0 = (DTYPE*) malloc(sizeof(DTYPE) * krnl_mat_size);
    x1 = (DTYPE*) malloc(sizeof(DTYPE) * krnl_mat_size);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * krnl_dim * n_check_pt);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * krnl_mat_size);
    assert(x0 != NULL && x1 != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < krnl_mat_size; i++) 
        x0[i] = (DTYPE) drand48() - 0.5;

    // Get reference results
    direct_nbody(
        krnl_param, krnl_eval, pt_dim, krnl_dim, 
        coord,              n_point, n_point,    x0, 
        coord + check_pt_s, n_point, n_check_pt, y0
    );
    
    // HSS matrix-vector multiplication
    H2P_matvec(h2pack, x0, y1);
    
    // Verify HSS matvec results
    DTYPE ref_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < krnl_dim * n_check_pt; i++)
    {
        DTYPE diff = y1[krnl_dim * check_pt_s + i] - y0[i];
        ref_norm += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("For %d validation points: ||y_{HSS} - y||_2 / ||y||_2 = %e\n", n_check_pt, err_norm / ref_norm);
    
    #if 0
    //  Construct Cholesky-based ULV decompsition 
    const DTYPE shift = 0;
    H2P_HSS_ULV_Cholesky_factorize(h2pack, shift);

    //  Direct Solve of the HSS matrix via ULV decompsition 
    for (int i = 0; i < krnl_mat_size; i++) y1[i] += shift * x0[i];
    H2P_HSS_ULV_Cholesky_solve(h2pack, 3, y1, x1);

    //  Check errors
    ref_norm = 0.0; 
    err_norm = 0.0;
    for (int i = 0; i < krnl_mat_size; i++)
    {
        DTYPE diff = x1[i] - x0[i];
        ref_norm += x0[i] * x0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("H2P_HSS_ULV_Cholesky_solve relerr = %e\n",  err_norm / ref_norm);
    #endif

    //  Construct LU-based ULV decompsition 
    const DTYPE shift = 1e-4;
    H2P_HSS_ULV_LU_factorize(h2pack, shift);

    //  Direct Solve of the HSS matrix via ULV decompsition 
    H2P_matvec(h2pack, x0, y1);
    for (int i = 0; i < krnl_mat_size; i++) y1[i] += shift * x0[i];
    H2P_HSS_ULV_LU_solve(h2pack, 3, y1, x1);

    //  Check errors
    ref_norm = 0.0; 
    err_norm = 0.0;
    for (int i = 0; i < krnl_mat_size; i++)
    {
        DTYPE diff = x1[i] - x0[i];
        ref_norm += x0[i] * x0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("H2P_HSS_ULV_LU_solve relerr = %e\n",  err_norm / ref_norm);
    printf("%e %e\n",  err_norm, ref_norm);

    //  Print out statistis about the H2Pack
    H2P_print_statistic(h2pack);

    free(x0);
    free(x1);
    free(y0);
    free(y1);
    free_aligned(coord);
    H2P_destroy(&h2pack);
}
