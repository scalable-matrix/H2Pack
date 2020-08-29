#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

//#include <ittnotify.h>

#include "H2Pack.h"
#include "H2Pack_kernels.h"

#include "parse_scalar_params.h"
#include "direct_nbody.h"

#include "debug.h"

int main(int argc, char **argv)
{
    //__itt_pause();
    srand48(time(NULL));
    
    parse_scalar_params(argc, argv);

    double st, et;

    H2Pack_p h2pack;
    
    H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    H2P_run_HSS(h2pack);
    
    H2P_calc_enclosing_box(test_params.pt_dim, test_params.n_point, test_params.coord, test_params.pp_fname, &h2pack->root_enbox);

    int max_leaf_points = 0;
    DTYPE max_leaf_size = 0.0;    
    H2P_partition_points(h2pack, test_params.n_point, test_params.coord, max_leaf_points, max_leaf_size);

    H2P_dense_mat_p *pp;
    st = get_wtime_sec();
    H2P_generate_proxy_point_ID_file(
        h2pack, test_params.krnl_param, test_params.krnl_eval,
        test_params.pp_fname, &pp
    );
    et = get_wtime_sec();
    printf("H2Pack load/generate proxy points used %.3lf (s)\n", et - st);
    
    H2P_build(
        h2pack, pp, test_params.BD_JIT, test_params.krnl_param, 
        test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
    );
    
    int n_check_pt = 50000, check_pt_s;
    if (n_check_pt >= test_params.n_point)
    {
        n_check_pt = test_params.n_point;
        check_pt_s = 0;
    } else {
        srand(time(NULL));
        check_pt_s = rand() % (test_params.n_point - n_check_pt);
    }
    printf("Calculating direct n-body reference result for points %d -> %d\n", check_pt_s, check_pt_s + n_check_pt - 1);
    
    DTYPE *x0, *x1, *y0, *y1;
    x0 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    x1 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_dim * n_check_pt);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    assert(x0 != NULL && x1 != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < test_params.krnl_mat_size; i++) 
    {
        //x0[i] = (DTYPE) pseudo_randn();
        x0[i] = (DTYPE) drand48() - 0.5;
    }

    // Get reference results
    direct_nbody(
        test_params.krnl_param, test_params.krnl_eval, test_params.pt_dim, test_params.krnl_dim, 
        test_params.coord,              test_params.n_point, test_params.n_point, x0, 
        test_params.coord + check_pt_s, test_params.n_point, n_check_pt,          y0
    );
    
    // Warm up, reset timers, and test the matvec performance
    H2P_matvec(h2pack, x0, y1);
    H2P_reset_timers(h2pack);
    //__itt_resume();
    for (int i = 0; i < 10; i++) 
        H2P_matvec(h2pack, x0, y1);
    //__itt_pause();
    
    
    // Verify HSS matvec results
    DTYPE ref_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_dim * n_check_pt; i++)
    {
        DTYPE diff = y1[test_params.krnl_dim * check_pt_s + i] - y0[i];
        ref_norm += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("For %d validation points: ||y_{HSS} - y||_2 / ||y||_2 = %e\n", n_check_pt, err_norm / ref_norm);
    
    #if 0
    // Test ULV Cholesky factorization
    const DTYPE shift = 0;
    H2P_HSS_ULV_Cholesky_factorize(h2pack, shift);

    for (int i = 0; i < test_params.krnl_mat_size; i++) y1[i] += shift * x0[i];
    // Warm up, reset timers, and test the ULV solve performance
    H2P_HSS_ULV_Cholesky_solve(h2pack, 3, y1, x1);
    h2pack->n_ULV_solve = 0;
    h2pack->timers[_ULV_SLV_TIMER_IDX] = 0.0;
    for (int i = 0; i < 10; i++) 
        H2P_HSS_ULV_Cholesky_solve(h2pack, 3, y1, x1);
    ref_norm = 0.0; 
    err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_mat_size; i++)
    {
        DTYPE diff = x1[i] - x0[i];
        ref_norm += x0[i] * x0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("H2P_HSS_ULV_Cholesky_solve relerr = %e\n",  err_norm / ref_norm);
    #endif

    //dump_HSS(h2pack);

    // Test ULV LU factorization
    const DTYPE shift = 0;
    H2P_HSS_ULV_LU_factorize(h2pack, shift);

    for (int i = 0; i < test_params.krnl_mat_size; i++) y1[i] += shift * x0[i];
    // Warm up, reset timers, and test the ULV solve performance
    H2P_HSS_ULV_LU_solve(h2pack, 3, y1, x1);
    h2pack->n_ULV_solve = 0;
    h2pack->timers[_ULV_SLV_TIMER_IDX] = 0.0;
    for (int i = 0; i < 10; i++) 
        H2P_HSS_ULV_LU_solve(h2pack, 3, y1, x1);
    ref_norm = 0.0; 
    err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_mat_size; i++)
    {
        DTYPE diff = x1[i] - x0[i];
        ref_norm += x0[i] * x0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("H2P_HSS_ULV_LU_solve relerr = %e\n",  err_norm / ref_norm);

    H2P_print_statistic(h2pack);

    free(x0);
    free(x1);
    free(y0);
    free(y1);
    free_aligned(test_params.coord);
    H2P_destroy(h2pack);
}
