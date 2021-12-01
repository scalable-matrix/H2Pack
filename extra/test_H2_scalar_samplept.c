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

// Copy from MATLAB code
int sample_approx_rank(const DTYPE tau, const DTYPE reltol)
{
    int r = 1, r_tmp;
    if (reltol < 2e-1) r = 2;
    if (reltol < 2e-2) r = 3;
    if (reltol < 2e-3) r = 4;
    if (reltol < 2e-4)
    {
        r_tmp = 2.0 * DFLOOR(DLOG(reltol) / DLOG(tau)) - 15.0;
        if (r_tmp < 20.0) r_tmp = 20.0;
        r = (int) DCEIL(DSQRT(r_tmp));
    }
    if (reltol < 7e-7)
    {
        r_tmp = 2.0 * DFLOOR(DLOG(reltol) / DLOG(tau)) - 10.0;
        if (r_tmp < 20.0) r_tmp = 20.0;
        r = (int) DCEIL(DSQRT(r_tmp));
    }
    if (reltol < 7e-9)
    {
        r_tmp = 2.0 * DFLOOR(DLOG(reltol) / DLOG(tau));
        if (r_tmp < 90.0) r_tmp = 90.0;
        r = (int) DCEIL(DSQRT(r_tmp));
    }
    return r;
}

int main(int argc, char **argv)
{
    //__itt_pause();
    srand48(time(NULL));
    
    printf("For this sample point example program, please enter an arbitrary proxy point file name if asked\n\n");
    parse_scalar_params(argc, argv);
    
    double st, et;

    H2Pack_p h2pack;
    
    H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    
    H2P_calc_enclosing_box(test_params.pt_dim, test_params.n_point, test_params.coord, test_params.pp_fname, &h2pack->root_enbox);

    int max_leaf_points = 0;
    DTYPE max_leaf_size = 0.0;    
    H2P_partition_points(h2pack, test_params.n_point, test_params.coord, max_leaf_points, max_leaf_size);

    DTYPE tau = 0.7;  // Separation threshold
    #if 0
    int approx_rank, approx_rank0;
    approx_rank0 = sample_approx_rank(tau, test_params.rel_tol);
    if (argc >= 9) approx_rank = atoi(argv[8]);
    else 
    {
        printf("Sample approx rank (suggested %d): ", approx_rank0);
        scanf("%d", &approx_rank);
    }
    #endif

    H2P_dense_mat_p *sample_pt;
    st = get_wtime_sec();
    H2P_select_sample_point(
        h2pack, test_params.krnl_param, test_params.krnl_eval, 
        tau, &sample_pt
    );
    et = get_wtime_sec();
    printf("H2Pack select sample points used %.3lf (s)\n", et - st);
    
    H2P_build_with_sample_point(
        h2pack, sample_pt, test_params.BD_JIT, test_params.krnl_param, 
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
    
    DTYPE *x, *y0, *y1;
    x  = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_dim * n_check_pt);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    assert(x != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < test_params.krnl_mat_size; i++) 
    {
        //x[i] = (DTYPE) pseudo_randn();
        x[i] = (DTYPE) drand48() - 0.5;
    }

    // Get reference results
    direct_nbody(
        test_params.krnl_param, test_params.krnl_eval, test_params.pt_dim, test_params.krnl_dim, 
        test_params.coord,              test_params.n_point, test_params.n_point, x, 
        test_params.coord + check_pt_s, test_params.n_point, n_check_pt,          y0
    );
    
    // Warm up, reset timers, and test the matvec performance
    H2P_matvec(h2pack, x, y1);
    H2P_reset_timers(h2pack);
    //__itt_resume();
    for (int i = 0; i < 10; i++) 
        H2P_matvec(h2pack, x, y1);
    //__itt_pause();
    
    H2P_print_statistic(h2pack);
    
    // Verify H2 matvec results
    DTYPE y0_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_dim * n_check_pt; i++)
    {
        DTYPE diff = y1[test_params.krnl_dim * check_pt_s + i] - y0[i];
        y0_norm  += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    y0_norm  = DSQRT(y0_norm);
    err_norm = DSQRT(err_norm);
    printf("For %d validation points: ||y_{H2} - y||_2 / ||y||_2 = %e\n", n_check_pt, err_norm / y0_norm);
    
    // Store H2 matrix data to file
    int store_to_file = 0;
    printf("Store H2 matrix data to file? 1-yes, 0-no : ");
    scanf("%d", &store_to_file);
    if (store_to_file)
    {
        char meta_json_fname[1024];
        char aux_json_fname[1024];
        char binary_fname[1024];
        printf("Enter meta JSON file name: ");
        scanf("%s", meta_json_fname);
        printf("Enter auxiliary JSON file name: ");
        scanf("%s", aux_json_fname);
        printf("Enter binary data file name: ");
        scanf("%s", binary_fname);
        H2P_store_to_file(h2pack, meta_json_fname, aux_json_fname, binary_fname);
        printf("done\n");
    }

    free(x);
    free(y0);
    free(y1);
    free_aligned(test_params.coord);
    H2P_destroy(&h2pack);

    return 0;
}
