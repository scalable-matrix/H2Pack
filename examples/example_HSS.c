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
    // Timing variable
    double st, et;

    // Point configuration, random generation
    int pt_dim  = 2;
    int n_point = 20000;
    DTYPE* coord = (DTYPE*) malloc_aligned(sizeof(DTYPE) * n_point * pt_dim, 64);
    assert(coord != NULL);

    DTYPE prefac = DPOW((DTYPE) n_point, 1.0 / (DTYPE) pt_dim);
    printf("Generating random coordinates in a scaled cubic box...");
    for (int i = 0; i < n_point * pt_dim; i++)
    {
        coord[i] = (DTYPE) drand48();
        coord[i] *= prefac;
    }
    printf(" done.\n");
 
    // Kernel configuration
    int krnl_dim = 1;
    /*
    DTYPE krnl_param[1] = {0.5};  //2D Gaussian kernel with the exponent parameter 
    kernel_eval_fptr krnl_eval = Gaussian_2D_eval_intrin_t;
    kernel_bimv_fptr krnl_bimv = Gaussian_2D_krnl_bimv_intrin_t;
    int krnl_bimv_flop = Gaussian_2D_krnl_bimv_flop;
    */
    DTYPE krnl_param[2] = {100.0, -0.5};  //2D quadratic kernel with the scale and exponent parameters
    kernel_eval_fptr krnl_eval = Quadratic_2D_eval_intrin_t;
    kernel_bimv_fptr krnl_bimv = Quadratic_2D_krnl_bimv_intrin_t;
    int krnl_bimv_flop = Quadratic_2D_krnl_bimv_flop;

    // HSS construction configuration
    int krnl_mat_size = krnl_dim * n_point;
    DTYPE rel_tol = 1e-6;
    const int BD_JIT = 1;

    // Initialization of H2Pack
    H2Pack_p h2pack;
    H2P_init(&h2pack, pt_dim, krnl_dim, QR_REL_NRM, &rel_tol);
    H2P_run_HSS(h2pack);
    
    // Hierarchical partitioning
    int max_leaf_points = 0;    // use the default in h2pack for maximum number of points in the leaf node
    DTYPE max_leaf_size = 0.0;  // use the default in h2pack for maximum edge length of leaf box
    //char *pp_fname = "./PP_Gaussian2D_1e-6.dat"; //  file name for storage and reuse of proxy points, can be set as NULL.
    char *pp_fname = "./PP_Quadratic2D_1e-6.dat";
    H2P_calc_enclosing_box(pt_dim, n_point, coord, pp_fname, &h2pack->root_enbox);
    H2P_partition_points(h2pack, n_point, coord, max_leaf_points, max_leaf_size);
    
    // Select proxy points
    H2P_dense_mat_p *pp;
    //  method 1: numerical proxy point selection, works for any kernel but require relatively expensive precomputation
    //            the computed proxy points will be stored in `pp_fname' (if not NULL) for reuse if needed. 
    if (1)
    {
        st = get_wtime_sec();
        H2P_generate_proxy_point_ID_file(h2pack, krnl_param, krnl_eval, pp_fname, &pp);
        et = get_wtime_sec();
        printf("H2Pack generate numerical proxy points used %.3lf (s)...\n", et - st);
    }
    else
    {
    //  method 2: proxy surface points, works for kernel from potential theory, has negligible cost.
        // The edge length of the root box enclosing all the points
        DTYPE max_L = h2pack->root_enbox[pt_dim];
        // A heuristic but effective selection of the number of proxy surface points given the expected relative tolerance
        int num_pp, num_pp_dim = ceil(-log10(rel_tol));
        if (num_pp_dim < 4 ) num_pp_dim = 4;
        if (num_pp_dim > 10) num_pp_dim = 10;
        if (pt_dim == 2) num_pp = 2 * pt_dim * num_pp_dim;
        if (pt_dim == 3) num_pp = 2 * pt_dim * num_pp_dim * num_pp_dim;
        st = get_wtime_sec();
        H2P_generate_proxy_point_surface(
            pt_dim, pt_dim, num_pp, h2pack->max_level,
            h2pack->HSS_min_adm_level, max_L, &pp
        );
        et = get_wtime_sec();
        printf("H2Pack generate proxy surface points used %.3lf (s)...\n", et - st);
    }
 
    // Construct HSS matrix representation
    H2P_build(h2pack, pp, BD_JIT, krnl_param, krnl_eval, krnl_bimv, krnl_bimv_flop);
    
    // Check multiplication errors
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
    st = get_wtime_sec();
    direct_nbody(
        krnl_param, krnl_eval, pt_dim, krnl_dim, 
        coord,              n_point, n_point,    x0, 
        coord + check_pt_s, n_point, n_check_pt, y0
    );
    et = get_wtime_sec();
    printf("Direct n-body for %d points takes %.3lf (s)\n", n_check_pt, et - st);
    
    // HSS matrix-vector multiplication
    st = get_wtime_sec();
    H2P_matvec(h2pack, x0, y1);
    et = get_wtime_sec();
    printf("Full HSS matvec takes %.3lf (s)\n", et - st);
    
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
    printf("The specified relative error threshold is %e\n", rel_tol);
    
    #if 0
    // Construct Cholesky-based ULV decomposition 
    // Note: The HSS matrix should be SPD
    const DTYPE shift = 0;
    H2P_HSS_ULV_Cholesky_factorize(h2pack, shift);

    // Direct Solve of the HSS matrix via ULV decomposition 
    for (int i = 0; i < krnl_mat_size; i++) y1[i] += shift * x0[i];
    H2P_HSS_ULV_Cholesky_solve(h2pack, 3, y1, x1);

    // Check errors
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
    #else
    // Construct LU-based ULV decomposition 
    // Could add an additional diagonal shift to the HSS matrix and then ULV decomposition.
    const DTYPE shift = 0;    
    H2P_HSS_ULV_LU_factorize(h2pack, shift);

    // Direct Solve of the HSS matrix via ULV decomposition 
    H2P_matvec(h2pack, x0, y1);
    for (int i = 0; i < krnl_mat_size; i++) y1[i] += shift * x0[i];
    H2P_HSS_ULV_LU_solve(h2pack, 3, y1, x1);

    // Check errors
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
    #endif
    printf("HSS logdet = %e\n", h2pack->HSS_logdet);

    // Print out details of the H2 matrix
    H2P_print_statistic(h2pack);

    // Store H2 matrix data to file
    int store_to_file = 0;
    printf("Store HSS matrix data to file? 1-yes, 0-no : ");
    scanf("%d", &store_to_file);
    if (store_to_file)
    {
        const char *meta_json_fname = "Quadratic_2D_1e-6_meta.json";
        const char *aux_json_fname  = "Quadratic_2D_1e-6_aux.json";
        const char *binary_fname    = "Quadratic_2D_1e-6.bin";
        printf("Storing HSS matrix data to files %s, %s, and %s...", meta_json_fname, aux_json_fname, binary_fname);
        fflush(stdout);
        H2P_store_to_file(h2pack, meta_json_fname, aux_json_fname, binary_fname);
    }

    free(x0);
    free(x1);
    free(y0);
    free(y1);
    free_aligned(coord);
    H2P_destroy(&h2pack);
}
