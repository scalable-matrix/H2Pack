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
    srand48(time(NULL));
    double st, et;

    // Point configuration, random generation
    int pt_dim  = 3;
    int n_point = 160000;
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
    DTYPE *krnl_param = NULL;  // Coulomb kernel has no parameter
    kernel_eval_fptr krnl_eval = Coulomb_3D_eval_intrin_d;
    kernel_bimv_fptr krnl_bimv = Coulomb_3D_krnl_bimv_intrin_d;
    int krnl_bimv_flops = Coulomb_3D_krnl_bimv_flop;

    // HSS/H2 construction configuration
    int krnl_mat_size = krnl_dim * n_point;
    DTYPE rel_tol = 1e-6;
    const int BD_JIT = 1;


    // Initialization of H2Pack
    H2Pack_p h2pack;
    H2P_init(&h2pack, pt_dim, krnl_dim, QR_REL_NRM, &rel_tol);
    
    // Hierarchical partitioning
    int max_leaf_points = 0;    // use the default in h2pack for maximum number of points in the leaf node
    DTYPE max_leaf_size = 0.0;  // use the default in h2pack for maximum edge length of leaf box
    char *pp_fname = "./PP_Coulomb3D_1e-6.dat";
    H2P_calc_enclosing_box(pt_dim, n_point, coord, pp_fname, &h2pack->root_enbox);
    H2P_partition_points(h2pack, n_point, coord, max_leaf_points, max_leaf_size);
    
    // Select proxy points
    H2P_dense_mat_p *pp;
    st = get_wtime_sec();
    H2P_generate_proxy_point_ID_file(h2pack, krnl_param, krnl_eval, pp_fname, &pp);
    et = get_wtime_sec();
    printf("H2Pack generate proxy points used %.3lf (s)...\n", et - st);
    
    // Construct H2 matrix representation
    H2P_build(h2pack, pp, BD_JIT, krnl_param, krnl_eval, krnl_bimv, krnl_bimv_flops);
    
    // Check multiplication error at 20000 entries
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
    
    DTYPE *x, *y0, *y1;
    x  = (DTYPE*) malloc(sizeof(DTYPE) * krnl_mat_size);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * krnl_dim * n_check_pt);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * krnl_mat_size);
    assert(x != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < krnl_mat_size; i++) 
        x[i] = (DTYPE) drand48() - 0.5;

 
    // Get reference results
    st = get_wtime_sec();
    direct_nbody(
        krnl_param, krnl_eval, pt_dim, krnl_dim, 
        coord,              n_point, n_point,    x, 
        coord + check_pt_s, n_point, n_check_pt, y0
    );
    et = get_wtime_sec();
    printf("Direct n-body for %d points takes %.3lf (s)\n", n_check_pt, et - st);
    
    // H2 matrix-vector multiplication
    H2P_matvec(h2pack, x, y1);
    
    // Print out details of the H2 matrix
    H2P_print_statistic(h2pack);
    
    // Verify H2 matvec results
    DTYPE y0_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < krnl_dim * n_check_pt; i++)
    {
        DTYPE diff = y1[krnl_dim * check_pt_s + i] - y0[i];
        y0_norm  += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    y0_norm  = DSQRT(y0_norm);
    err_norm = DSQRT(err_norm);
    printf("For %d validation points: ||y_{H2} - y||_2 / ||y||_2 = %e\n", n_check_pt, err_norm / y0_norm);
    
    free(x);
    free(y0);
    free(y1);
    free_aligned(coord);
    H2P_destroy(&h2pack);
}
