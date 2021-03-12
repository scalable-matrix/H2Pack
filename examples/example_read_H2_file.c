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

    // Kernel configuration
    int krnl_dim = 1;
    DTYPE *krnl_param = NULL;  // Coulomb kernel has no parameter
    kernel_eval_fptr krnl_eval = Coulomb_3D_eval_intrin_t;
    kernel_bimv_fptr krnl_bimv = Coulomb_3D_krnl_bimv_intrin_t;
    int krnl_bimv_flops = Coulomb_3D_krnl_bimv_flop;
    /*
    int krnl_dim = 3;
    DTYPE krnl_param[2] = {1.0, 0.1};  // Stokes kernel with parameter, eta, a
    kernel_eval_fptr krnl_eval = Stokes_eval_std;
    kernel_bimv_fptr krnl_bimv = Stokes_krnl_bimv_intrin_t;
    int krnl_bimv_flops = Stokes_krnl_bimv_flop;
    */

    // Read H2 matrix data from file and construct H2Pack
    const int BD_JIT = 1;
    H2Pack_p h2pack;
    const char *meta_json_fname = "Coulomb_3D_1e-6_meta.json";
    const char *aux_json_fname  = "Coulomb_3D_1e-6_aux.json";
    const char *binary_fname    = "Coulomb_3D_1e-6.bin";
    //const char *meta_json_fname = "Stokes_3D_1e-6_meta.json";
    //const char *aux_json_fname  = "Stokes_3D_1e-6_aux.json";
    //const char *binary_fname    = "Stokes_3D_1e-6.bin";
    printf("Reading H2 matrix data from files %s, %s, and %s\n", meta_json_fname, aux_json_fname, binary_fname);
    H2P_read_from_file(
        &h2pack, meta_json_fname, aux_json_fname, binary_fname, BD_JIT, 
        krnl_param, krnl_eval, krnl_bimv, krnl_bimv_flops
    );
    int pt_dim  = h2pack->pt_dim;
    int n_point = h2pack->n_point;
    int krnl_mat_size = h2pack->krnl_mat_size;
    DTYPE rel_tol = h2pack->QR_stop_tol;
    DTYPE *coord  = h2pack->coord0;  // Input (not sorted) point coordinates

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
    st = get_wtime_sec();
    H2P_matvec(h2pack, x, y1);
    et = get_wtime_sec();
    printf("Full H2 matvec takes %.3lf (s)\n", et - st);
    
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
    printf("The specified relative error threshold is %e\n", rel_tol);

    free(x);
    free(y0);
    free(y1);
    H2P_destroy(&h2pack);
}