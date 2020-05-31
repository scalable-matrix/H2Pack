#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_kernels.h"
#include "H2Pack_utils.h"

#include "parse_tensor_params.h"
#include "direct_nbody.h"

int main(int argc, char **argv)
{
    srand48(time(NULL));
    
    parse_tensor_params(argc, argv);
    
    H2Pack_t h2pack;
    
    H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    
    int max_leaf_points = 300;
    DTYPE max_leaf_size = 0.0;
    // Some special settings for RPY
    if (test_params.kernel_id == 1) 
    {
        H2P_run_RPY(h2pack);
        // We need to ensure the size of each leaf box >= 2 * max(radii), but the 
        // stopping criteria is "if (box_size <= max_leaf_size)", so max_leaf_size
        // should be set as 4 * max(radii).
        DTYPE *radii = test_params.coord + 3 * test_params.n_point; 
        for (int i = 0; i < test_params.n_point; i++)
            max_leaf_size = (max_leaf_size < radii[i]) ? radii[i] : max_leaf_size;
        max_leaf_size *= 4.0;
    }
    H2P_partition_points(h2pack, test_params.n_point, test_params.coord, max_leaf_points, max_leaf_size);

    H2P_dense_mat_t *pp;
    DTYPE max_L = h2pack->enbox[h2pack->root_idx * 2 * test_params.pt_dim + test_params.pt_dim];
    int start_level = 2;
    int num_pp = ceil(-log10(test_params.rel_tol)) - 1;
    if (num_pp < 4 ) num_pp = 4;
    if (num_pp > 10) num_pp = 10;
    num_pp = 6 * num_pp * num_pp;
    H2P_generate_proxy_point_surface(
        test_params.pt_dim, test_params.xpt_dim, num_pp, 
        h2pack->max_level, start_level, max_L, &pp
    );
    
    H2P_build(
        h2pack, pp, test_params.BD_JIT, test_params.krnl_param, 
        test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
    );
    
    int n_vec = 16, n_thread = omp_get_num_threads();
    size_t mat_size = test_params.krnl_mat_size * n_vec;
    DTYPE *x0, *x1, *y0, *y1, *y2;
    x0 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    x1 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    y2 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    assert(x0 != NULL && x1 != NULL && y0 != NULL && y1 != NULL && y2 != NULL);
    for (int i = 0; i < mat_size; i++) 
    {
        //x0[i] = (DTYPE) pseudo_randn();
        x0[i] = (DTYPE) drand48();
        y0[i] = 0.0;
        y1[i] = 0.0;
    }
    
    // Warm up, reset timers, and test the matvec performance
    H2P_matvec(h2pack, x0, y0);
    h2pack->n_matvec = 0;
    h2pack->timers[_MV_FW_TIMER_IDX]  = 0.0;
    h2pack->timers[_MV_MID_TIMER_IDX] = 0.0;
    h2pack->timers[_MV_BW_TIMER_IDX]  = 0.0;
    h2pack->timers[_MV_DEN_TIMER_IDX] = 0.0;
    h2pack->timers[_MV_RDC_TIMER_IDX] = 0.0;
    for (int i = 0; i < n_vec; i++)
    {
        DTYPE *x_ivec = x0 + i * test_params.krnl_mat_size;
        DTYPE *y_ivec = y0 + i * test_params.krnl_mat_size;
        H2P_matvec(h2pack, x_ivec, y_ivec);
    }
    printf("After %d matvec calls:\n", n_vec);
    H2P_print_statistic(h2pack);
    
    // Test the matmul performance
    H2P_transpose_dmat(n_thread, n_vec, test_params.krnl_mat_size, x0, test_params.krnl_mat_size, x1, n_vec);
    h2pack->n_matvec = 0;
    h2pack->timers[_MV_FW_TIMER_IDX]  = 0.0;
    h2pack->timers[_MV_MID_TIMER_IDX] = 0.0;
    h2pack->timers[_MV_BW_TIMER_IDX]  = 0.0;
    h2pack->timers[_MV_DEN_TIMER_IDX] = 0.0;
    h2pack->timers[_MV_RDC_TIMER_IDX] = 0.0;
    H2P_matmul(h2pack, CblasRowMajor, n_vec, x1, n_vec, y1, n_vec);
    H2P_transpose_dmat(n_thread, test_params.krnl_mat_size, n_vec, y1, n_vec, y2, test_params.krnl_mat_size);
    printf("After 1 matmul calls:\n");
    H2P_print_statistic(h2pack);

    // Verify H2 matmul results
    DTYPE max_relerr = 0.0, avg_relerr = 0.0; 
    DTYPE y0_2norm, err_2norm, relerr;
    for (int i = 0; i < n_vec; i++)
    {
        DTYPE *y0_ivec = y0 + i * test_params.krnl_mat_size;
        DTYPE *y2_ivec = y2 + i * test_params.krnl_mat_size;
        calc_err_2norm(test_params.krnl_mat_size, y0_ivec, y2_ivec, &y0_2norm, &err_2norm);
        relerr = err_2norm / y0_2norm;
        if (relerr > max_relerr) max_relerr = relerr;
        avg_relerr += relerr;
    }
    avg_relerr /= (DTYPE) n_vec;
    printf("%d vectors max/avg relerr = %e, %e\n", n_vec, max_relerr, avg_relerr);
    
    free(x0);
    free(x1);
    free(y0);
    free(y1);
    free(y2);
    free_aligned(test_params.coord);
    H2P_destroy(h2pack);
}
