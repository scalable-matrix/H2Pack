#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_kernels.h"

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
    
    // Check if point index permutation is correct in H2Pack
    DTYPE coord_diff_sum = 0.0;
    for (int i = 0; i < test_params.n_point; i++)
    {
        DTYPE *coord_s_i = h2pack->coord + i;
        DTYPE *coord_i   = test_params.coord + h2pack->coord_idx[i];
        for (int j = 0; j < test_params.pt_dim; j++)
        {
            int idx_j = j * test_params.n_point;
            coord_diff_sum += DABS(coord_s_i[idx_j] - coord_i[idx_j]);
        }
    }
    printf("Point index permutation results %s", coord_diff_sum < 1e-15 ? "are correct\n" : "are wrong\n");

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
    
    int n_check_pt = 10000, check_pt_s;
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
        x[i] = (DTYPE) drand48();
    }
    
    // Get reference results
    direct_nbody(
        test_params.krnl_param, test_params.krnl_eval, test_params.pt_dim, test_params.krnl_dim, 
        h2pack->coord,              test_params.n_point, test_params.n_point, x, 
        h2pack->coord + check_pt_s, test_params.n_point, n_check_pt,          y0
    );
    
    // Warm up, reset timers, and test the matvec performance
    H2P_matvec(h2pack, x, y1); 
    h2pack->n_matvec = 0;
    h2pack->timers[_MV_FW_TIMER_IDX]  = 0.0;
    h2pack->timers[_MV_MID_TIMER_IDX] = 0.0;
    h2pack->timers[_MV_BW_TIMER_IDX]  = 0.0;
    h2pack->timers[_MV_DEN_TIMER_IDX] = 0.0;
    h2pack->timers[_MV_RDC_TIMER_IDX] = 0.0;
    for (int i = 0; i < 10; i++) 
        H2P_matvec(h2pack, x, y1);

    H2P_print_statistic(h2pack);
    
    // Verify H2 matvec results
    DTYPE ref_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_dim * n_check_pt; i++)
    {
        DTYPE diff = y1[test_params.krnl_dim * check_pt_s + i] - y0[i];
        ref_norm += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("For %d validation points: ||y_{H2} - y||_2 / ||y||_2 = %e\n", n_check_pt, err_norm / ref_norm);
    
    free(x);
    free(y0);
    free(y1);
    free_aligned(test_params.coord);
    H2P_destroy(h2pack);
}
