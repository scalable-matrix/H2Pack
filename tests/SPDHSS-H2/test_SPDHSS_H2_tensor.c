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

#include "../parse_tensor_params.h"
#include "../direct_nbody.h"

#include "pcg_tests.h"

int main(int argc, char **argv)
{
    //__itt_pause();
    srand48(time(NULL));
    
    parse_tensor_params(argc, argv);
    
    double st, et;

    H2Pack_t h2mat, hssmat;
    
    H2P_init(&h2mat, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);

    int max_leaf_points = 300;
    DTYPE max_leaf_size = 0.0;
    // Some special settings for RPY
    if (test_params.kernel_id == 1) 
    {
        H2P_run_RPY(h2mat);
        // We need to ensure the size of each leaf box >= 2 * max(radii), but the 
        // stopping criteria is "if (box_size <= max_leaf_size)", so max_leaf_size
        // should be set as 4 * max(radii).
        DTYPE *radii = test_params.coord + 3 * test_params.n_point; 
        for (int i = 0; i < test_params.n_point; i++)
            max_leaf_size = (max_leaf_size < radii[i]) ? radii[i] : max_leaf_size;
        max_leaf_size *= 4.0;
    }
    
    H2P_partition_points(h2mat, test_params.n_point, test_params.coord, max_leaf_points, max_leaf_size);
    H2P_HSS_calc_adm_inadm_pairs(h2mat);
    
    // Check if point index permutation is correct in H2Pack
    DTYPE coord_diff_sum = 0.0;
    for (int i = 0; i < test_params.n_point; i++)
    {
        DTYPE *coord_s_i = h2mat->coord + i;
        DTYPE *coord_i   = test_params.coord + h2mat->coord_idx[i];
        for (int j = 0; j < test_params.pt_dim; j++)
        {
            int idx_j = j * test_params.n_point;
            coord_diff_sum += DABS(coord_s_i[idx_j] - coord_i[idx_j]);
        }
    }
    printf("Point index permutation results %s", coord_diff_sum < 1e-15 ? "are correct\n" : "are wrong\n");

    H2P_dense_mat_t *pp;
    DTYPE max_L = h2mat->enbox[h2mat->root_idx * 2 * test_params.pt_dim + test_params.pt_dim];
    int start_level = 2;
    int num_pp = ceil(-log10(test_params.rel_tol)) - 1;
    if (num_pp < 4 ) num_pp = 4;
    if (num_pp > 10) num_pp = 10;
    num_pp = 6 * num_pp * num_pp;
    H2P_generate_proxy_point_surface(
        test_params.pt_dim, test_params.xpt_dim, num_pp, 
        h2mat->max_level, start_level, max_L, &pp
    );
    
    H2P_build(
        h2mat, pp, test_params.BD_JIT, test_params.krnl_param, 
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
    
    DTYPE *x0, *x1, *x2, *x3, *x4, *y0, *y1;
    x0 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    x1 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    assert(x0 != NULL && x1 != NULL);
    assert(y0 != NULL && y1 != NULL);
    for (int i = 0; i < test_params.krnl_mat_size; i++) 
    {
        x0[i] = (DTYPE) pseudo_randn();
        // x0[i] = (DTYPE) drand48();
    }

    // Get reference results
    direct_nbody(
        test_params.krnl_param, test_params.krnl_eval, test_params.pt_dim, test_params.krnl_dim, 
        h2mat->coord,              test_params.n_point, test_params.n_point, x0, 
        h2mat->coord + check_pt_s, test_params.n_point, n_check_pt,          y0
    );
    
    H2P_matvec(h2mat, x0, y1);
    
    printf("H2 matrix:\n");
    H2P_print_statistic(h2mat);

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

    printf("\nConstructing SPDHSS from H2\n");
    const int max_rank = 100 * 3;
    const DTYPE shift = 1e-2;
    H2P_SPDHSS_H2_build(max_rank, shift, h2mat, &hssmat);
 
    // Check HSS matvec accuracy
    memcpy(y0, y1, sizeof(DTYPE) * test_params.krnl_mat_size);
    H2P_matvec(hssmat, x0, y1);
    ref_norm = 0.0; 
    err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_mat_size; i++)
    {
        DTYPE diff = y1[i] - y0[i];
        ref_norm += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("||y_{SPDHSS} - y_{H2}||_2 / ||y_{H2}||_2 = %e\n", err_norm / ref_norm);

    // Test ULV Cholesky factorization
    H2P_HSS_ULV_Cholesky_factorize(hssmat, shift);

    for (int i = 0; i < test_params.krnl_mat_size; i++) y1[i] += shift * x0[i];
    H2P_HSS_ULV_Cholesky_solve(hssmat, 3, y1, x1);
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
    printf("H2P_HSS_ULV_Cholesky_solve, relerr = %e\n", err_norm / ref_norm);

    // Preconditioned CG test
    const int max_iter = 50;
    const DTYPE CG_tol = 1e-6;
    pcg_tests(test_params.krnl_mat_size, h2mat, hssmat, shift, max_rank, max_iter, CG_tol, 0);

    printf("\nSPDHSS matrix:\n");
    H2P_print_statistic(hssmat);

    free(x0);
    free(x1);
    free(y0);
    free(y1);
    free_aligned(test_params.coord);
    H2P_destroy(h2mat);
    H2P_destroy(hssmat);
}
