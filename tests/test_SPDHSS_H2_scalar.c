#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

//#include <ittnotify.h>

#include "H2Pack.h"

#include "parse_scalar_params.h"

#include "direct_nbody.h"

#include "pcg.h"

void HSS_ULV_Chol_precond(H2Pack_t hssmat, const DTYPE *b, DTYPE *x)
{
    H2P_HSS_ULV_Cholesky_solve(hssmat, 3, b, x);
}

int main(int argc, char **argv)
{
    //__itt_pause();
    srand48(time(NULL));
    
    parse_scalar_params(argc, argv);
    
    double st, et;

    H2Pack_t h2mat, hssmat;
    
    H2P_init(&h2mat, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    
    H2P_partition_points(h2mat, test_params.n_point, test_params.coord, 0, 0);
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
    DTYPE Quadratic_krnl_param[2] = {1.0, -0.5};
    void *krnl_param = NULL;
    if (test_params.kernel_id == 3) krnl_param = (void*) &Quadratic_krnl_param[0];
    
    st = get_wtime_sec();
    H2P_generate_proxy_point_ID(
        test_params.pt_dim, test_params.krnl_dim, test_params.rel_tol, h2mat->max_level, 
        h2mat->min_adm_level, max_L, krnl_param, test_params.krnl_eval, &pp
    );
    et = get_wtime_sec();
    printf("H2Pack generate proxy points used %.3lf (s)\n", et - st);
    
    H2P_build(
        h2mat, pp, test_params.BD_JIT, krnl_param, 
        test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
    );

    int n_check_pt = 50000, check_pt_s;
    if (n_check_pt > test_params.n_point)
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
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    assert(x0 != NULL && x1 != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < test_params.krnl_mat_size; i++) 
    {
        x0[i] = (DTYPE) pseudo_randn();
        // x0[i] = (DTYPE) drand48();
    }

    // Get reference results
    direct_nbody(
        krnl_param, test_params.krnl_eval, test_params.pt_dim, test_params.krnl_dim, 
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

    printf("Constructing SPDHSS from H2\n");
    const int max_rank = 100;
    const DTYPE shift = 0.0;
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
    st = get_wtime_sec();
    H2P_HSS_ULV_Cholesky_factorize(hssmat, shift);
    et = get_wtime_sec();
    printf("H2P_HSS_ULV_Cholesky_factorize used %.3lf sec\n", et - st);

    for (int i = 0; i < test_params.krnl_mat_size; i++) y1[i] += shift * x0[i];
    st = get_wtime_sec();
    H2P_HSS_ULV_Cholesky_solve(hssmat, 3, y1, x1);
    et = get_wtime_sec();
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
    printf("H2P_HSS_ULV_Cholesky_solve     used %.3lf sec, relerr = %e\n", et - st, err_norm / ref_norm);

    printf("SPDHSS matrix:\n");
    H2P_print_statistic(hssmat);

    // Preconditioned CG test
    for (int i = 0; i < test_params.krnl_mat_size; i++)
    {
        y0[i] = drand48();
        x0[i] = 0.0;
        x1[i] = 0.0;
    }
    printf("Starting PCG solve without preconditioner...\n");
    int flag0, flag1, iter0, iter1;
    DTYPE relres0, relres1;
    pcg(
        test_params.krnl_mat_size, 1e-6, 50, 
        H2P_matvec, h2mat, y0, NULL, NULL, x0,
        &flag0, &relres0, &iter0, NULL
    );
    printf("Starting PCG solve with SPDHSS preconditioner...\n");
    pcg(
        test_params.krnl_mat_size, 1e-6, 50, 
        H2P_matvec, h2mat, y0, HSS_ULV_Chol_precond, hssmat, x1,
        &flag1, &relres1, &iter1, NULL
    );

    free(x0);
    free(x1);
    free(y0);
    free(y1);
    free_aligned(test_params.coord);
    H2P_destroy(h2mat);
}
