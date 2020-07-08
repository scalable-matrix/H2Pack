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

static void Gaussian_3D_eval_std_d(KRNL_EVAL_PARAM)
{
    EXTRACT_3D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE l = param_[0];
    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        const DTYPE z0_i = z0[i];
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            mat_irow[j] = exp(-l * r2);
        }
    }
}

static void Gaussian_3D_bimv_std_d(KRNL_BIMV_PARAM)
{
    EXTRACT_3D_COORD();
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE l = param_[0];
    for (int i = 0; i < n0; i += 2)
    {
        const DTYPE x0_i0 = x0[i];
        const DTYPE y0_i0 = y0[i];
        const DTYPE z0_i0 = z0[i];
        const DTYPE x0_i1 = x0[i + 1];
        const DTYPE y0_i1 = y0[i + 1];
        const DTYPE z0_i1 = z0[i + 1];
        const DTYPE xin1_i0 = x_in_1[i];
        const DTYPE xin1_i1 = x_in_1[i + 1];
        DTYPE sum_i0 = 0.0, sum_i1 = 0.0;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            DTYPE d0, d1, r20, r21;

            d0 = x0_i0 - x1[j];
            d1 = x0_i1 - x1[j];
            r20 = d0 * d0;
            r21 = d1 * d1;

            d0 = y0_i0 - y1[j];
            d1 = y0_i1 - y1[j];
            r20 += d0 * d0;
            r21 += d1 * d1;

            d0 = z0_i0 - z1[j];
            d1 = z0_i1 - z1[j];
            r20 += d0 * d0;
            r21 += d1 * d1;

            r20 = exp(-l * r20);
            r21 = exp(-l * r21);

            sum_i0 += r20 * x_in_0[j];
            sum_i1 += r21 * x_in_0[j];
            x_out_1[j] += (r20 * xin1_i0 + r21 * xin1_i1);
        }
        x_out_0[i]   += sum_i0;
        x_out_0[i+1] += sum_i1;
    }
}

int main(int argc, char **argv)
{
    //__itt_pause();
    srand48(time(NULL));
    
    parse_scalar_params(argc, argv);
    test_params.krnl_eval = Gaussian_3D_eval_std_d;
    test_params.krnl_bimv = NULL; //Gaussian_3D_bimv_std_d;

    double st, et;

    H2Pack_t h2pack;
    
    H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    
    
    H2P_partition_points(h2pack, test_params.n_point, test_params.coord, 0, 0);
    
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
    st = get_wtime_sec();
    H2P_generate_proxy_point_ID(
        test_params.pt_dim, test_params.krnl_dim, test_params.rel_tol, h2pack->max_level, 
        h2pack->min_adm_level, max_L, test_params.krnl_param, test_params.krnl_eval, &pp
    );
    et = get_wtime_sec();
    printf("H2Pack generate proxy points used %.3lf (s)\n", et - st);
    
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
    
    free(x);
    free(y0);
    free(y1);
    free_aligned(test_params.coord);
    H2P_destroy(h2pack);
}
