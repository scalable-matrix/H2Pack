#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_kernels.h"

#include "parse_scalar_params.h"
#include "direct_nbody.h"

int main(int argc, char **argv)
{
    srand48(time(NULL));
    
    parse_scalar_params(argc, argv);
    
    double st, et;

    H2Pack_t h2pack;

    // Test parameters
    const int n_rel_tol      = 3;
    const int krnl_param_len = 1;
    const int n_krnl_param   = 5;
    const DTYPE rel_tols[n_rel_tol] = {1e-3, 1e-6, 1e-9};
    const DTYPE krnl_params[n_krnl_param * krnl_param_len] = {1e-2, 1e-1, 1e0, 1e1, 1e2};

    // Loop over rel_tol and krnl_param combinations
    for (int i_rel_tol = 0; i_rel_tol < n_rel_tol; i_rel_tol++)
    {
        test_params.rel_tol = rel_tols[i_rel_tol];
        for (int i_krnl_param = 0; i_krnl_param < n_krnl_param; i_krnl_param++)
        {
            const DTYPE *krnl_param_ = krnl_params + i_krnl_param * krnl_param_len;
            test_params.krnl_param = (void*) krnl_param_;

            printf("Current parameters: rel_tol = %.1e, krnl_param[] = ", test_params.rel_tol);
            for (int i = 0; i < krnl_param_len; i++) printf("%.1e ", krnl_param_[i]);
            printf("\n");

            H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
            
            H2P_partition_points(h2pack, test_params.n_point, test_params.coord, 0, 0);

            // Generate proxy points
            H2P_dense_mat_t *pp = NULL;
            DTYPE max_L = h2pack->enbox[h2pack->root_idx * 2 * test_params.pt_dim + test_params.pt_dim];
            st = get_wtime_sec();
            H2P_generate_proxy_point_ID(
                test_params.pt_dim, test_params.krnl_dim, test_params.rel_tol, h2pack->max_level, 
                h2pack->min_adm_level, max_L, test_params.krnl_param, test_params.krnl_eval, &pp
            );
            et = get_wtime_sec();
            printf("H2Pack generate proxy points used %.3lf (s)\n", et - st);
            
            // Build H2 representation
            st = get_wtime_sec();
            H2P_build(
                h2pack, pp, test_params.BD_JIT, test_params.krnl_param, 
                test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
            );
            et = get_wtime_sec();
            printf("H2Pack H2 construction used %.3lf (s)\n", et - st);

            // Allocate input & output vectors
            int n_check_pt = 50000, check_pt_s;
            if (n_check_pt >= test_params.n_point)
            {
                n_check_pt = test_params.n_point;
                check_pt_s = 0;
            } else {
                srand(time(NULL));
                check_pt_s = rand() % (test_params.n_point - n_check_pt);
            }
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
            printf("Calculating direct n-body reference result for points %d -> %d\n", check_pt_s, check_pt_s + n_check_pt - 1);
            direct_nbody(
                test_params.krnl_param, test_params.krnl_eval, test_params.pt_dim, test_params.krnl_dim, 
                h2pack->coord,              test_params.n_point, test_params.n_point, x, 
                h2pack->coord + check_pt_s, test_params.n_point, n_check_pt,          y0
            );
            
            // Check H2 matvec accuracy
            H2P_matvec(h2pack, x, y1);
            H2P_print_statistic(h2pack);
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

            // Destroy H2Pack structure and I/O vectors
            H2P_destroy(h2pack);
            free(h2pack);
            free(x);
            free(y0);
            free(y1);
            printf("\n\n\n");
        }  // End of i_krnl_param loop
    }  // End of i_rel_tol loop
    
    free_aligned(test_params.coord);

    return 0;
}
