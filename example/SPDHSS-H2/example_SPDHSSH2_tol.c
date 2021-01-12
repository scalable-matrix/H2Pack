#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_kernels.h"

#include "../../extra/direct_nbody.h"
#include "parse_scalar_params.h"
#include "pcg_tests.h"

/*
 *  Numerical test with diagonal-shifted scalar kernel matrices and SPDHSS-H2 as preconditioner
 *  
 *  Example run: 
 *  ./example_SPDHSSH2_tol.exe 3 80000 1e-8 1 1 ./point_set/3Dball_80000.csv 0.1 1e-2
 *  Input: 
 *      3 --> point dimension
 *      80000 --> number of points
 *      1e-8 --> relative error threshold for H2 matrix construction
 *      1 --> JIT-mode flag for H2 matrix
 *      1 --> index for kernel function, check parse_scalar_params for more details
 *      ./point_set/3Dball_80000.csv --> csv file that stores point coordinates
 *      0.1 -->  kernel parameter (can be two doubles)
 *      1e-2 --> diagonal shift
 */

int main(int argc, char **argv)
{
    srand48(time(NULL));

    //  Parse input
    parse_scalar_params(argc, argv);
    DTYPE single_krnl_param[1], double_krnl_param[2];   //  kernel parameter
    DTYPE shift = 0.0;                                  //  diagonal shift 
    if (argc == 9) //   parameter for a single-parametered kernel function
    {
        single_krnl_param[0] = (DTYPE)atof(argv[7]);
        shift = (DTYPE)atof(argv[8]);
        test_params.krnl_param = (void*) &single_krnl_param;
        printf("single parameter set to %f\n", single_krnl_param[0]);
    }
    if (argc == 10) //  parameters for a double-parametered kernel function
    {
        double_krnl_param[0] = (DTYPE)atof(argv[7]);
        double_krnl_param[1] = (DTYPE)atof(argv[8]);
        shift = (DTYPE)atof(argv[9]);
        test_params.krnl_param = (void*) &double_krnl_param;
        printf("double parameters set to %f and %f\n", double_krnl_param[0], double_krnl_param[1]);
    }

    H2Pack_p h2mat, hssmat;

    double st, et;  //  timing variable 
    DTYPE *x0, *x1, *y0, *y1;   //  intermediate vectors for matvec and solve
    x0 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    x1 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    assert(x0 != NULL && x1 != NULL);
    assert(y0 != NULL && y1 != NULL);

    /*
     *  PART 1: H2 matrix construction and error checking
     */
    //  H2 matrix initialization and partition
    H2P_init(&h2mat, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    H2P_calc_enclosing_box(test_params.pt_dim, test_params.n_point, test_params.coord, NULL, &h2mat->root_enbox);
    H2P_partition_points(h2mat, test_params.n_point, test_params.coord, 0, 0);
    H2P_HSS_calc_adm_inadm_pairs(h2mat);
    
    //  H2 matrix proxy points selection
    H2P_dense_mat_p *pp;
    st = get_wtime_sec();
    H2P_generate_proxy_point_ID_file(h2mat, test_params.krnl_param, test_params.krnl_eval, NULL, &pp);
    et = get_wtime_sec();
    printf("H2Pack generate proxy points used %.3lf (s)\n", et - st);
    
    //  H2 matrix construction
    H2P_build(
        h2mat, pp, test_params.BD_JIT, test_params.krnl_param, 
        test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
    );

    //  H2 matvec
    int n_check_pt = 20000, check_pt_s;
    if (n_check_pt >= test_params.n_point)
    {
        n_check_pt = test_params.n_point;
        check_pt_s = 0;
    } else {
        srand(time(NULL));
        check_pt_s = rand() % (test_params.n_point - n_check_pt);
    }
    for (int i = 0; i < test_params.krnl_mat_size; i++) 
        x0[i] = (DTYPE) pseudo_randn();
    H2P_matvec(h2mat, x0, y1);

    //  reference results
    direct_nbody(
        test_params.krnl_param, test_params.krnl_eval, test_params.pt_dim, test_params.krnl_dim, 
        test_params.coord,              test_params.n_point, test_params.n_point, x0, 
        test_params.coord + check_pt_s, test_params.n_point, n_check_pt,          y0
    );
    

    // H2 matvec error 
    DTYPE ref_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_dim * n_check_pt; i++)
    {
        DTYPE diff = y1[test_params.krnl_dim * check_pt_s + i] - y0[i];
        ref_norm += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("H2 matrix:\n");
    printf("For %d validation points: ||y_{H2} - y||_2 / ||y||_2 = %e\n", n_check_pt, err_norm / ref_norm);
    H2P_print_statistic(h2mat);


    /*
     *  Part 2: SPDHSS/FSAI/BJ preconditioning for PCG
     */

    // Preconditioned CG test
    const int max_iter = 3000;
    const DTYPE CG_tol = 1e-4;
    printf("\nDiagonal shift: sigma = %.2e\n", shift);

    //  Prepare a matvec reference result
    H2P_matvec(h2mat, x0, y0);
    for (int i = 0; i < test_params.krnl_mat_size; i++)
        y0[i] += shift * x0[i];

    //  SPDHSS-H2 construction
    int hss_rank = 100;
    DTYPE hss_tol = 1e-2;
    printf("\nConstructing SPDHSS from H2 with rel_tol %.2e\n", hss_tol);

    while (1)
    {
        printf("\nSPDHSS construction with %d random vector\n", hss_rank);
        H2P_SPDHSS_H2_build(hss_rank, hss_tol, shift, h2mat, &hssmat);

        int max_node_rank = 0;
        for (int i = 0; i < hssmat->n_UJ; i++)
        {
            int rank_i = hssmat->U[i]->ncol;
            if (rank_i > 0)
                max_node_rank = (rank_i > max_node_rank) ? rank_i : max_node_rank;
        }

        if (max_node_rank >= hss_rank - 5)
        {
            printf("Maximum approximation rank reached, reconstructing .... \n");
            hss_rank = 2 * hss_rank;
            H2P_destroy(&hssmat);
        }
        else
        {
            printf("SPDHSS maximum approximation rank: %d \n", max_node_rank);
            break;
        }
    }
 
    //  SPDHSS matvec and its error
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

    //  SPDHSS ULV Cholesky factorization
    H2P_HSS_ULV_Cholesky_factorize(hssmat, 0.0);

    //  SPDHSS ULV application 
    for (int i = 0; i < 10; i++)
        H2P_HSS_ULV_Cholesky_solve(hssmat, 3, y1, x1);
    printf("\nSPDHSS matrix with rank %d:\n", hss_rank);
    H2P_print_statistic(hssmat);

    //  PCG with SPDHSS
    pcg_tests(test_params.krnl_mat_size, h2mat, hssmat, shift, 0, max_iter, CG_tol, 5);
    H2P_destroy(&hssmat);
    fflush(stdout);

    //  Unpreconditioned
    pcg_tests(test_params.krnl_mat_size, h2mat, hssmat, shift, 0, max_iter, CG_tol, 1);
    fflush(stdout);

    free(x0);
    free(x1);
    free(y0);
    free(y1);
    free_aligned(test_params.coord);
    H2P_destroy(&h2mat);
}
