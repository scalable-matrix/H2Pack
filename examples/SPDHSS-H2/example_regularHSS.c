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
 *  Numerical test with diagonal-shifted scalar kernel matrices and regular HSS as preconditioner if it's SPD.
 *  
 *  Example run: 
 *  ./example_regularHSS.exe 3 80000 1e-8 1 1 ./point_set/3Dball_80000.csv 0.1 1e-2
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
    
    parse_scalar_params(argc, argv);
    DTYPE single_krnl_param[1], double_krnl_param[2];   //  kernel parameter
    DTYPE shift = 0.0;                                  //  diagonal shift 
    if (argc == 9) //   parameter for a single-parametered kernel function
    {
        single_krnl_param[0] = (DTYPE)atof(argv[7]);
        shift = (DTYPE)atof(argv[8]);
        test_params.krnl_param = (void*) &single_krnl_param;
        printf("Kernel function parameter is reset to %f\n", single_krnl_param[0]);
    }
    if (argc == 10) //  parameters for a double-parametered kernel function
    {
        double_krnl_param[0] = (DTYPE)atof(argv[7]);
        double_krnl_param[1] = (DTYPE)atof(argv[8]);
        shift = (DTYPE)atof(argv[9]);
        test_params.krnl_param = (void*) &double_krnl_param;
        printf("Kernel function parameters are reset to %f and %f\n", double_krnl_param[0], double_krnl_param[1]);
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

    //  H2 matvec error 
    DTYPE ref_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_dim * n_check_pt; i++)
    {
        DTYPE diff = y1[test_params.krnl_dim * check_pt_s + i] - y0[i];
        ref_norm += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);

    //  Print info of H2 matrix
    printf("H2 matrix:\n");
    printf("For %d validation points: ||y_{H2} - y||_2 / ||y||_2 = %e\n", n_check_pt, err_norm / ref_norm);
    H2P_print_statistic(h2mat);
    fflush(stdout);


    /*
     *  Part 2: Regular HSS construction and its preconditioned solving with H2 matvec
     */
    // Preconditioned CG test
    const int max_iter = 3000;
    const DTYPE CG_tol = 1e-4;
    printf("\nDiagonal shift: sigma = %.2e\n", shift);

    //  Prepare a matvec reference result
    H2P_matvec(h2mat, x0, y0);
    for (int i = 0; i < test_params.krnl_mat_size; i++)
        y0[i] += shift * x0[i];
    
    //  Regular HSS with rank 100 
    printf("\n =====    Regular HSS construction with rank = 100   =====\n");
    int hssrank = 100;

    //  HSS matrix initialization and partition
    H2P_init(&hssmat, test_params.pt_dim, test_params.krnl_dim, QR_RANK, &hssrank);
    H2P_run_HSS(hssmat);
    H2P_calc_enclosing_box(test_params.pt_dim, test_params.n_point, test_params.coord, NULL, &hssmat->root_enbox);
    H2P_partition_points(hssmat, test_params.n_point, test_params.coord, 0, 0);
    
    //  HSS matrix proxy point selection
    st = get_wtime_sec();
    H2P_dense_mat_p *pp1;
    H2P_generate_proxy_point_ID_file(hssmat, test_params.krnl_param, test_params.krnl_eval, NULL, &pp1);
    et = get_wtime_sec();
    printf("H2Pack generate proxy points used %.3lf (s)\n", et - st);

    //  HSS matrix construction
    H2P_build(
        hssmat, pp1, test_params.BD_JIT, test_params.krnl_param, 
        test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
    );

    //  HSS matvec 
    H2P_matvec(hssmat, x0, y1);
    for(int i = 0; i < test_params.krnl_mat_size; i++)
        y1[i] += shift * x0[i];

    //  HSS matvec error check (compared to H2 matvec)
    ref_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_mat_size; i++)
    {
        DTYPE diff = y1[i] - y0[i];
        ref_norm += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("||y_{HSS} - y_{H2}||_2 / ||y_{H2}||_2 = %e\n", err_norm / ref_norm);

    //  HSS ULV Cholesky factorization
    H2P_HSS_ULV_Cholesky_factorize(hssmat, shift);
    if (hssmat->is_HSS_SPD)
    {
        printf("The constructed HSS matrix is SPD\n Start PCG now");
        //  PCG solving
        pcg_tests(test_params.krnl_mat_size, h2mat, hssmat, shift, 0, max_iter, CG_tol, 5);
    }
    else
        printf("The constructed HSS matrix is non-SPD\n");

    printf("\nRegular HSS matrix with rank %d:\n", hssrank);
    H2P_print_statistic(hssmat);
    H2P_destroy(&hssmat);


    //  clean up
    free(x0);
    free(x1);
    free(y0);
    free(y1);
    free_aligned(test_params.coord);
    H2P_destroy(&h2mat);
}
