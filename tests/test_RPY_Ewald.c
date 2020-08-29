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

struct H2P_test_params
{
    int   pt_dim;
    int   xpt_dim;
    int   krnl_dim;
    int   n_point;
    int   krnl_mat_size;
    int   BD_JIT;
    int   krnl_mv_flops;
    DTYPE rel_tol;
    DTYPE *coord;
    DTYPE unit_cell[8];
    kernel_eval_fptr krnl_eval;
    kernel_eval_fptr pkrnl_eval;
    kernel_mv_fptr   krnl_mv;
};
struct H2P_test_params test_params;

void parse_RPY_Ewald_params(int argc, char **argv)
{
    test_params.pt_dim        = 3;
    test_params.xpt_dim       = test_params.pt_dim + 1;
    test_params.krnl_dim      = 3;
    test_params.BD_JIT        = 1;
    test_params.krnl_mv_flops = RPY_krnl_mv_flop;
    test_params.krnl_eval     = RPY_eval_std;
    test_params.pkrnl_eval    = RPY_Ewald_eval_std;
    test_params.krnl_mv       = RPY_krnl_mv_intrin_d;

    if (argc < 2)
    {
        printf("Number of points   = ");
        scanf("%d", &test_params.n_point);
    } else {
        test_params.n_point = atoi(argv[1]);
        printf("Number of points   = %d\n", test_params.n_point);
    }
    test_params.krnl_mat_size = test_params.n_point * test_params.krnl_dim;

    if (argc < 3)
    {
        printf("QR relative tol    = ");
        scanf("%lf", &test_params.rel_tol);
    } else {
        test_params.rel_tol = atof(argv[2]);
        printf("QR relative tol    = %e\n", test_params.rel_tol);
    }

    test_params.coord = (DTYPE*) malloc_aligned(sizeof(DTYPE) * test_params.n_point * test_params.xpt_dim, 64);
    assert(test_params.coord != NULL);

    // Note: coordinates need to be stored in column-major style, i.e. test_params.coord 
    // is row-major and each column stores the coordinate of a point. 
    int need_gen = 1;
    if (argc >= 4)
    {
        if (strstr(argv[3], ".csv") != NULL)
        {
            printf("Reading coordinates from CSV file...");

            DTYPE *tmp = (DTYPE*) malloc(sizeof(DTYPE) * test_params.n_point * test_params.xpt_dim);
            FILE *inf = fopen(argv[3], "r");

            // Size of unit box, should be the same value
            for (int i = 0; i < test_params.pt_dim; i++)
                fscanf(inf, "%lf,", &test_params.unit_cell[test_params.pt_dim + i]);
            DTYPE dummy;
            for (int i = test_params.pt_dim; i < test_params.xpt_dim; i++) fscanf(inf, "%lf,", &dummy);

            // Point coordinates and radii
            for (int i = 0; i < test_params.n_point; i++)
            {
                for (int j = 0; j < test_params.xpt_dim-1; j++) 
                    fscanf(inf, "%lf,", &tmp[i * test_params.xpt_dim + j]);
                fscanf(inf, "%lf\n", &tmp[i * test_params.xpt_dim + test_params.xpt_dim-1]);
            }
            fclose(inf);

            #if 0
            // Find the left-most corner
            for (int i = 0; i < test_params.pt_dim; i++)
                test_params.unit_cell[i] = tmp[i];
            for (int i = 1; i < test_params.n_point; i++)
            {
                DTYPE *coord_i = tmp + i * test_params.xpt_dim;
                for (int j = 0; j < test_params.pt_dim; j++)
                    if (coord_i[j] < test_params.unit_cell[j]) test_params.unit_cell[j] = coord_i[j];
            }
            #endif
            // Manually override the left-corner of unit cell as the original point, should be removed later
            for (int i = 0; i < test_params.pt_dim; i++) test_params.unit_cell[i] = 0.0;

            // Transpose the coordinate array
            for (int i = 0; i < test_params.xpt_dim; i++)
            {
                for (int j = 0; j < test_params.n_point; j++)
                    test_params.coord[i * test_params.n_point + j] = tmp[j * test_params.xpt_dim + i];
            }
            free(tmp);

            printf(" done.\n");
            need_gen = 0;
        }  // End of "if (strstr(argv[3], ".csv") != NULL)"
    }  // End of "if (argc >= 4)"

    if (need_gen == 1)
    {
        DTYPE *x = test_params.coord;
        DTYPE *y = test_params.coord + test_params.n_point;
        DTYPE *z = test_params.coord + test_params.n_point * 2;
        DTYPE *a = test_params.coord + test_params.n_point * 3;
        DTYPE sum_a3 = 0.0;
        for (int i = 0; i < test_params.n_point; i++)
        {
            a[i] = 0.5 + 5.0 * (DTYPE) drand48();
            sum_a3 += a[i] * a[i] * a[i];
        }
        DTYPE vol_frac = 0.1;
        DTYPE base = 4.0 / 3.0 * M_PI * sum_a3 / vol_frac;
        DTYPE expn = 1.0 / (DTYPE) test_params.pt_dim;
        DTYPE prefac = DPOW(base, expn);
        printf("CSV coordinate file not provided. Generating random coordinates in box [0, %.3lf]^%d...", prefac, test_params.pt_dim);
        for (int i = 0; i < test_params.n_point; i++)
        {
            x[i] = (DTYPE) drand48() * prefac;
            y[i] = (DTYPE) drand48() * prefac;
            z[i] = (DTYPE) drand48() * prefac;
        }
        // Unit cell has left corner at the original point and size == prefac
        for (int i = 0; i < test_params.pt_dim; i++)
        {
            test_params.unit_cell[i] = 0.0;
            test_params.unit_cell[i + test_params.pt_dim] = prefac;
        }
        printf(" done.\n");
    }  // End of "if (need_gen == 1)"
}

int main(int argc, char **argv)
{
    srand48(time(NULL));

    parse_RPY_Ewald_params(argc, argv);

    H2Pack_p ph2mat;
    
    H2P_init(&ph2mat, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);

    // Partition points and generate (in)adm pairs
    int max_leaf_points = 300;
    DTYPE max_leaf_size = 0.0;
    H2P_run_RPY_Ewald(ph2mat);
    // We need to ensure the size of each leaf box >= 2 * max(radii), but the 
    // stopping criteria is "if (box_size <= max_leaf_size)", so max_leaf_size
    // should be set as 4 * max(radii).
    DTYPE *radii = test_params.coord + 3 * test_params.n_point; 
    for (int i = 0; i < test_params.n_point; i++)
        max_leaf_size = (max_leaf_size < radii[i]) ? radii[i] : max_leaf_size;
    max_leaf_size *= 4.0;
    H2P_partition_points_periodic(
        ph2mat, test_params.n_point, test_params.coord, max_leaf_points, 
        max_leaf_size, test_params.unit_cell
    );

    // Generate surface proxy points
    H2P_dense_mat_p *pp;
    DTYPE max_L = ph2mat->enbox[ph2mat->root_idx * 2 * test_params.pt_dim + test_params.pt_dim];
    int min_level = ph2mat->min_adm_level;
    int max_level = ph2mat->max_level;
    int num_pp    = 400;
    H2P_generate_proxy_point_surface(
        test_params.pt_dim, test_params.xpt_dim, num_pp, 
        max_level, min_level, max_L, &pp
    );

    // Set up RPY Ewald parameters and work buffer
    const int nr = 4, nk = 4;
    const DTYPE eta = 1.0 / 6.0 / M_PI;
    const DTYPE L   = test_params.unit_cell[3];
    const DTYPE xi  = DSQRT(M_PI) / L;
    DTYPE RPY_param[1] = {eta};
    DTYPE RPY_Ewald_param[5], *ewald_workbuf;
    RPY_Ewald_param[0] = L;
    RPY_Ewald_param[1] = xi;
    RPY_Ewald_param[2] = (DTYPE) nr;
    RPY_Ewald_param[3] = (DTYPE) nk;
    RPY_Ewald_init_workbuf(L, xi, nr, nk, &ewald_workbuf);
    memcpy(RPY_Ewald_param + 4, &ewald_workbuf, sizeof(DTYPE*));

    // Build H2 matrix for RPY Ewald
    H2P_build_periodic(
        ph2mat, pp, test_params.BD_JIT, RPY_param, test_params.krnl_eval,
        RPY_Ewald_param, test_params.pkrnl_eval, test_params.krnl_mv, 
        test_params.krnl_mv_flops
    );

    int n_check_pt = 1000, check_pt_s;
    if (n_check_pt >= test_params.n_point)
    {
        n_check_pt = test_params.n_point;
        check_pt_s = 0;
    } else {
        srand(time(NULL));
        check_pt_s = rand() % (test_params.n_point - n_check_pt);
    }
    printf("Calculating direct n-body reference result for points %d -> %d\n", check_pt_s, check_pt_s + n_check_pt - 1);
    
    int n_vec = 10, krnl_mat_size = test_params.krnl_mat_size;
    DTYPE *x, *y0, *y1, *y2;
    x  = (DTYPE*) malloc(sizeof(DTYPE) * krnl_mat_size * n_vec);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * krnl_mat_size);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * krnl_mat_size * n_vec);
    y2 = (DTYPE*) malloc(sizeof(DTYPE) * krnl_mat_size * n_vec);
    assert(x != NULL && y0 != NULL && y1 != NULL && y2 != NULL);
    for (int i = 0; i < krnl_mat_size * n_vec; i++) 
    {
        x[i] = (DTYPE) drand48() - 0.5;
    }

    // Get reference results
    direct_nbody(
        RPY_Ewald_param, test_params.pkrnl_eval, test_params.pt_dim, test_params.krnl_dim, 
        test_params.coord,              test_params.n_point, test_params.n_point, x, 
        test_params.coord + check_pt_s, test_params.n_point, n_check_pt,          y0
    );

    // Verify H2 matvec results
    H2P_matvec_periodic(ph2mat, x, y1); 
    DTYPE ref_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_dim * n_check_pt; i++)
    {
        DTYPE y1_i = y1[test_params.krnl_dim * check_pt_s + i];
        DTYPE diff = y1_i - y0[i];
        ref_norm += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    ref_norm = DSQRT(ref_norm);
    err_norm = DSQRT(err_norm);
    printf("For %d validation points: ||y_{H2} - y||_2 / ||y||_2 = %e\n", n_check_pt, err_norm / ref_norm);

    // Test matvec performance
    H2P_reset_timers(ph2mat);
    for (int i = 0; i < n_vec; i++) 
    {
        DTYPE *x_i  = x  + i * krnl_mat_size;
        DTYPE *y1_i = y1 + i * krnl_mat_size;
        H2P_matvec_periodic(ph2mat, x_i, y1_i);
    }

    H2P_print_statistic(ph2mat);

    // Test column-major matmul performance
    double st = get_wtime_sec();
    H2P_matmul_periodic(ph2mat, CblasColMajor, n_vec, x, krnl_mat_size, y2, krnl_mat_size);
    double et = get_wtime_sec();
    printf("One col-major matmul used %.3lf sec\n", et - st);

    // Check H2 column-major matmul results
    DTYPE cm_max_relerr = 0.0;
    DTYPE cm_avg_relerr = 0.0; 
    DTYPE y0_2norm, err_2norm, relerr;
    for (int i = 0; i < n_vec; i++)
    {
        DTYPE *y1_ivec = y1 + i * krnl_mat_size;
        DTYPE *y2_ivec = y2 + i * krnl_mat_size;
        calc_err_2norm(krnl_mat_size, y1_ivec, y2_ivec, &y0_2norm, &err_2norm);
        relerr = err_2norm / y0_2norm;
        if (relerr > cm_max_relerr) cm_max_relerr = relerr;
        cm_avg_relerr += relerr;
    }
    cm_avg_relerr /= (DTYPE) n_vec;
    printf("%d vectors col-major matmul max/avg relerr = %e, %e\n", n_vec, cm_max_relerr, cm_avg_relerr);

    free(x);
    free(y0);
    free(y1);
    free(y2);
    free(ewald_workbuf);
    H2P_destroy(ph2mat);
}