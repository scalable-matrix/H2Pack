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
    int   krnl_bimv_flops;
    void  *krnl_param;
    DTYPE rel_tol;
    DTYPE *coord;
    DTYPE unit_cell[8];
    kernel_eval_fptr krnl_eval;
    kernel_eval_fptr pkrnl_eval;
    kernel_mv_fptr   krnl_mv;
};
struct H2P_test_params test_params;

DTYPE RPY_krnl_param[1] = {1.0};

void parse_RPY_Ewald_params(int argc, char **argv)
{
    test_params.pt_dim          = 3;
    test_params.xpt_dim         = test_params.pt_dim + 1;
    test_params.krnl_dim        = 3;
    test_params.BD_JIT          = 1;
    test_params.krnl_bimv_flops = RPY_krnl_bimv_flop;
    test_params.krnl_param      = (void*) &RPY_krnl_param[0];
    test_params.krnl_eval       = RPY_eval_std;
    //test_params.pkrnl_eval      = RPY_Ewald_eval_std;
    //test_params.krnl_mv         = RPY_krnl_mv_intrin_d;

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
    /*
    printf(
        "n_point, left corner, L = %d, (%lf, %lf, %lf), %lf\n", 
        test_params.n_point, test_params.unit_cell[0], test_params.unit_cell[1], 
        test_params.unit_cell[2], test_params.unit_cell[3]
    );
    */

    H2Pack_t ph2mat;
    
    H2P_init(&ph2mat, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);

    int max_leaf_points = 200;
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

    printf("n_r_adm_pair, n_r_inadm_pair = %d, %d\n", ph2mat->n_r_adm_pair, ph2mat->n_r_inadm_pair);
    FILE *ouf0 = fopen("add_r_adm_pairs.m", "w");
    fprintf(ouf0, "C_r_adm_pairs = [\n");
    for (int i = 0; i < ph2mat->n_r_adm_pair; i++)
    {
        fprintf(ouf0, "%d %d ", ph2mat->r_adm_pairs[2*i]+1, ph2mat->r_adm_pairs[2*i+1]+1);
        for (int j = 0; j < test_params.pt_dim; j++)
            fprintf(ouf0, "%.15lf ", ph2mat->per_adm_shifts[i * test_params.pt_dim + j]);
        fprintf(ouf0, "\n");
    }
    fprintf(ouf0, "];\n");
    fclose(ouf0);

    FILE *ouf1 = fopen("add_r_inadm_pairs.m", "w");
    fprintf(ouf1, "C_r_inadm_pairs = [\n");
    for (int i = 0; i < ph2mat->n_r_inadm_pair; i++)
    {
        fprintf(ouf1, "%d %d ", ph2mat->r_inadm_pairs[2*i]+1, ph2mat->r_inadm_pairs[2*i+1]+1);
        for (int j = 0; j < test_params.pt_dim; j++)
            fprintf(ouf1, "%.15lf ", ph2mat->per_inadm_shifts[i * test_params.pt_dim + j]);
        fprintf(ouf1, "\n");
    }
    fprintf(ouf1, "];\n");
    fclose(ouf1);

    H2P_destroy(ph2mat);
}