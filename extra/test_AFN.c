#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_kernels.h"
#include "AFN_precond.h"

int main(int argc, char **argv)
{
    int npt, pt_dim, max_k, ss_npt, fsai_npt;
    DTYPE mu, l, *coord = NULL;
    if (argc < 9)
    {
        printf("Usage: %s npt pt_dim l mu max_k ss_npt fsai_npt coord_bin\n", argv[0]);
        printf("  - npt       [int]    : Number of points\n");
        printf("  - pt_dim    [int]    : Point dimension\n");
        printf("  - l         [double] : Gaussian kernel parameter\n");
        printf("  - mu        [double] : Kernel matrix diagonal shift\n");
        printf("  - max_k     [int]    : Maximum global low-rank approximation rank\n");
        printf("  - ss_npt    [int]    : Number of points in the sample set\n");
        printf("  - fsai_npt  [int]    : Maximum number of nonzeros in each row of the FSAI matrix\n");
        printf("  - coord_bin [str]    : Binary file containing the coordinates, size pt_dim * npt,\n");
        printf("                         row major, each column is a point\n");
        return 255;
    } else {
        npt      = atoi(argv[1]);
        pt_dim   = atoi(argv[2]);
        l        = atof(argv[3]);
        mu       = atof(argv[4]);
        max_k    = atoi(argv[5]);
        ss_npt   = atoi(argv[6]);
        fsai_npt = atoi(argv[7]);
        coord    = (DTYPE*) malloc(sizeof(DTYPE) * npt * pt_dim);
        FILE *inf = fopen(argv[8], "rb");
        fread(coord, sizeof(DTYPE), npt * pt_dim, inf);
        fclose(inf);
    }
    INFO_PRINTF("Test kernel: k(x, y) = exp(- %.2f * ||x-y||_2^2), diagonal shift = %.4e\n", l, mu);
    INFO_PRINTF("%d points in %d-D, sample set size = %d\n", npt, pt_dim, ss_npt);
    INFO_PRINTF("Maximum global low-rank approximation rank = %d\n", max_k);
    INFO_PRINTF("Maximum number of nonzeros in each row of the FSAI matrix = %d\n", fsai_npt);

    AFN_precond_p AFN_precond = NULL;
    kernel_eval_fptr krnl_eval = Gaussian_3D_eval_intrin_t;
    void *krnl_param = &l;
    double st, et;
    st = get_wtime_sec();
    AFN_precond_init(krnl_eval, krnl_param, npt, pt_dim, coord, mu, max_k, ss_npt, fsai_npt, &AFN_precond);
    et = get_wtime_sec();
    printf("AFN_precond_init used time = %.2f s\n", et - st);

    DTYPE *x = (DTYPE *) malloc(sizeof(DTYPE) * npt);
    DTYPE *y = (DTYPE *) malloc(sizeof(DTYPE) * npt);
    for (int i = 0; i < npt; i++) x[i] = (i + 1) % 13 - 7.0;
    st = get_wtime_sec();
    AFN_precond_apply(AFN_precond, x, y);
    et = get_wtime_sec();
    printf("AFN_precond_apply used time = %.2f s\n", et - st);

    FILE *ouf = fopen("AFN_y.bin", "wb");
    fwrite(y, sizeof(DTYPE), npt, ouf);
    fclose(ouf);

    AFN_precond_destroy(&AFN_precond);

    free(coord);
    return 0;
}
