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
    int npt, pt_dim, max_k, ss_npt;
    DTYPE mu, l, *coord = NULL;
    if (argc < 8)
    {
        fprintf(stderr, "Usage: %s npt pt_dim max_k mu l ss_npt coord_bin\n", argv[0]);
        return 255;
    } else {
        npt     = atoi(argv[1]);
        pt_dim  = atoi(argv[2]);
        max_k   = atoi(argv[3]);
        mu      = atof(argv[4]);
        l       = atof(argv[5]);
        ss_npt  = atoi(argv[6]);
        coord   = (DTYPE*) malloc(sizeof(DTYPE) * npt * pt_dim);
        FILE *inf = fopen(argv[7], "rb");
        fread(coord, sizeof(DTYPE), npt * pt_dim, inf);
        fclose(inf);
    }
    INFO_PRINTF("Test kernel: k(x, y) = exp(- %.2f * ||x-y||_2^2), diagonal shift = %.2f\n", l, mu);
    INFO_PRINTF("%d points in %d-D, sample size = %d, max K11 block size = %d\n", npt, pt_dim, ss_npt, max_k);

    kernel_eval_fptr krnl_eval = Gaussian_3D_eval_intrin_t;
    double st, et;
    st = get_wtime_sec();
    int r = AFNi_rank_est(krnl_eval, &l, npt, pt_dim, coord, mu, max_k, ss_npt, 1);
    et = get_wtime_sec();
    printf("Estimated rank = %d, used time = %.2f s\n", r, et - st);

    free(coord);
    return 0;
}
