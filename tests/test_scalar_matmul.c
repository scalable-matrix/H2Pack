#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_kernels.h"
#include "H2Pack_utils.h"

#include "parse_scalar_params.h"
#include "direct_nbody.h"
#include "test_H2_matmul.h"

int main(int argc, char **argv)
{
    srand48(time(NULL));
    
    parse_scalar_params(argc, argv);

    H2Pack_t h2pack;
    double st, et;
    
    H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    
    H2P_partition_points(h2pack, test_params.n_point, test_params.coord, 0, 0);

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
    
    int n_vec = 16;
    test_H2_matmul(h2pack, n_vec);

    free_aligned(test_params.coord);
    H2P_destroy(h2pack);
}
