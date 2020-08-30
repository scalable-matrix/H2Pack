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

    H2Pack_p h2pack;
    double st, et;
    
    H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    
    H2P_calc_enclosing_box(test_params.pt_dim, test_params.n_point, test_params.coord, test_params.pp_fname, &h2pack->root_enbox);

    int max_leaf_points = 0;
    DTYPE max_leaf_size = 0.0;    
    H2P_partition_points(h2pack, test_params.n_point, test_params.coord, max_leaf_points, max_leaf_size);

    H2P_dense_mat_p *pp;
    st = get_wtime_sec();
    H2P_generate_proxy_point_ID_file(
        h2pack, test_params.krnl_param, test_params.krnl_eval,
        test_params.pp_fname, &pp
    );
    et = get_wtime_sec();
    printf("H2Pack load/generate proxy points used %.3lf (s)\n", et - st);
    
    H2P_build(
        h2pack, pp, test_params.BD_JIT, test_params.krnl_param, 
        test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
    );
    
    int n_vecs[10] = {2, 2, 4, 8, 12, 16, 20, 24, 28, 32};
    for (int i = 0; i < 10; i++)
        test_H2_matmul(h2pack, n_vecs[i]);

    h2pack->n_matvec = 0;  // Skip printing matvec timings
    H2P_print_statistic(h2pack);

    free_aligned(test_params.coord);
    H2P_destroy(&h2pack);

    return 0;
}
