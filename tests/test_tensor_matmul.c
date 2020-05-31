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

#include "parse_tensor_params.h"
#include "direct_nbody.h"
#include "test_H2_matmul.h"

int main(int argc, char **argv)
{
    srand48(time(NULL));
    
    parse_tensor_params(argc, argv);
    
    H2Pack_t h2pack;
    double st, et;
    
    H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    
    int max_leaf_points = 300;
    DTYPE max_leaf_size = 0.0;
    // Some special settings for RPY
    if (test_params.kernel_id == 1) 
    {
        H2P_run_RPY(h2pack);
        // We need to ensure the size of each leaf box >= 2 * max(radii), but the 
        // stopping criteria is "if (box_size <= max_leaf_size)", so max_leaf_size
        // should be set as 4 * max(radii).
        DTYPE *radii = test_params.coord + 3 * test_params.n_point; 
        for (int i = 0; i < test_params.n_point; i++)
            max_leaf_size = (max_leaf_size < radii[i]) ? radii[i] : max_leaf_size;
        max_leaf_size *= 4.0;
    }
    H2P_partition_points(h2pack, test_params.n_point, test_params.coord, max_leaf_points, max_leaf_size);

    H2P_dense_mat_t *pp;
    DTYPE max_L = h2pack->enbox[h2pack->root_idx * 2 * test_params.pt_dim + test_params.pt_dim];
    int start_level = 2;
    int num_pp = ceil(-log10(test_params.rel_tol)) - 1;
    if (num_pp < 4 ) num_pp = 4;
    if (num_pp > 10) num_pp = 10;
    num_pp = 6 * num_pp * num_pp;
    H2P_generate_proxy_point_surface(
        test_params.pt_dim, test_params.xpt_dim, num_pp, 
        h2pack->max_level, start_level, max_L, &pp
    );
    
    H2P_build(
        h2pack, pp, test_params.BD_JIT, test_params.krnl_param, 
        test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
    );
    
    const int n_vecs[13] = {2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64};
    for (int i = 0; i < 13; i++)
        test_H2_matmul(h2pack, n_vecs[i]);

    h2pack->n_matvec = 0;  // Skip printing matvec timings
    H2P_print_statistic(h2pack);

    free_aligned(test_params.coord);
    H2P_destroy(h2pack);
}
