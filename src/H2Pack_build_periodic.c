#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_build_periodic.h"
#include "H2Pack_utils.h"
#include "utils.h"

// Build periodic block for root node
void H2P_build_periodic_block(H2Pack_t h2pack)
{
    int pt_dim    = h2pack->pt_dim;
    int xpt_dim   = h2pack->xpt_dim;
    int krnl_dim  = h2pack->krnl_dim;
    int root_idx  = h2pack->root_idx;
    int n_lattice = h2pack->n_lattice;
    void  *krnl_param   = h2pack->krnl_param;
    void  *pkrnl_param  = h2pack->pkrnl_param;
    DTYPE *enbox0_width = h2pack->enbox + (root_idx * (2 * pt_dim) + pt_dim);
    DTYPE *per_lattices = h2pack->per_lattices;
    H2P_dense_mat_t  root_J_coord   = h2pack->J_coord[root_idx];
    H2P_dense_mat_t  root_J_coord_s = h2pack->tb[0]->mat0;
    H2P_dense_mat_t  krnl_mat_blk   = h2pack->tb[0]->mat1;
    kernel_eval_fptr krnl_eval  = h2pack->krnl_eval;
    kernel_eval_fptr pkrnl_eval = h2pack->pkrnl_eval;

    int n_point_root = root_J_coord->ncol;
    int per_blk_size = n_point_root * krnl_dim;
    DTYPE *per_blk = (DTYPE*) malloc_aligned(sizeof(DTYPE) * per_blk_size * per_blk_size, 64);
    ASSERT_PRINTF(per_blk != NULL, "Failed to allocate periodic block of size %d^2\n", per_blk_size);

    // O = pkernel({root_J_coord, root_J_coord});
    pkrnl_eval(
        root_J_coord->data, root_J_coord->ld, root_J_coord->ncol,
        root_J_coord->data, root_J_coord->ld, root_J_coord->ncol,
        pkrnl_param, per_blk, per_blk_size
    );
    DTYPE shift[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    H2P_dense_mat_resize(krnl_mat_blk, per_blk_size, per_blk_size);
    H2P_dense_mat_resize(root_J_coord_s, xpt_dim, n_point_root);
    H2P_copy_matrix_block(
        xpt_dim, n_point_root, root_J_coord->data, root_J_coord->ld, 
        root_J_coord_s->data, root_J_coord_s->ld
    );
    for (int l = 0; l < n_lattice; l++)
    {
        // shift = lattice(l, 1 : pt_dim) .* root_box(pt_dim+1 : 2 * pt_dim);
        // shift = [shift, zeros(1, xpt_dim - pt_dim)];
        DTYPE *lattice_l = per_lattices + l * pt_dim;
        for (int j = 0; j < pt_dim; j++) shift[j] = enbox0_width[j] * lattice_l[j];
        // root_J_coord_s = coord_shift(root_J_coord, shift, 1);
        H2P_shift_coord(root_J_coord_s, shift,  1.0);
        // O = O - kernel({root_J_coord, root_J_coord_s});
        krnl_eval(
            root_J_coord->data,   root_J_coord->ld,   root_J_coord->ncol,
            root_J_coord_s->data, root_J_coord_s->ld, root_J_coord->ncol,
            krnl_param, krnl_mat_blk->data, krnl_mat_blk->ld
        );
        #pragma omp simd
        for (int i = 0; i < per_blk_size * per_blk_size; i++)
            per_blk[i] -= krnl_mat_blk->data[i];
        // Reset root_J_coord_s = root_J_coord
        H2P_shift_coord(root_J_coord_s, shift, -1.0);
    }

    h2pack->per_blk = per_blk;
}

// Build H2 representation with a regular kernel function and
// a periodic system kernel (Ewald summation) function
void H2P_build_periodic(
    H2Pack_t h2pack, H2P_dense_mat_t *pp, const int BD_JIT, 
    void *krnl_param,  kernel_eval_fptr krnl_eval, 
    void *pkrnl_param, kernel_eval_fptr pkrnl_eval, 
    kernel_mv_fptr krnl_mv, const int krnl_mv_flops
)
{
    double st, et;
    double *timers = h2pack->timers;

    if (pp == NULL)
    {
        ERROR_PRINTF("You need to provide a set of proxy points.\n");
        return;
    }
    
    if (krnl_eval == NULL)
    {
        ERROR_PRINTF("You need to provide a valid krnl_eval().\n");
        return;
    }

    if (BD_JIT != 1)
    {
        ERROR_PRINTF("Only support BD_JIT=1 in this function for the moment.\n");
        return;
    }

    h2pack->pp = pp;
    h2pack->BD_JIT = BD_JIT;
    h2pack->krnl_param  = krnl_param;
    h2pack->krnl_eval   = krnl_eval;
    h2pack->pkrnl_param = pkrnl_param;
    h2pack->pkrnl_eval  = pkrnl_eval;
    h2pack->krnl_mv     = krnl_mv;
    h2pack->krnl_bimv_flops = krnl_mv_flops - 2;
    if (BD_JIT == 1 && krnl_mv == NULL) 
        WARNING_PRINTF("krnl_eval() will be used in BD_JIT matvec. For better performance, consider using a krnl_mv().\n");

    // 1. Build projection matrices and skeleton row sets
    st = get_wtime_sec();
    H2P_build_H2_UJ_proxy(h2pack);
    et = get_wtime_sec();
    timers[_U_BUILD_TIMER_IDX] = et - st;

    // 2. Generate H2 generator matrices metadata
    st = get_wtime_sec();
    H2P_generate_B_metadata(h2pack);
    et = get_wtime_sec();
    timers[_B_BUILD_TIMER_IDX] = et - st;
    
    // 3. Generate H2 dense blocks metadata
    st = get_wtime_sec();
    H2P_generate_D_metadata(h2pack);
    et = get_wtime_sec();
    timers[_D_BUILD_TIMER_IDX] = et - st;

    // 4. Build periodic block for root node, add its timing to B build timing
    st = get_wtime_sec();
    H2P_build_periodic_block(h2pack);
    et = get_wtime_sec();
    timers[_B_BUILD_TIMER_IDX] = et - st;

    // 5. Set up forward and backward permutation indices
    int n_point    = h2pack->n_point;
    int krnl_dim   = h2pack->krnl_dim;
    int *coord_idx = h2pack->coord_idx;
    int *fwd_pmt_idx = (int*) malloc(sizeof(int) * n_point * krnl_dim);
    int *bwd_pmt_idx = (int*) malloc(sizeof(int) * n_point * krnl_dim);
    for (int i = 0; i < n_point; i++)
    {
        for (int j = 0; j < krnl_dim; j++)
        {
            fwd_pmt_idx[i * krnl_dim + j] = coord_idx[i] * krnl_dim + j;
            bwd_pmt_idx[coord_idx[i] * krnl_dim + j] = i * krnl_dim + j;
        }
    }
    h2pack->fwd_pmt_idx = fwd_pmt_idx;
    h2pack->bwd_pmt_idx = bwd_pmt_idx;
}
