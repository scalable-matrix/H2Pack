#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_matmul.h"
#include "H2Pack_matmul_periodic.h"
#include "H2Pack_utils.h"
#include "utils.h"

// H2 matmul intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
void H2P_matmul_periodic_intmd_mult(
    H2Pack_p h2pack, const int n_vec, 
    const DTYPE *mat_x, const int ldx, const int x_row_stride, const CBLAS_TRANSPOSE x_trans,
          DTYPE *mat_y, const int ldy, const int y_row_stride, const CBLAS_TRANSPOSE y_trans
)
{
    int   pt_dim          = h2pack->pt_dim;
    int   xpt_dim         = h2pack->xpt_dim;
    int   n_point         = h2pack->n_point;
    int   n_node          = h2pack->n_node;
    int   n_thread        = h2pack->n_thread;
    int   *node_level     = h2pack->node_level;
    int   *pt_cluster     = h2pack->pt_cluster;
    int   *mat_cluster    = h2pack->mat_cluster;
    int   *B_nrow         = h2pack->B_nrow;
    int   *B_ncol         = h2pack->B_ncol;
    int   *B_p2i_rowptr   = h2pack->B_p2i_rowptr;
    int   *B_p2i_colidx   = h2pack->B_p2i_colidx;
    int   *B_p2i_val      = h2pack->B_p2i_val;
    DTYPE *coord          = h2pack->coord;
    DTYPE *per_adm_shifts = h2pack->per_adm_shifts;
    void  *krnl_param     = h2pack->krnl_param;
    H2P_dense_mat_p  *y0 = h2pack->y0;
    H2P_dense_mat_p  *J_coord = h2pack->J_coord;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    H2P_thread_buf_p *thread_buf = h2pack->tb;

    // 1. Initialize y1 on the first run or reset the size of each y1
    H2P_matmul_init_y1(h2pack, n_vec);
    H2P_dense_mat_p *y1 = h2pack->y1;

    // 2. Intermediate sweep
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p Bij      = thread_buf[tid]->mat0;
        H2P_dense_mat_p coord1_s = thread_buf[tid]->mat1;
        DTYPE shift[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int level0 = node_level[node0];
            
            H2P_dense_mat_p y1_0 = y1[node0];
            memset(y1_0->data, 0, sizeof(DTYPE) * y1_0->nrow * y1_0->ncol);

            for (int i = B_p2i_rowptr[node0]; i < B_p2i_rowptr[node0 + 1]; i++)
            {
                int node1    = B_p2i_colidx[i];
                int pair_idx = B_p2i_val[i] - 1;
                int level1   = node_level[node1];

                H2P_dense_mat_p y0_1 = y0[node1];

                DTYPE *per_adm_shift_i = per_adm_shifts + pair_idx * pt_dim;
                for (int k = 0; k < pt_dim; k++) shift[k] = per_adm_shift_i[k];

                int Bij_nrow = B_nrow[pair_idx];
                int Bij_ncol = B_ncol[pair_idx];
                H2P_dense_mat_resize(Bij, Bij_nrow, Bij_ncol);

                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    H2P_dense_mat_copy(J_coord[node1], coord1_s);
                    H2P_shift_coord(coord1_s, shift, 1.0);
                    krnl_eval(
                        J_coord[node0]->data, J_coord[node0]->ld, J_coord[node0]->ncol,
                        coord1_s->data,       coord1_s->ld,       coord1_s->ncol,
                        krnl_param, Bij->data, Bij->ld
                    );
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, Bij_nrow, n_vec, Bij_ncol,
                        1.0, Bij->data, Bij->ld, y0_1->data, y0_1->ld, 1.0, y1_0->data, y1_0->ld
                    );
                }  // End of "if (level0 == level1)"

                // (2) node1 is a leaf node and its level is larger than node0,
                //     only compress on node0's side
                if (level0 > level1)
                {
                    int pt_s1     = pt_cluster[node1 * 2];
                    int node1_npt = pt_cluster[node1 * 2 + 1] - pt_s1 + 1;
                    int vec_s1    = mat_cluster[node1 * 2];
                    
                    H2P_dense_mat_resize(coord1_s, xpt_dim, node1_npt);
                    copy_matrix(sizeof(DTYPE), xpt_dim, node1_npt, coord + pt_s1, n_point, coord1_s->data, coord1_s->ld, 0);
                    H2P_shift_coord(coord1_s, shift, 1.0);

                    krnl_eval(
                        J_coord[node0]->data, J_coord[node0]->ld, J_coord[node0]->ncol,
                        coord1_s->data,       coord1_s->ld,       coord1_s->ncol,
                        krnl_param, Bij->data, Bij->ld
                    );
                    const DTYPE *mat_x_spos = mat_x + vec_s1 * x_row_stride;
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, x_trans, Bij_nrow, n_vec, Bij_ncol,
                        1.0, Bij->data, Bij->ld, mat_x_spos, ldx, 1.0, y1_0->data, y1_0->ld
                    );
                }  // End of "if (level0 > level1)"

                // (3) node0 is a leaf node and its level is larger than node1,
                //     only compress on node1's side
                if (level0 < level1)
                {
                    int pt_s0     = pt_cluster[node0 * 2];
                    int node0_npt = pt_cluster[node0 * 2 + 1] - pt_s0 + 1;
                    int vec_s0    = mat_cluster[node0 * 2];

                    H2P_dense_mat_copy(J_coord[node1], coord1_s);
                    H2P_shift_coord(coord1_s, shift, 1.0);

                    krnl_eval(
                        coord + pt_s0,  n_point,      node0_npt,
                        coord1_s->data, coord1_s->ld, coord1_s->ncol,
                        krnl_param, Bij->data, Bij->ld
                    );
                    DTYPE *mat_y_spos = mat_y + vec_s0 * y_row_stride;
                    if (y_trans == CblasNoTrans)
                    {
                        CBLAS_GEMM(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans, Bij_nrow, n_vec, Bij_ncol,
                            1.0, Bij->data, Bij->ld, y0_1->data, y0_1->ld, 1.0, mat_y_spos, ldy
                        );
                    } else {
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasTrans, n_vec, Bij_nrow, Bij_ncol,
                            1.0, y0_1->data, y0_1->ld, Bij->data, Bij->ld, 1.0, mat_y_spos, ldy
                        );
                    }
                }  // End of "if (level0 < level1)"
            }  // End of i loop
        }  // End of node0 loop
    }  // End of "#pragma omp parallel"
}

// H2 matmul dense multiplication, calculate D_{ij} * x_j
void H2P_matmul_periodic_dense_mult(
    H2Pack_p h2pack, const int n_vec, 
    const DTYPE *mat_x, const int ldx, const int x_row_stride, const CBLAS_TRANSPOSE x_trans,
          DTYPE *mat_y, const int ldy, const int y_row_stride, const CBLAS_TRANSPOSE y_trans
)
{
    int   pt_dim            = h2pack->pt_dim;
    int   xpt_dim           = h2pack->xpt_dim;
    int   n_point           = h2pack->n_point;
    int   n_node            = h2pack->n_node;
    int   n_leaf_node       = h2pack->n_leaf_node;
    int   n_thread          = h2pack->n_thread;
    int   *pt_cluster       = h2pack->pt_cluster;
    int   *mat_cluster      = h2pack->mat_cluster;
    int   *D_nrow           = h2pack->D_nrow;
    int   *D_ncol           = h2pack->D_ncol;
    int   *D_p2i_rowptr     = h2pack->D_p2i_rowptr;
    int   *D_p2i_colidx     = h2pack->D_p2i_colidx;
    int   *D_p2i_val        = h2pack->D_p2i_val;
    DTYPE *coord            = h2pack->coord;
    DTYPE *per_inadm_shifts = h2pack->per_inadm_shifts;
    void  *krnl_param       = h2pack->krnl_param;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    H2P_thread_buf_p *thread_buf = h2pack->tb;

    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p Dij      = thread_buf[tid]->mat0;
        H2P_dense_mat_p coord1_s = thread_buf[tid]->mat1;

        DTYPE shift[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int pt_s0     = pt_cluster[2 * node0];
            int node0_npt = pt_cluster[2 * node0 + 1] - pt_s0 + 1;
            int y_offset  = mat_cluster[2 * node0];
            DTYPE *mat_y_spos = mat_y + y_offset * y_row_stride;

            for (int i = D_p2i_rowptr[node0]; i < D_p2i_rowptr[node0 + 1]; i++)
            {
                int node1     = D_p2i_colidx[i];
                int pair_idx  = D_p2i_val[i] - 1;
                int pt_s1     = pt_cluster[2 * node1];
                int node1_npt = pt_cluster[2 * node1 + 1] - pt_s1 + 1;
                int x_offset  = mat_cluster[2 * node1];
                const DTYPE *mat_x_spos = mat_x + x_offset * x_row_stride;
                
                int Dij_nrow = D_nrow[pair_idx];
                int Dij_ncol = D_ncol[pair_idx];
                H2P_dense_mat_resize(Dij, Dij_nrow, Dij_ncol);

                if (pair_idx < n_leaf_node)
                {
                    // (i, i) pair, no shift
                    for (int k = 0; k < pt_dim; k++) shift[k] = 0.0;
                } else {
                    // The (pair_idx - n_leaf_node)-th inadmissible pair, need shifting
                    DTYPE *per_inadm_shift_i = per_inadm_shifts + (pair_idx - n_leaf_node) * pt_dim;
                    for (int k = 0; k < pt_dim; k++) shift[k] = per_inadm_shift_i[k];
                }

                H2P_dense_mat_resize(coord1_s, xpt_dim, node1_npt);
                copy_matrix(sizeof(DTYPE), xpt_dim, node1_npt, coord + pt_s1, n_point, coord1_s->data, coord1_s->ld, 0);
                H2P_shift_coord(coord1_s, shift, 1.0);
                krnl_eval(
                    coord + pt_s0,  n_point,      node0_npt,
                    coord1_s->data, coord1_s->ld, coord1_s->ncol,
                    krnl_param, Dij->data, Dij->ld
                );

                if (x_trans == CblasNoTrans)
                {
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, Dij_nrow, n_vec, Dij_ncol,
                        1.0, Dij->data, Dij->ld, mat_x_spos, ldx, 1.0, mat_y_spos, ldy
                    );
                } else {
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasTrans, n_vec, Dij_nrow, Dij_ncol,
                        1.0, mat_x_spos, ldx, Dij->data, Dij->ld, 1.0, mat_y_spos, ldy
                    );
                }  // End of "if (x_trans == CblasNoTrans)"
            }  // End of i loop
        }  // End of node0 loop
    }  // End of "#pragma omp parallel"
}

// H2 representation multiplies a dense general matrix
void H2P_matmul_periodic(
    H2Pack_p h2pack, const CBLAS_LAYOUT layout, const int n_vec, 
    const DTYPE *mat_x, const int ldx, DTYPE *mat_y, const int ldy
)
{
    double st, et;
    int    krnl_mat_size = h2pack->krnl_mat_size;
    int    mm_max_n_vec  = h2pack->mm_max_n_vec;
    double *timers       = h2pack->timers;
    size_t *mat_size     = h2pack->mat_size;

    size_t pmt_xy_size = (size_t) krnl_mat_size * (size_t) mm_max_n_vec;
    free(h2pack->pmt_x);
    free(h2pack->pmt_y);
    h2pack->pmt_x = (DTYPE*) malloc(sizeof(DTYPE) * pmt_xy_size);
    h2pack->pmt_y = (DTYPE*) malloc(sizeof(DTYPE) * pmt_xy_size);
    ASSERT_PRINTF(
        h2pack->pmt_x != NULL && h2pack->pmt_y != NULL,
        "Failed to allocate working arrays of size %zu for matmul\n", 2 * pmt_xy_size
    );
    DTYPE *pmt_x = h2pack->pmt_x;
    DTYPE *pmt_y = h2pack->pmt_y;

    int x_col_stride, y_col_stride, pmt_row_stride, ld_pmt;
    CBLAS_TRANSPOSE x_trans, y_trans;
    if (layout == CblasRowMajor)
    {
        x_col_stride   = 1;
        y_col_stride   = 1;
        ld_pmt         = mm_max_n_vec;
        pmt_row_stride = ld_pmt;
        x_trans = CblasNoTrans;
        y_trans = CblasNoTrans;
    } else {
        x_col_stride   = ldx;
        y_col_stride   = ldy;
        ld_pmt         = krnl_mat_size;
        pmt_row_stride = 1;
        x_trans = CblasTrans;
        y_trans = CblasTrans;
    }

    for (int i_vec = 0; i_vec < n_vec; i_vec += mm_max_n_vec)
    {
        int curr_n_vec = (i_vec + mm_max_n_vec <= n_vec) ? mm_max_n_vec : (n_vec - i_vec);
        const DTYPE *curr_mat_x = mat_x + i_vec * x_col_stride;
        DTYPE *curr_mat_y = mat_y + i_vec * y_col_stride;

        // 1. Forward permute input matrix block
        st = get_wtime_sec();
        H2P_permute_matrix_row_forward(h2pack, layout, curr_n_vec, curr_mat_x, ldx, pmt_x, ld_pmt);
        et = get_wtime_sec();
        timers[MV_VOP_TIMER_IDX] += et - st;
        mat_size[MV_VOP_SIZE_IDX] += 2 * krnl_mat_size * curr_n_vec;
    
        // 2. Reset output matrix
        st = get_wtime_sec();
        if (layout == CblasRowMajor)
        {
            size_t row_msize = sizeof(DTYPE) * curr_n_vec;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < krnl_mat_size; i++)
            {
                DTYPE *mat_y_i = pmt_y + i * ld_pmt;
                memset(mat_y_i, 0, row_msize);
            }
        } else {
            #pragma omp parallel
            {
                for (int i = 0; i < curr_n_vec; i++)
                {
                    DTYPE *mat_y_i = pmt_y + i * ld_pmt;
                    #pragma omp for schedule(static)
                    for (int j = 0; j < krnl_mat_size; j++) mat_y_i[j] = 0.0;
                }
            }
        }  // End of "if (layout == CblasRowMajor)"
        et = get_wtime_sec();
        timers[MV_VOP_TIMER_IDX] += et - st;
        mat_size[MV_VOP_SIZE_IDX] += krnl_mat_size * curr_n_vec;
        
        // 3. Forward transformation, calculate U_j^T * x_j
        st = get_wtime_sec();
        H2P_matmul_fwd_transform(
            h2pack, curr_n_vec, 
            pmt_x, ld_pmt, pmt_row_stride, x_trans
        );
        et = get_wtime_sec();
        timers[MV_FWD_TIMER_IDX] += et - st;

        // 4. Intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
        st = get_wtime_sec();
        H2P_matmul_periodic_intmd_mult(
            h2pack, curr_n_vec, 
            pmt_x, ld_pmt, pmt_row_stride, x_trans, 
            pmt_y, ld_pmt, pmt_row_stride, y_trans
        );
        // Multiply the periodic block for root node
        // y1{root} = y1{root} + O * y0{root};  % y1{root} should be empty
        int root_idx     = h2pack->root_idx;
        int root_J_npt   = h2pack->J[root_idx]->length;
        int per_blk_size = root_J_npt * h2pack->krnl_dim;
        H2P_dense_mat_p y0_root = h2pack->y0[root_idx];
        H2P_dense_mat_p y1_root = h2pack->y1[root_idx];
        H2P_dense_mat_resize(y1_root, per_blk_size, curr_n_vec);
        CBLAS_GEMM(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, per_blk_size, curr_n_vec, per_blk_size, 
            1.0, h2pack->per_blk, per_blk_size, y0_root->data, curr_n_vec, 0.0, y1_root->data, curr_n_vec
        );
        et = get_wtime_sec();
        timers[MV_MID_TIMER_IDX] += et - st;

        // 5. Backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
        st = get_wtime_sec();
        H2P_matmul_bwd_transform(
            h2pack, curr_n_vec, 
            pmt_y, ld_pmt, pmt_row_stride, y_trans
        );
        et = get_wtime_sec();
        timers[MV_BWD_TIMER_IDX] += et - st;

        // 6. Dense multiplication, calculate D_{ij} * x_j
        st = get_wtime_sec();
        H2P_matmul_periodic_dense_mult(
            h2pack, curr_n_vec, 
            pmt_x, ld_pmt, pmt_row_stride, x_trans, 
            pmt_y, ld_pmt, pmt_row_stride, y_trans
        );
        et = get_wtime_sec();
        timers[MV_DEN_TIMER_IDX] += et - st;

        // 7. Backward permute the output matrix
        st = get_wtime_sec();
        H2P_permute_matrix_row_backward(h2pack, layout, curr_n_vec, pmt_y, ld_pmt, curr_mat_y, ldy);
        et = get_wtime_sec();
        timers[MV_VOP_TIMER_IDX] += et - st;
        mat_size[MV_VOP_SIZE_IDX] += 4 * krnl_mat_size * curr_n_vec;
    }  // End of i_vec loop

    h2pack->n_matvec += n_vec;
}
