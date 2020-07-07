#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_matvec_periodic.h"
#include "H2Pack_utils.h"
#include "x86_intrin_wrapper.h"
#include "utils.h"

// H2 matvec intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
// Need to calculate all B_{ij} matrices before using it
void H2P_matvec_periodic_intmd_mult_JIT(H2Pack_t h2pack, const DTYPE *x, DTYPE *y)
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
    H2P_dense_mat_t  *y0 = h2pack->y0;
    H2P_dense_mat_t  *J_coord = h2pack->J_coord;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    kernel_mv_fptr   krnl_mv     = h2pack->krnl_mv;
    H2P_thread_buf_t *thread_buf = h2pack->tb;

    H2P_matvec_init_y1(h2pack);
    H2P_dense_mat_t *y1 = h2pack->y1;

    int root_idx = h2pack->root_idx;
    H2P_dense_mat_resize(y1[root_idx], 1, h2pack->U[root_idx]->ncol + 1);
    memset(y1[root_idx]->data, 0, sizeof(DTYPE) * (h2pack->U[root_idx]->ncol + 1));

    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_t Bij      = thread_buf[tid]->mat0;
        H2P_dense_mat_t coord1_s = thread_buf[tid]->mat1;
        DTYPE shift[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        thread_buf[tid]->timer = -get_wtime_sec();

        #pragma omp for schedule(static)
        for (int i = 0; i < n_node; i++)
        {
            if (y1[i]->ld == 0) continue;
            memset(y1[i]->data, 0, sizeof(DTYPE) * y1[i]->ncol);
        }

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int level0 = node_level[node0];
            
            H2P_dense_mat_t y1_0 = y1[node0];
            memset(y1_0->data, 0, sizeof(DTYPE) * y1_0->nrow * y1_0->ncol);

            for (int i = B_p2i_rowptr[node0]; i < B_p2i_rowptr[node0 + 1]; i++)
            {
                int node1    = B_p2i_colidx[i];
                int pair_idx = B_p2i_val[i] - 1;
                int level1   = node_level[node1];
                
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
                    if (krnl_mv != NULL)
                    {

                    } else {
                        krnl_eval(
                            J_coord[node0]->data, J_coord[node0]->ld, J_coord[node0]->ncol,
                            coord1_s->data,       coord1_s->ld,       coord1_s->ncol,
                            krnl_param, Bij->data, Bij->ld
                        );
                        CBLAS_GEMV(
                            CblasRowMajor, CblasNoTrans, Bij_nrow, Bij_ncol, 
                            1.0, Bij->data, Bij->ld, y0[node1]->data, 1, 1.0, y1[node0]->data, 1
                        );
                    }
                }  // End of "if (level0 == level1)"

                // (2) node1 is a leaf node and its level is higher than node0's level, 
                //     only compressed on node0's side, node1's side don't need the 
                //     downward sweep and can directly accumulate result to output vector
                if (level0 > level1)
                {
                    int pt_s1     = pt_cluster[node1 * 2];
                    int node1_npt = pt_cluster[node1 * 2 + 1] - pt_s1 + 1;
                    int vec_s1    = mat_cluster[node1 * 2];
                    
                    H2P_dense_mat_resize(coord1_s, xpt_dim, node1_npt);
                    H2P_copy_matrix_block(xpt_dim, node1_npt, coord + pt_s1, n_point, coord1_s->data, coord1_s->ld);
                    H2P_shift_coord(coord1_s, shift, 1.0);
                    if (krnl_mv != NULL)
                    {
                        const DTYPE *x_spos = x + pt_s1;
                    } else {
                        const DTYPE *x_spos = x + vec_s1;
                        krnl_eval(
                            J_coord[node0]->data, J_coord[node0]->ld, J_coord[node0]->ncol,
                            coord1_s->data,       coord1_s->ld,       coord1_s->ncol,
                            krnl_param, Bij->data, Bij->ld
                        );
                        CBLAS_GEMV(
                            CblasRowMajor, CblasNoTrans, Bij_nrow, Bij_ncol, 
                            1.0, Bij->data, Bij->ld, x_spos, 1, 1.0, y1[node0]->data, 1
                        );
                    }
                }  // End of "if (level0 > level1)"

                // (3) node0 is a leaf node and its level is higher than node1's level, 
                //     only compressed on node1's side, node0's side don't need the 
                //     downward sweep and can directly accumulate result to output vector
                if (level0 < level1)
                {
                    int pt_s0     = pt_cluster[node0 * 2];
                    int node0_npt = pt_cluster[node0 * 2 + 1] - pt_s0 + 1;
                    int vec_s0    = mat_cluster[node0 * 2];

                    H2P_dense_mat_copy(J_coord[node1], coord1_s);
                    H2P_shift_coord(coord1_s, shift, 1.0);
                    if (krnl_mv != NULL)
                    {
                        DTYPE *y_spos = y + pt_s0;
                    } else {
                        DTYPE *y_spos = y + vec_s0;
                        krnl_eval(
                            coord + pt_s0,  n_point,      node0_npt,
                            coord1_s->data, coord1_s->ld, coord1_s->ncol,
                            krnl_param, Bij->data, Bij->ld
                        );
                        CBLAS_GEMV(
                            CblasRowMajor, CblasNoTrans, Bij_nrow, Bij_ncol, 
                            1.0, Bij->data, Bij->ld, y0[node1]->data, 1, 1.0, y_spos, 1
                        );
                    }
                }  // End of "if (level0 < level1)"
            }  // End of node1 loop
        }  // End of node0 loop
        thread_buf[tid]->timer += get_wtime_sec();
    }  // End of "#pragma omp parallel"
    
    #ifdef PROFILING_OUTPUT
    double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
    for (int i = 0; i < n_thread; i++)
    {
        double thread_i_timer = thread_buf[i]->timer;
        avg_t += thread_i_timer;
        max_t = MAX(max_t, thread_i_timer);
        min_t = MIN(min_t, thread_i_timer);
    }
    avg_t /= (double) n_thread;
    printf("[PROFILING] MatVec intermediate sweep: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
}

// H2 matvec dense multiplication, calculate D_{ij} * x_j
// Need to calculate all D_{ij} matrices before using it
void H2P_matvec_periodic_dense_mult_JIT(H2Pack_t h2pack, const DTYPE *x, DTYPE *y)
{
    int   pt_dim            = h2pack->pt_dim;
    int   xpt_dim           = h2pack->xpt_dim;
    int   krnl_dim          = h2pack->krnl_dim;
    int   n_point           = h2pack->n_point;
    int   n_node            = h2pack->n_node;
    int   n_leaf_node       = h2pack->n_leaf_node;
    int   n_thread          = h2pack->n_thread;
    int   *leaf_nodes       = h2pack->level_nodes;
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
    H2P_dense_mat_t  *y0 = h2pack->y0;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    kernel_mv_fptr   krnl_mv     = h2pack->krnl_mv;
    H2P_thread_buf_t *thread_buf = h2pack->tb;

    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_t Dij      = thread_buf[tid]->mat0;
        H2P_dense_mat_t coord1_s = thread_buf[tid]->mat1;

        DTYPE shift[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        
        thread_buf[tid]->timer = -get_wtime_sec();

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int pt_s0     = pt_cluster[2 * node0];
            int node0_npt = pt_cluster[2 * node0 + 1] - pt_s0 + 1;
            int y_offset  = (krnl_mv == NULL) ? mat_cluster[node0 * 2] : pt_cluster[node0 * 2];
            DTYPE *y_spos = y + y_offset;

            for (int i = D_p2i_rowptr[node0]; i < D_p2i_rowptr[node0 + 1]; i++)
            {
                int node1     = D_p2i_colidx[i];
                int pair_idx  = D_p2i_val[i] - 1;
                int pt_s1     = pt_cluster[2 * node1];
                int node1_npt = pt_cluster[2 * node1 + 1] - pt_s1 + 1;
                int x_offset  = (krnl_mv == NULL) ? mat_cluster[node1 * 2] : pt_cluster[node1 * 2];
                const DTYPE *x_spos = x + x_offset;

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
                H2P_copy_matrix_block(xpt_dim, node1_npt, coord + pt_s1, n_point, coord1_s->data, coord1_s->ld);
                H2P_shift_coord(coord1_s, shift, 1.0);
                
                if (krnl_mv != NULL)
                {

                } else {
                    krnl_eval(
                        coord + pt_s0,  n_point,      node0_npt,
                        coord1_s->data, coord1_s->ld, coord1_s->ncol,
                        krnl_param, Dij->data, Dij->ld
                    );
                    CBLAS_GEMV(
                        CblasRowMajor, CblasNoTrans, Dij_nrow, Dij_ncol, 
                        1.0, Dij->data, Dij->ld, x_spos, 1, 1.0, y_spos, 1
                    );
                }

            }  // End of i loop
        }  // End of node0 loop
        thread_buf[tid]->timer += get_wtime_sec();
    }  // End of "pragma omp parallel"
    
    #ifdef PROFILING_OUTPUT
    double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
    for (int i = 0; i < n_thread; i++)
    {
        double thread_i_timer = thread_buf[i]->timer;
        avg_t += thread_i_timer;
        max_t = MAX(max_t, thread_i_timer);
        min_t = MIN(min_t, thread_i_timer);
    }
    avg_t /= (double) n_thread;
    printf("[PROFILING] MatVec dense block sweep: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
}

// H2 representation multiplies a column vector
void H2P_matvec_periodic(H2Pack_t h2pack, const DTYPE *x, DTYPE *y)
{
    double st, et;
    int   krnl_mat_size = h2pack->krnl_mat_size;
    int   n_thread      = h2pack->n_thread;
    int   BD_JIT        = h2pack->BD_JIT;
    int   krnl_dim      = h2pack->krnl_dim;
    int   n_point       = h2pack->n_point;
    int   need_trans    = ((h2pack->krnl_mv != NULL) && (BD_JIT == 1) && (krnl_dim > 1));
    DTYPE *xT           = h2pack->xT;
    DTYPE *yT           = h2pack->yT;
    H2P_thread_buf_t *thread_buf = h2pack->tb;

    const DTYPE *x_ = need_trans ? xT : x;
    DTYPE *y_ = need_trans ? yT : y;

    if (BD_JIT != 1)
    {
        ERROR_PRINTF("Only support BD_JIT=1 in this function for the moment.\n");
        return;
    }

    // 1. Reset y result to 0 and transpose x if necessary
    st = get_wtime_sec();
    #pragma omp parallel for simd
    for (int i = 0; i < krnl_mat_size; i++)
    {
        y[i]  = 0.0;
        yT[i] = 0.0;
    }
    if (need_trans) H2P_transpose_dmat(n_thread, n_point, krnl_dim, x, krnl_dim, xT, n_point);
    et = get_wtime_sec();
    h2pack->timers[_MV_RDC_TIMER_IDX] += et - st;

    // 2. Forward transformation, calculate U_j^T * x_j
    st = get_wtime_sec();
    H2P_matvec_fwd_transform(h2pack, x);
    et = get_wtime_sec();
    h2pack->timers[_MV_FW_TIMER_IDX] += et - st;
    
    // 3. Intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
    st = get_wtime_sec();
    if (BD_JIT == 1)
    {
        if (need_trans) H2P_transpose_y0_from_krnldim(h2pack);
        H2P_matvec_periodic_intmd_mult_JIT(h2pack, x_, y_);
        if (need_trans) H2P_transpose_y1_to_krnldim(h2pack);
    } else {
        // "Lost butterfly"
    }
    // Multiply the periodic block for root node
    // y1{root} = y1{root} + O * y0{root};
    int root_idx     = h2pack->root_idx;
    int per_blk_size = h2pack->J[root_idx]->length * krnl_dim;
    H2P_dense_mat_t y0_root = h2pack->y0[root_idx];
    H2P_dense_mat_t y1_root = h2pack->y1[root_idx];
    CBLAS_GEMV(
        CblasRowMajor, CblasNoTrans, per_blk_size, per_blk_size, 
        1.0, h2pack->per_blk, per_blk_size, y0_root->data, 1, 1.0, y1_root->data, 1
    );
    et = get_wtime_sec();
    h2pack->timers[_MV_MID_TIMER_IDX] += et - st;

    // 4. Backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
    st = get_wtime_sec();
    H2P_matvec_bwd_transform(h2pack, x, y);
    et = get_wtime_sec();
    h2pack->timers[_MV_BW_TIMER_IDX] += et - st;
    
    // 5. Dense multiplication, calculate D_i * x_i
    st = get_wtime_sec();
    if (BD_JIT == 1)
    {
        H2P_matvec_periodic_dense_mult_JIT(h2pack, x_, y_);
    } else {
        // "Lost butterfly"
    }
    et = get_wtime_sec();
    h2pack->timers[_MV_DEN_TIMER_IDX] += et - st;
    
    // 6. Sum yT partial results into y if needed 
    st = get_wtime_sec();
    // We use xT here to hold the transpose of yT
    if (need_trans)
    {
        H2P_transpose_dmat(n_thread, krnl_dim, n_point, yT, n_point, xT, krnl_dim);
        #pragma omp parallel for simd
        for (int i = 0; i < krnl_mat_size; i++) y[i] += xT[i];
        h2pack->mat_size[_MV_RDC_SIZE_IDX] += 4 * h2pack->krnl_mat_size;
    }
    et = get_wtime_sec();
    h2pack->timers[_MV_RDC_TIMER_IDX] += et - st;

    h2pack->n_matvec++;
}

