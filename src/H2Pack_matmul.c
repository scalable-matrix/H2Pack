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
#include "H2Pack_utils.h"
#include "x86_intrin_wrapper.h"
#include "utils.h"

void H2P_matmul_init_y0(H2Pack_t h2pack, const int n_vec)
{
    if (h2pack->y0 != NULL) return;
    int n_node = h2pack->n_node;
    h2pack->y0 = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    ASSERT_PRINTF(
        h2pack->y0 != NULL, 
        "Failed to allocate %d H2P_dense_mat_t for H2 matmul buffer\n", n_node
    );
    H2P_dense_mat_t *y0 = h2pack->y0;
    H2P_dense_mat_t *U  = h2pack->U;
    for (int node = 0; node < n_node; node++)
    {
        int ncol = U[node]->ncol;
        if (ncol > 0) 
        {
            H2P_dense_mat_init(&y0[node], ncol, n_vec);
        } else {
            H2P_dense_mat_init(&y0[node], 0, 0);
            y0[node]->nrow = 0;
            y0[node]->ncol = 0;
            y0[node]->ld   = 0;
        }
    }
}

// H2 matmul forward transformation, calculate U_j^T * x_j
void H2P_matmul_fwd_transform(
    H2Pack_t h2pack, const int n_vec, 
    const DTYPE *mat_x, const int ldx, const int x_stride, const CBLAS_TRANSPOSE x_trans
)
{
    int n_thread       = h2pack->n_thread;
    int max_child      = h2pack->max_child;
    int max_level      = h2pack->max_level;
    int n_node         = h2pack->n_node;
    int n_leaf_node    = h2pack->n_leaf_node;
    int *children      = h2pack->children;
    int *n_child       = h2pack->n_child;
    int *level_nodes   = h2pack->level_nodes;
    int *level_n_node  = h2pack->level_n_node;
    int *mat_cluster   = h2pack->mat_cluster;
    H2P_thread_buf_t *thread_buf = h2pack->tb;

    int min_adm_level, max_adm_height;
    if (h2pack->is_HSS == 0)
    {
        min_adm_level  = h2pack->min_adm_level;
        max_adm_height = h2pack->max_adm_height;
    } else {
        min_adm_level  = h2pack->HSS_min_adm_level;
        max_adm_height = h2pack->HSS_max_adm_height;
    }

    // 1. Initialize y0 on the first run
    H2P_matmul_init_y0(h2pack, n_vec);

    // 2. Upward sweep
    H2P_dense_mat_t *y0 = h2pack->y0;
    H2P_dense_mat_t *U  = h2pack->U;
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);

        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            H2P_int_vec_t   idx    = thread_buf[tid]->idx0;
            H2P_dense_mat_t y0_tmp = thread_buf[tid]->mat0;
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                H2P_dense_mat_t U_node = U[node];

                H2P_dense_mat_resize(y0[node], U_node->ncol, n_vec);
                if (n_child_node == 0)
                {
                    // Leaf node, directly multiply x_j with U_j^T
                    int s_row = mat_cluster[2 * node];
                    int e_row = mat_cluster[2 * node + 1];
                    int nrow = e_row - s_row + 1;
                    const DTYPE *mat_x_blk = mat_x + s_row * x_stride;
                    CBLAS_GEMM(
                        CblasRowMajor, CblasTrans, x_trans, U_node->ncol, n_vec, nrow,
                        1.0, U_node->data, U_node->ld, mat_x_blk, ldx, 0.0, y0[node]->data, y0[node]->ld
                    );
                } else {
                    // Non-leaf node, concatenate y0 in the children nodes and multiply it with U_j^T
                    // Multiple U{node} with each child nodes' y0 directly
                    int *node_children = children + node * max_child;
                    int y0_tmp_nrow = 0;
                    DTYPE beta = 0.0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        H2P_dense_mat_t y0_k = y0[child_k];
                        DTYPE *U_node_k_row = U_node->data + y0_tmp_nrow * U_node->ld;
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasNoTrans, U_node->ncol, n_vec, y0_k->nrow,
                            1.0, U_node_k_row, U_node->ld, y0_k->data, y0_k->ld, beta, y0[node]->data, y0[node]->ld
                        );
                        beta = 1.0;
                        y0_tmp_nrow += y0[child_k]->nrow;
                    }  // End of k loop
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
        }  // End of "#pragma omp parallel"
    }  // End of i loop
}

void H2P_matmul_init_y1(H2Pack_t h2pack, const int n_vec)
{
    int n_node = h2pack->n_node;
    int *node_n_r_adm = (h2pack->is_HSS == 1) ? h2pack->node_n_r_inadm : h2pack->node_n_r_adm;
    H2P_dense_mat_t *U = h2pack->U;
    if (h2pack->y1 == NULL)
    {
        h2pack->y1 = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
        ASSERT_PRINTF(
            h2pack->y1 != NULL,
            "Failed to allocate %d H2P_dense_mat_t for H2 matvec buffer\n", n_node
        );
        for (int i = 0; i < n_node; i++) 
            H2P_dense_mat_init(&h2pack->y1[i], 0, 0);
    }
    H2P_dense_mat_t *y1 = h2pack->y1;
    for (int i = 0; i < n_node; i++) 
    {
        // Use ld to mark if y1[i] is visited in this intermediate sweep
        y1[i]->ld = 0;
        if (node_n_r_adm[i]) H2P_dense_mat_resize(y1[i], U[i]->ncol, n_vec);
    }
}

// H2 matmul intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
void H2P_matmul_intmd_mult(
    H2Pack_t h2pack, const int n_vec, 
    const DTYPE *mat_x, const int ldx, const int x_stride, const CBLAS_TRANSPOSE x_trans,
          DTYPE *mat_y, const int ldy, const int y_stride, const CBLAS_TRANSPOSE y_trans
)
{
    int n_node        = h2pack->n_node;
    int n_thread      = h2pack->n_thread;
    int *node_level   = h2pack->node_level;
    int *mat_cluster  = h2pack->mat_cluster;
    int *B_p2i_rowptr = h2pack->B_p2i_rowptr;
    int *B_p2i_colidx = h2pack->B_p2i_colidx;
    int *B_p2i_val    = h2pack->B_p2i_val;
    H2P_thread_buf_t *thread_buf = h2pack->tb;
    H2P_dense_mat_t *y0 = h2pack->y0;

    // 1. Initialize y1 on the first run or reset the size of each y1
    H2P_matmul_init_y1(h2pack, n_vec);
    H2P_dense_mat_t *y1 = h2pack->y1;

    // 2. Intermediate sweep
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_t Bij  = thread_buf[tid]->mat0;
        H2P_dense_mat_t Bij0 = thread_buf[tid]->mat1;

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int level0 = node_level[node0];
            
            H2P_dense_mat_t y1_0 = y1[node0];
            memset(y1_0->data, 0, sizeof(DTYPE) * y1_0->nrow * y1_0->ncol);

            for (int i = B_p2i_rowptr[node0]; i < B_p2i_rowptr[node0 + 1]; i++)
            {
                int node1  = B_p2i_colidx[i];
                int level1 = node_level[node1];

                H2P_dense_mat_t y0_1 = y0[node1];
                H2P_get_Bij_block(h2pack, node0, node1, Bij0);
                if (Bij0->ld > 0)
                {
                    H2P_dense_mat_resize(Bij, Bij0->nrow, Bij0->ncol);
                    H2P_copy_matrix_block(Bij0->nrow, Bij0->ncol, Bij0->data, Bij0->ld, Bij->data, Bij->ld);
                } else {
                    Bij0->ld = -Bij0->ld;
                    H2P_dense_mat_resize(Bij, Bij0->ncol, Bij0->nrow);
                    H2P_transpose_dmat(1, Bij0->nrow, Bij0->ncol, Bij0->data, Bij0->ld, Bij->data, Bij->ld);
                }

                // We only handle the update on node0's side, the symmetric operation for
                // updating on node1's side is handled by double counting the inadmissible pairs

                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, Bij->nrow, n_vec, Bij->ncol,
                        1.0, Bij->data, Bij->ld, y0_1->data, y0_1->ld, 1.0, y1_0->data, y1_0->ld
                    );
                }  // End of "if (level0 == level1)"

                // (2) node1 is a leaf node and its level is larger than node0,
                //     only compress on node0's side
                if (level0 > level1)
                {
                    int mat_x_srow = mat_cluster[node1 * 2];
                    const DTYPE *mat_x_spos = mat_x + mat_x_srow * x_stride;
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, x_trans, Bij->nrow, n_vec, Bij->ncol,
                        1.0, Bij->data, Bij->ld, mat_x_spos, ldx, 1.0, y1_0->data, y1_0->ld
                    );
                }  // End of "if (level0 > level1)"

                // (3) node0 is a leaf node and its level is larger than node1,
                //     only compress on node1's side
                if (level0 < level1)
                {
                    int mat_y_srow = mat_cluster[node0 * 2];
                    DTYPE *mat_y_spos = mat_y + mat_y_srow * y_stride;
                    if (y_trans == CblasNoTrans)
                    {    
                        CBLAS_GEMM(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans, Bij->nrow, n_vec, Bij->ncol,
                            1.0, Bij->data, Bij->ld, y0_1->data, y0_1->ld, 1.0, mat_y_spos, ldy
                        );
                    } else {
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasTrans, n_vec, Bij->nrow, Bij->ncol,
                            1.0, y0_1->data, y0_1->ld, Bij->data, Bij->ld, 1.0, mat_y_spos, ldy
                        );
                    }
                }  // End of "if (level0 < level1)"
            }  // End of i loop
        }  // End of node0 loop
    }  // End of "#pragma omp parallel"
}

// H2 matmul backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
void H2P_matmul_bwd_transform(
    H2Pack_t h2pack, const int n_vec, 
    DTYPE *mat_y, const int ldy, const int y_stride, const CBLAS_TRANSPOSE y_trans
)
{
    int n_thread        = h2pack->n_thread;
    int max_child       = h2pack->max_child;
    int n_leaf_node     = h2pack->n_leaf_node;
    int max_level       = h2pack->max_level;
    int min_adm_level   = (h2pack->is_HSS) ? h2pack->HSS_min_adm_level : h2pack->min_adm_level;
    int *children       = h2pack->children;
    int *n_child        = h2pack->n_child;
    int *level_n_node   = h2pack->level_n_node;
    int *level_nodes    = h2pack->level_nodes;
    int *mat_cluster    = h2pack->mat_cluster;
    H2P_dense_mat_t *U  = h2pack->U;
    H2P_dense_mat_t *y1 = h2pack->y1;
    H2P_thread_buf_t *thread_buf = h2pack->tb;
    
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);
        
        #pragma omp parallel num_threads(n_thread_i) 
        {
            int tid = omp_get_thread_num();
            H2P_dense_mat_t y1_tmp = thread_buf[tid]->mat0;
            
            thread_buf[tid]->timer = -get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                int *child_nodes = children + node * max_child;
                
                if (y1[node]->ld == 0) continue;
                
                H2P_dense_mat_resize(y1_tmp, U[node]->nrow, n_vec);

                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, U[node]->nrow, n_vec, U[node]->ncol,
                    1.0, U[node]->data, U[node]->ld, y1[node]->data, y1[node]->ld, 0.0, y1_tmp->data, y1_tmp->ld
                );
                
                if (n_child_node == 0)
                {
                    // Leaf node, accumulate final results to output vector
                    int s_row = mat_cluster[2 * node];
                    int e_row = mat_cluster[2 * node + 1];
                    int n_row = e_row - s_row + 1;
                    if (y_trans == CblasNoTrans)
                    {
                        for (int k = 0; k < n_row; k++)
                        {
                            DTYPE *mat_y_k  = mat_y + (s_row + k) * ldy;
                            DTYPE *y1_tmp_k = y1_tmp->data + k * y1_tmp->ld;
                            #pragma omp simd
                            for (int l = 0; l < n_vec; l++)
                                mat_y_k[l] += y1_tmp_k[l];
                        }
                    } else {
                        for (int l = 0; l < n_vec; l++)
                        {
                            DTYPE *mat_y_l  = mat_y + l * ldy;
                            DTYPE *y1_tmp_l = y1_tmp->data + l;
                            #pragma omp simd
                            for (int k = 0; k < n_row; k++)
                                mat_y_l[s_row + k] += y1_tmp_l[k * y1_tmp->ld];
                        }
                    }
                } else {
                    // Non-leaf node, push down y1 values
                    int y1_tmp_idx = 0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = child_nodes[k];
                        int child_k_len = U[child_k]->ncol;
                        DTYPE *y1_tmp_spos = y1_tmp->data + y1_tmp_idx * y1_tmp->ld;
                        if (y1[child_k]->ld == 0)
                        {
                            H2P_dense_mat_resize(y1[child_k], child_k_len, n_vec);
                            H2P_copy_matrix_block(child_k_len, n_vec, y1_tmp_spos, y1_tmp->ld, y1[child_k]->data, y1[child_k]->ld);
                        } else {
                            for (int k0 = 0; k0 < child_k_len; k0++)
                            {
                                DTYPE *y1_tmp_k0 = y1_tmp_spos + k0 * y1_tmp->ld;
                                DTYPE *y1_child_k_k0 = y1[child_k]->data + k0 * y1[child_k]->ld;
                                #pragma omp simd
                                for (int l = 0; l < n_vec; l++)
                                    y1_child_k_k0[l] += y1_tmp_k0[l];
                            }
                        }
                        y1_tmp_idx += child_k_len;
                    }  // End of k loop
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
            thread_buf[tid]->timer += get_wtime_sec();
        }  // End of "pragma omp parallel"
    }  // End of i loop
}

// H2 matmul dense multiplication, calculate D_{ij} * x_j
void H2P_matmul_dense_mult(
    H2Pack_t h2pack, const int n_vec, 
    const DTYPE *mat_x, const int ldx, const int x_stride, const CBLAS_TRANSPOSE x_trans,
          DTYPE *mat_y, const int ldy, const int y_stride, const CBLAS_TRANSPOSE y_trans
)
{
    int n_node        = h2pack->n_node;
    int n_thread      = h2pack->n_thread;
    int *mat_cluster  = h2pack->mat_cluster;
    int *D_p2i_rowptr = h2pack->D_p2i_rowptr;
    int *D_p2i_colidx = h2pack->D_p2i_colidx;
    int *D_p2i_val    = h2pack->D_p2i_val;
    H2P_thread_buf_t *thread_buf = h2pack->tb;

    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_t Dij  = thread_buf[tid]->mat0;
        H2P_dense_mat_t Dij0 = thread_buf[tid]->mat1;

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int mat_y_srow = mat_cluster[2 * node0];
            DTYPE *mat_y_spos = mat_y + mat_y_srow * y_stride;

            for (int i = D_p2i_rowptr[node0]; i < D_p2i_rowptr[node0 + 1]; i++)
            {
                int node1 = D_p2i_colidx[i];
                int mat_x_srow = mat_cluster[2 * node1];
                const DTYPE *mat_x_spos = mat_x + mat_x_srow * x_stride;
                
                H2P_get_Dij_block(h2pack, node0, node1, Dij0);
                if (Dij0->ld > 0)
                {
                    H2P_dense_mat_resize(Dij, Dij0->nrow, Dij0->ncol);
                    H2P_copy_matrix_block(Dij0->nrow, Dij0->ncol, Dij0->data, Dij0->ld, Dij->data, Dij->ld);
                } else {
                    Dij0->ld = -Dij0->ld;
                    H2P_dense_mat_resize(Dij, Dij0->ncol, Dij0->nrow);
                    H2P_transpose_dmat(1, Dij0->nrow, Dij0->ncol, Dij0->data, Dij0->ld, Dij->data, Dij->ld);
                }

                // We only handle y_i = D_{ij} * x_j, its symmetric operation
                // y_j = D_{ij}' * x_i is handled by double counting inadmissible pairs
                if (x_trans == CblasNoTrans && y_trans == CblasNoTrans)
                {
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, Dij->nrow, n_vec, Dij->ncol,
                        1.0, Dij->data, Dij->ld, mat_x_spos, ldx, 1.0, mat_y_spos, ldy
                    );
                } else {
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasTrans, n_vec, Dij->nrow, Dij->ncol,
                        1.0, mat_x_spos, ldx, Dij->data, Dij->ld, 1.0, mat_y_spos, ldy
                    );
                }
            }  // End of i loop
        }  // End of node0 loop
    }  // End of "#pragma omp parallel"
}

// H2 representation multiplies a dense general matrix
void H2P_matmul(
    H2Pack_t h2pack, const CBLAS_LAYOUT layout, const int n_vec, 
    const DTYPE *mat_x, const int ldx, DTYPE *mat_y, const int ldy
)
{
    double st, et;
    int krnl_mat_size = h2pack->krnl_mat_size;
    int krnl_dim      = h2pack->krnl_dim;
    int n_point       = h2pack->n_point;

    int x_stride, y_stride;
    CBLAS_TRANSPOSE x_trans, y_trans;
    if (layout == CblasRowMajor)
    {
        x_stride = ldx;
        y_stride = ldy;
        x_trans  = CblasNoTrans;
        y_trans  = CblasNoTrans;
    } else {
        x_stride = 1;
        y_stride = 1;
        x_trans  = CblasTrans;
        y_trans  = CblasTrans;
    }
    
    // 1. Reset output matrix
    st = get_wtime_sec();
    if (layout == CblasRowMajor)
    {
        size_t row_msize = sizeof(DTYPE) * n_vec;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < krnl_mat_size; i++)
        {
            DTYPE *mat_y_i = mat_y + i * ldy;
            memset(mat_y_i, 0, row_msize);
        }
    } else {
        #pragma omp parallel
        {
            for (int i = 0; i < n_vec; i++)
            {
                DTYPE *mat_y_i = mat_y + i * ldy;
                #pragma omp for schedule(static)
                for (int j = 0; j < krnl_mat_size; j++) mat_y_i[j] = 0.0;
            }
        }
    }
    et = get_wtime_sec();
    h2pack->timers[_MV_RDC_TIMER_IDX] += et - st;
    
    // 2. Forward transformation, calculate U_j^T * x_j
    st = get_wtime_sec();
    H2P_matmul_fwd_transform(h2pack, n_vec, mat_x, ldx, x_stride, x_trans);
    et = get_wtime_sec();
    h2pack->timers[_MV_FW_TIMER_IDX] += et - st;

    // 3. Intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
    st = get_wtime_sec();
    H2P_matmul_intmd_mult(h2pack, n_vec, mat_x, ldx, x_stride, x_trans, mat_y, ldy, y_stride, y_trans);
    et = get_wtime_sec();
    h2pack->timers[_MV_MID_TIMER_IDX] += et - st;

    // 4. Backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
    st = get_wtime_sec();
    H2P_matmul_bwd_transform(h2pack, n_vec, mat_y, ldy, y_stride, y_trans);
    et = get_wtime_sec();
    h2pack->timers[_MV_BW_TIMER_IDX] += et - st;

    // 5. Dense multiplication, calculate D_{ij} * x_j
    st = get_wtime_sec();
    H2P_matmul_dense_mult(h2pack, n_vec, mat_x, ldx, x_stride, x_trans, mat_y, ldy, y_stride, y_trans);
    et = get_wtime_sec();
    h2pack->timers[_MV_DEN_TIMER_IDX] += et - st;

    h2pack->n_matvec += n_vec;
}
