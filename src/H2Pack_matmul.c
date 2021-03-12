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
#include "utils.h"

// Initialize auxiliary array y0 used in H2 matmul forward transformation
void H2P_matmul_init_y0(H2Pack_p h2pack, const int n_vec)
{
    if (h2pack->y0 != NULL) return;
    int n_node = h2pack->n_node;
    h2pack->y0 = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    ASSERT_PRINTF(
        h2pack->y0 != NULL, 
        "Failed to allocate %d H2P_dense_mat_p for H2 matmul buffer\n", n_node
    );
    H2P_dense_mat_p *y0 = h2pack->y0;
    H2P_dense_mat_p *U  = h2pack->U;
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
    H2Pack_p h2pack, const int n_vec, 
    const DTYPE *mat_x, const int ldx, const int x_row_stride, const CBLAS_TRANSPOSE x_trans
)
{
    int n_thread       = h2pack->n_thread;
    int max_child      = h2pack->max_child;
    int max_level      = h2pack->max_level;
    int min_adm_level  = (h2pack->is_HSS) ? h2pack->HSS_min_adm_level : h2pack->min_adm_level;
    int n_leaf_node    = h2pack->n_leaf_node;
    int *children      = h2pack->children;
    int *n_child       = h2pack->n_child;
    int *level_nodes   = h2pack->level_nodes;
    int *level_n_node  = h2pack->level_n_node;
    int *mat_cluster   = h2pack->mat_cluster;

    // 1. Initialize y0 on the first run
    H2P_matmul_init_y0(h2pack, n_vec);

    // 2. Upward sweep
    H2P_dense_mat_p *y0 = h2pack->y0;
    H2P_dense_mat_p *U  = h2pack->U;
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);

        #pragma omp parallel num_threads(n_thread_i)
        {
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                H2P_dense_mat_p U_node = U[node];

                H2P_dense_mat_resize(y0[node], U_node->ncol, n_vec);
                if (n_child_node == 0)
                {
                    // Leaf node, directly multiply x_j with U_j^T
                    int s_row = mat_cluster[2 * node];
                    int e_row = mat_cluster[2 * node + 1];
                    int nrow = e_row - s_row + 1;
                    const DTYPE *mat_x_blk = mat_x + s_row * x_row_stride;
                    CBLAS_GEMM(
                        CblasRowMajor, CblasTrans, x_trans, U_node->ncol, n_vec, nrow,
                        1.0, U_node->data, U_node->ld, mat_x_blk, ldx, 0.0, y0[node]->data, y0[node]->ld
                    );
                } else {
                    // Non-leaf node, multiple U{node}^T with each child node y0 directly
                    int *node_children = children + node * max_child;
                    int U_srow = 0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        H2P_dense_mat_p y0_k = y0[child_k];
                        DTYPE *U_node_k = U_node->data + U_srow * U_node->ld;
                        DTYPE beta = (k == 0) ? 0.0 : 1.0;
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasNoTrans, U_node->ncol, n_vec, y0_k->nrow,
                            1.0, U_node_k, U_node->ld, y0_k->data, y0_k->ld, beta, y0[node]->data, y0[node]->ld
                        );
                        U_srow += y0_k->nrow;
                    }  // End of k loop
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
        }  // End of "#pragma omp parallel"
    }  // End of i loop
}

// Initialize auxiliary array y1 used in H2 matmul intermediate multiplication
void H2P_matmul_init_y1(H2Pack_p h2pack, const int n_vec)
{
    int n_node = h2pack->n_node;
    int *node_n_r_adm = (h2pack->is_HSS == 1) ? h2pack->node_n_r_inadm : h2pack->node_n_r_adm;
    H2P_dense_mat_p *U = h2pack->U;
    if (h2pack->y1 == NULL)
    {
        h2pack->y1 = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
        ASSERT_PRINTF(
            h2pack->y1 != NULL,
            "Failed to allocate %d H2P_dense_mat_t for H2 matvec buffer\n", n_node
        );
        for (int i = 0; i < n_node; i++) 
            H2P_dense_mat_init(&h2pack->y1[i], 0, 0);
    }
    H2P_dense_mat_p *y1 = h2pack->y1;
    for (int i = 0; i < n_node; i++) 
    {
        // Use ld to mark if y1[i] is visited in this intermediate sweep
        y1[i]->ld = 0;
        if (node_n_r_adm[i]) H2P_dense_mat_resize(y1[i], U[i]->ncol, n_vec);
    }
}

// H2 matmul intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
void H2P_matmul_intmd_mult(
    H2Pack_p h2pack, const int n_vec, 
    const DTYPE *mat_x, const int ldx, const int x_row_stride, const CBLAS_TRANSPOSE x_trans,
          DTYPE *mat_y, const int ldy, const int y_row_stride, const CBLAS_TRANSPOSE y_trans
)
{
    int n_node        = h2pack->n_node;
    int n_thread      = h2pack->n_thread;
    int *node_level   = h2pack->node_level;
    int *mat_cluster  = h2pack->mat_cluster;
    int *B_p2i_rowptr = h2pack->B_p2i_rowptr;
    int *B_p2i_colidx = h2pack->B_p2i_colidx;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    H2P_dense_mat_p *y0 = h2pack->y0;

    // 1. Initialize y1 on the first run or reset the size of each y1
    H2P_matmul_init_y1(h2pack, n_vec);
    H2P_dense_mat_p *y1 = h2pack->y1;

    // 2. Intermediate sweep
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p Bij = thread_buf[tid]->mat0;

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int level0 = node_level[node0];
            
            H2P_dense_mat_p y1_0 = y1[node0];
            memset(y1_0->data, 0, sizeof(DTYPE) * y1_0->nrow * y1_0->ncol);

            for (int i = B_p2i_rowptr[node0]; i < B_p2i_rowptr[node0 + 1]; i++)
            {
                int node1  = B_p2i_colidx[i];
                int level1 = node_level[node1];

                int Bij_nrow, Bij_ncol, Bij_ld, Bij_trans;
                H2P_dense_mat_p y0_1 = y0[node1];
                H2P_get_Bij_block(h2pack, node0, node1, Bij);
                DTYPE *Bij_data = Bij->data;
                if (Bij->ld > 0)
                {
                    Bij_nrow  = Bij->nrow;
                    Bij_ncol  = Bij->ncol;
                    Bij_ld    = Bij->ld;
                    Bij_trans = 0;
                } else {
                    Bij_nrow  = Bij->ncol;
                    Bij_ncol  = Bij->nrow;
                    Bij_ld    = -Bij->ld;
                    Bij_trans = 1;
                }

                // We only handle the update on node0's side, the symmetric operation for
                // updating on node1's side is handled by double counting the inadmissible pairs

                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    CBLAS_TRANSPOSE Bij_trans_ = (Bij_trans == 0) ? CblasNoTrans : CblasTrans;
                    CBLAS_GEMM(
                        CblasRowMajor, Bij_trans_, CblasNoTrans, Bij_nrow, n_vec, Bij_ncol,
                        1.0, Bij_data, Bij_ld, y0_1->data, y0_1->ld, 1.0, y1_0->data, y1_0->ld
                    );
                }  // End of "if (level0 == level1)"

                // (2) node1 is a leaf node and its level is larger than node0,
                //     only compress on node0's side
                if (level0 > level1)
                {
                    int mat_x_srow = mat_cluster[node1 * 2];
                    const DTYPE *mat_x_spos = mat_x + mat_x_srow * x_row_stride;
                    CBLAS_TRANSPOSE Bij_trans_ = (Bij_trans == 0) ? CblasNoTrans : CblasTrans;
                    CBLAS_GEMM(
                        CblasRowMajor, Bij_trans_, x_trans, Bij_nrow, n_vec, Bij_ncol,
                        1.0, Bij_data, Bij_ld, mat_x_spos, ldx, 1.0, y1_0->data, y1_0->ld
                    );
                }  // End of "if (level0 > level1)"

                // (3) node0 is a leaf node and its level is larger than node1,
                //     only compress on node1's side
                if (level0 < level1)
                {
                    int mat_y_srow = mat_cluster[node0 * 2];
                    DTYPE *mat_y_spos = mat_y + mat_y_srow * y_row_stride;
                    if (y_trans == CblasNoTrans)
                    {
                        CBLAS_TRANSPOSE Bij_trans_ = (Bij_trans == 0) ? CblasNoTrans : CblasTrans;
                        CBLAS_GEMM(
                            CblasRowMajor, Bij_trans_, CblasNoTrans, Bij_nrow, n_vec, Bij_ncol,
                            1.0, Bij_data, Bij_ld, y0_1->data, y0_1->ld, 1.0, mat_y_spos, ldy
                        );
                    } else {
                        CBLAS_TRANSPOSE Bij_trans_ = (Bij_trans == 0) ? CblasTrans : CblasNoTrans;
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, Bij_trans_, n_vec, Bij_nrow, Bij_ncol,
                            1.0, y0_1->data, y0_1->ld, Bij_data, Bij_ld, 1.0, mat_y_spos, ldy
                        );
                    }
                }  // End of "if (level0 < level1)"
            }  // End of i loop
        }  // End of node0 loop
    }  // End of "#pragma omp parallel"
}

// H2 matmul backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
void H2P_matmul_bwd_transform(
    H2Pack_p h2pack, const int n_vec, 
    DTYPE *mat_y, const int ldy, const int y_row_stride, const CBLAS_TRANSPOSE y_trans
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
    H2P_dense_mat_p *U  = h2pack->U;
    H2P_dense_mat_p *y1 = h2pack->y1;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);
        
        #pragma omp parallel num_threads(n_thread_i) 
        {
            int tid = omp_get_thread_num();
            H2P_dense_mat_p y1_tmp = thread_buf[tid]->mat0;
            
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
                            copy_matrix_block(sizeof(DTYPE), child_k_len, n_vec, y1_tmp_spos, y1_tmp->ld, y1[child_k]->data, y1[child_k]->ld);
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
    H2Pack_p h2pack, const int n_vec, 
    const DTYPE *mat_x, const int ldx, const int x_row_stride, const CBLAS_TRANSPOSE x_trans,
          DTYPE *mat_y, const int ldy, const int y_row_stride, const CBLAS_TRANSPOSE y_trans
)
{
    int n_node        = h2pack->n_node;
    int n_thread      = h2pack->n_thread;
    int *mat_cluster  = h2pack->mat_cluster;
    int *D_p2i_rowptr = h2pack->D_p2i_rowptr;
    int *D_p2i_colidx = h2pack->D_p2i_colidx;
    H2P_thread_buf_p *thread_buf = h2pack->tb;

    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p Dij = thread_buf[tid]->mat0;

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int mat_y_srow = mat_cluster[2 * node0];
            DTYPE *mat_y_spos = mat_y + mat_y_srow * y_row_stride;

            for (int i = D_p2i_rowptr[node0]; i < D_p2i_rowptr[node0 + 1]; i++)
            {
                int node1 = D_p2i_colidx[i];
                int mat_x_srow = mat_cluster[2 * node1];
                const DTYPE *mat_x_spos = mat_x + mat_x_srow * x_row_stride;
                
                int Dij_nrow, Dij_ncol, Dij_ld, Dij_trans;
                H2P_get_Dij_block(h2pack, node0, node1, Dij);
                DTYPE *Dij_data = Dij->data;
                if (Dij->ld > 0)
                {
                    Dij_nrow  = Dij->nrow;
                    Dij_ncol  = Dij->ncol;
                    Dij_ld    = Dij->ld;
                    Dij_trans = 0;
                } else {
                    Dij_nrow  = Dij->ncol;
                    Dij_ncol  = Dij->nrow;
                    Dij_ld    = -Dij->ld;
                    Dij_trans = 1;
                }  // End of "if (Dij0->ld > 0)"

                // We only handle y_i = D_{ij} * x_j, its symmetric operation
                // y_j = D_{ij}' * x_i is handled by double counting inadmissible pairs
                if (x_trans == CblasNoTrans)
                {
                    CBLAS_TRANSPOSE Dij_trans_ = (Dij_trans == 0) ? CblasNoTrans : CblasTrans;
                    CBLAS_GEMM(
                        CblasRowMajor, Dij_trans_, CblasNoTrans, Dij_nrow, n_vec, Dij_ncol,
                        1.0, Dij_data, Dij_ld, mat_x_spos, ldx, 1.0, mat_y_spos, ldy
                    );
                } else {
                    CBLAS_TRANSPOSE Dij_trans_ = (Dij_trans == 0) ? CblasTrans : CblasNoTrans;
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, Dij_trans_, n_vec, Dij_nrow, Dij_ncol,
                        1.0, mat_x_spos, ldx, Dij_data, Dij_ld, 1.0, mat_y_spos, ldy
                    );
                }  // End of "if (x_trans == CblasNoTrans)"
            }  // End of i loop
        }  // End of node0 loop
    }  // End of "#pragma omp parallel"
}

// Permute the multiplicand matrix from the original point ordering to the 
// sorted point ordering inside H2Pack
void H2P_permute_matrix_row_forward(
    H2Pack_p h2pack, const CBLAS_LAYOUT layout, const int n_vec, 
    const DTYPE *mat_x, const int ldx, DTYPE *pmt_mat_x, const int ldp
)
{
    int krnl_mat_size = h2pack->krnl_mat_size;
    int *fwd_pmt_idx  = h2pack->fwd_pmt_idx;
    if (layout == CblasRowMajor)
    {
        gather_matrix_rows(sizeof(DTYPE), krnl_mat_size, n_vec, fwd_pmt_idx, mat_x, ldx, pmt_mat_x, ldp);
    } else {
        gather_matrix_cols(sizeof(DTYPE), n_vec, krnl_mat_size, fwd_pmt_idx, mat_x, ldx, pmt_mat_x, ldp);
    }
}

// Permute the output matrix from the sorted point ordering inside H2Pack 
// to the original point ordering
void H2P_permute_matrix_row_backward(
    H2Pack_p h2pack, const CBLAS_LAYOUT layout, const int n_vec, 
    const DTYPE *mat_x, const int ldx, DTYPE *pmt_mat_x, const int ldp
)
{
    int krnl_mat_size = h2pack->krnl_mat_size;
    int *bwd_pmt_idx  = h2pack->bwd_pmt_idx;
    if (layout == CblasRowMajor)
    {
        gather_matrix_rows(sizeof(DTYPE), krnl_mat_size, n_vec, bwd_pmt_idx, mat_x, ldx, pmt_mat_x, ldp);
    } else {
        gather_matrix_cols(sizeof(DTYPE), n_vec, krnl_mat_size, bwd_pmt_idx, mat_x, ldx, pmt_mat_x, ldp);
    }
}

// H2 representation multiplies a dense general matrix
void H2P_matmul(
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
        H2P_matmul_intmd_mult(
            h2pack, curr_n_vec, 
            pmt_x, ld_pmt, pmt_row_stride, x_trans, 
            pmt_y, ld_pmt, pmt_row_stride, y_trans
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
        H2P_matmul_dense_mult(
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
