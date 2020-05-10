#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_utils.h"
#include "utils.h"

// Build explicit U matrices from nested U matrices
// Input parameter:
//   h2pack : H2Pack data structure after H2P_build()
// Output paramater:
//   exU : Size h2pack->n_node, explicit U matrices for each node
void H2P_build_explicit_U(H2Pack_t h2pack, H2P_dense_mat_t **exU_)
{
    int n_node        = h2pack->n_node;
    int n_leaf_node   = h2pack->n_leaf_node;
    int n_thread      = h2pack->n_thread;
    int max_level     = h2pack->max_level;
    int max_child     = h2pack->max_child;
    int *n_child      = h2pack->n_child;
    int *children     = h2pack->children;
    int *level_nodes  = h2pack->level_nodes;
    int *level_n_node = h2pack->level_n_node;
    H2P_dense_mat_t  *U = h2pack->U;
    H2P_thread_buf_t *thread_buf = h2pack->tb;

    H2P_dense_mat_t *exU = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    assert(exU != NULL);
    memset(exU, 0, sizeof(H2P_dense_mat_t) * n_node);

    for (int i = max_level; i >= 1; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);

        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            H2P_int_vec_t   idx  = thread_buf[tid]->idx0;
            H2P_dense_mat_t bd_U = thread_buf[tid]->mat0;
            H2P_int_vec_set_capacity(idx, max_child);
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                H2P_dense_mat_t U_node = U[node];

                if (U_node->nrow == 0) continue;
                if (n_child_node == 0)
                {
                    H2P_dense_mat_init(&exU[node], U_node->nrow, U_node->ncol);
                    H2P_copy_matrix_block(U_node->nrow, U_node->ncol, U_node->data, U_node->ld, exU[node]->data, exU[node]->ld);
                } else {
                    // TODO: don't use H2P_dense_mat_blkdiag(), apply children nodes' U directly
                    int *child_nodes = children + node * max_child;
                    memcpy(idx->data, child_nodes, sizeof(int) * n_child_node);
                    idx->length = n_child_node;
                    H2P_dense_mat_blkdiag(U, idx, bd_U);
                    H2P_dense_mat_init(&exU[node], bd_U->nrow, U_node->ncol);
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, bd_U->nrow, U_node->ncol, U_node->nrow,
                        1.0, bd_U->data, bd_U->ld, U_node->data, U_node->ld, 0.0, exU[node]->data, exU[node]->ld
                    );
                }
            }  // End of j loop
        }  // End of "#pragma omp parallel"
    }  // End of i loop

    for (int i = 0; i < n_node; i++)
    {
        if (exU[i] != NULL) continue;
        H2P_dense_mat_init(&exU[i], 8, 1);
        exU[i]->nrow = 0;
        exU[i]->ncol = 0;
        exU[i]->ld   = 0;
    }

    *exU_ = exU;
}

// Compute the level of two node's lowest common ancestor
// Input parameters:
//   parent     : Size n_node, parent of each node
//   node_level : Size n_node, level of each node
//   n_level    : Total number of levels (max_level + 1 if root is level 0)
//   node{0, 1} : Target node pair
//   work       : Work buffer, size >= 2 * n_level
int H2P_tree_common_ancestor_level(
    const int *parent, const int *node_level, const int n_level, 
    const int node0, const int node1, int *work
)
{
    int *path0 = work;
    int *path1 = work + n_level;
    memset(path0, 0, sizeof(int) * n_level);
    memset(path1, 0, sizeof(int) * n_level);
    int p0 = node0, p1 = node1;
    int level0 = node_level[node0], level1 = node_level[node1];
    for (int i = level0; i >= 0; i--)
    {
        path0[i] = p0;
        p0 = parent[p0];
    }
    for (int i = level1; i >= 0; i--)
    {
        path1[i] = p1;
        p1 = parent[p1];
    }
    int min_level_01 = level0 < level1 ? level0 : level1;
    int level = 0;
    for (int i = 0; i <= min_level_01; i++)
    {
        if (path0[i] != path1[i])
        {
            level = i - 1;
            break;
        }
    }
    return level;
}

void H2P_SPDHSS_H2_acc_matvec(H2Pack_t h2mat, const int max_rank, H2P_dense_mat_t **Yk_)
{
    int n_node         = h2mat->n_node;
    int n_leaf_node    = h2mat->n_leaf_node;
    int n_thread       = h2mat->n_thread;
    int n_r_inadm_pair = h2mat->n_r_inadm_pair;
    int n_r_adm_pair   = h2mat->n_r_adm_pair;
    int max_level      = h2mat->max_level;
    int max_child      = h2mat->max_child;
    int min_adm_level  = h2mat->min_adm_level;
    int *parent        = h2mat->parent;
    int *n_child       = h2mat->n_child;
    int *children      = h2mat->children;
    int *level_nodes   = h2mat->level_nodes;
    int *level_n_node  = h2mat->level_n_node;
    int *leaf_nodes    = h2mat->height_nodes;
    int *node_level    = h2mat->node_level;
    int *r_adm_pairs   = h2mat->r_adm_pairs;
    int *r_inadm_pairs = h2mat->r_inadm_pairs;
    int *mat_cluster   = h2mat->mat_cluster;
    H2P_dense_mat_t  *U = h2mat->U;
    H2P_thread_buf_t *thread_buf = h2mat->tb;

    // 1. Build explicit U matrix for each node
    // TODO: don't build each exU just in time and delete it after being used
    H2P_dense_mat_t *exU;
    H2P_build_explicit_U(h2mat, &exU);

    // 2. Prepare the Gaussian random matrix vec and Yk_mat
    // Yk_mat(:, n_vec*(i-1) + 1:n_vec) stores the matvec results for nodes at the i-th level
    const int kms = h2mat->krnl_mat_size;
    const int n_vec = max_rank;
    const int Yk_mat_ld = n_vec * max_level;
    size_t vec_msize = sizeof(DTYPE) * kms * n_vec;
    DTYPE *vec    = (DTYPE*) malloc(vec_msize);
    DTYPE *Yk_mat = (DTYPE*) malloc(vec_msize * max_level);  // Note: we have max_level+1 levels in total
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        int s_row, n_row;
        calc_block_spos_len(kms, n_thread, tid, &s_row, &n_row);
        H2P_gen_normal_distribution(0.0, 1.0, n_row * n_vec, vec + s_row * n_vec);
        memset(Yk_mat + s_row * n_vec * max_level, 0, sizeof(DTYPE) * n_row * n_vec * max_level);
    }

    // 3. H2 matvec upward sweep
    H2P_dense_mat_t *y0 = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    assert(y0 != NULL);
    for (int i = max_level; i >= min_adm_level; i++)
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

                H2P_dense_mat_init(&y0[node], U_node->ncol, n_vec);
                if (n_child_node == 0)
                {
                    // Leaf node, directly multiply x_j with U_j^T
                    int s_row = mat_cluster[2 * node];
                    int e_row = mat_cluster[2 * node + 1];
                    int nrow = e_row - s_row + 1;
                    DTYPE *vec_blk = vec + s_row * n_vec;
                    CBLAS_GEMM(
                        CblasRowMajor, CblasTrans, CblasNoTrans, U_node->ncol, n_vec, nrow,
                        1.0, U_node->data, U_node->ld, vec_blk, n_vec, 0.0, y0[node]->data, y0[node]->ld
                    );
                } else {
                    // Non-leaf node, concatenate y0 in the children nodes and multiply it with U_j^T
                    // TODO: don't use H2P_dense_mat_vertcat(), apply children nodes' y0 directly
                    int *child_nodes = children + node * max_child;
                    memcpy(idx->data, child_nodes, sizeof(int) * n_child_node);
                    idx->length = n_child_node;
                    H2P_dense_mat_vertcat(y0, idx, y0_tmp);
                    CBLAS_GEMM(
                        CblasRowMajor, CblasTrans, CblasNoTrans, U_node->ncol, n_vec, y0_tmp->nrow,
                        1.0, U_node->data, U_node->ld, y0_tmp->data, y0_tmp->ld, 0.0, y0[node]->data, y0[node]->ld
                    );
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
        }  // End of "#pragma omp parallel"
    }  // End of i loop

    // 4. For each pair of siblings (i, j), compute
    //    Yk{i} += Aij  * vec_j
    //    Yk{j} += Aij' * vec_i
    // Yk{i/j} are stored in the corresponding columns of the actual Yk
    // TODO: use B_blk / D_blk for task partition
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_int_vec_t   work = thread_buf[tid]->idx0;
        H2P_dense_mat_t Dij  = thread_buf[tid]->mat0;
        H2P_dense_mat_t Bij  = thread_buf[tid]->mat0;
        H2P_dense_mat_t tmpM = thread_buf[tid]->mat1;
        H2P_int_vec_set_capacity(work, 2 * max_level + 4);

        // 4.1 Inadmissible pairs
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < n_r_inadm_pair; i++)
        {
            int node0 = r_inadm_pairs[2 * i];
            int node1 = r_inadm_pairs[2 * i + 1];
            if (node0 == node1) continue;  // Do we really need this?
            int ca_level = H2P_tree_common_ancestor_level(parent, node_level, max_level+1, node0, node1, work->data) + 1;
            int s_col  = n_vec * (ca_level - 1);
            int n_col  = n_vec;
            int s_row0 = mat_cluster[2 * node0];
            int e_row0 = mat_cluster[2 * node0 + 1];
            int n_row0 = e_row0 - s_row0 + 1;
            int s_row1 = mat_cluster[2 * node1];
            int e_row1 = mat_cluster[2 * node1 + 1];
            int n_row1 = e_row1 - s_row1 + 1;
            H2P_get_Dij_block(h2mat, node0, node1, Dij);
            DTYPE *Yk_mat_blk0 = Yk_mat + s_row0 * Yk_mat_ld + s_col;
            DTYPE *Yk_mat_blk1 = Yk_mat + s_row1 * Yk_mat_ld + s_col;
            DTYPE *vec_blk0    = vec + s_row0 * n_vec;
            DTYPE *vec_blk1    = vec + s_row1 * n_vec;
            // Yk_mat(idx1, col_idx) = Yk_mat(idx1, col_idx) + D{D_idx}  * vec(idx2, :);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, n_row0, n_vec, n_row1,
                1.0, Dij->data, Dij->ld, vec_blk1, n_vec, 1.0, Yk_mat_blk0, Yk_mat_ld
            );
            // Yk_mat(idx2, col_idx) = Yk_mat(idx2, col_idx) + D{D_idx}' * vec(idx1, :);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, n_row1, n_vec, n_row0,
                1.0, Dij->data, Dij->ld, vec_blk0, n_vec, 1.0, Yk_mat_blk1, Yk_mat_ld
            );
        }  // End of i loop

        // 4.2 Admissible pairs
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < n_r_adm_pair; i++)
        {
            int node0 = r_adm_pairs[2 * i];
            int node1 = r_adm_pairs[2 * i + 1];
            if (node0 == node1) continue;  // Do we really need this?
            int ca_level = H2P_tree_common_ancestor_level(parent, node_level, max_level+1, node0, node1, work->data) + 1;
            int s_col  = n_vec * (ca_level - 1);
            int n_col  = n_vec;
            int s_row0 = mat_cluster[2 * node0];
            int e_row0 = mat_cluster[2 * node0 + 1];
            int n_row0 = e_row0 - s_row0 + 1;
            int s_row1 = mat_cluster[2 * node1];
            int e_row1 = mat_cluster[2 * node1 + 1];
            int n_row1 = e_row1 - s_row1 + 1;
            H2P_get_Bij_block(h2mat, node0, node1, Bij);
            DTYPE *Yk_mat_blk0 = Yk_mat + s_row0 * Yk_mat_ld + s_col;
            DTYPE *Yk_mat_blk1 = Yk_mat + s_row1 * Yk_mat_ld + s_col;
            DTYPE *vec_blk0    = vec + s_row0 * n_vec;
            DTYPE *vec_blk1    = vec + s_row1 * n_vec;

            int level0 = node_level[node0];
            int level1 = node_level[node1];
            H2P_dense_mat_t y0_0  = y0[node0];
            H2P_dense_mat_t y0_1  = y0[node1];
            H2P_dense_mat_t exU_0 = exU[node0];
            H2P_dense_mat_t exU_1 = exU[node1];

            // A. Two nodes are of the same level, compress on both side
            if (level0 == level1)
            {
                // Yk_mat(idx1, col_idx) = Yk_mat(idx1, col_idx) + exU{c1} * (Bij  * y0{c2});
                H2P_dense_mat_resize(tmpM, Bij->nrow, y0_1->ncol);
                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, Bij->nrow, y0_1->ncol, Bij->ncol,
                    1.0, Bij->data, Bij->ld, y0_1->data, y0_1->ld, 0.0, tmpM->data, tmpM->ld
                );
                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, exU_0->nrow, tmpM->ncol, exU_0->ncol,
                    1.0, exU_0->data, exU_0->ld, tmpM->data, tmpM->ld, 1.0, Yk_mat_blk0, Yk_mat_ld
                );
                // Yk_mat(idx2, col_idx) = Yk_mat(idx2, col_idx) + exU{c2} * (Bij' * y0{c1});
                H2P_dense_mat_resize(tmpM, Bij->ncol, y0_0->ncol);
                CBLAS_GEMM(
                    CblasRowMajor, CblasTrans, CblasNoTrans, Bij->ncol, y0_0->ncol, Bij->nrow, 
                    1.0, Bij->data, Bij->ld, y0_0->data, y0_0->ld, 0.0, tmpM->data, tmpM->ld
                );
                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, exU_1->nrow, tmpM->ncol, exU_1->ncol, 
                    1.0, exU_1->data, exU_1->ld, tmpM->data, tmpM->ld, 1.0, Yk_mat_blk1, Yk_mat_ld
                );
            }  // End of "if (level0 == level1)"

            // B. node1 is a leaf node and its level is larger than node0,
            //    only compress on node0's side
            if (level0 > level1)
            {
                // Yk_mat(idx1, col_idx) = Yk_mat(idx1, col_idx) + exU{c1} * (Bij * vec(idx2, :));
                H2P_dense_mat_resize(tmpM, Bij->nrow, n_vec);
                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, Bij->nrow, n_vec, Bij->ncol,
                    1.0, Bij->data, Bij->ld, vec_blk1, n_vec, 0.0, tmpM->data, tmpM->ld
                );
                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, exU_0->nrow, tmpM->ncol, exU_0->ncol,
                    1.0, exU_0->data, exU_0->ld, tmpM->data, tmpM->ld, 1.0, Yk_mat_blk0, Yk_mat_ld
                );
                // Yk_mat(idx2, col_idx) = Yk_mat(idx2, col_idx) + Bij' * y0{c1};
                CBLAS_GEMM(
                    CblasRowMajor, CblasTrans, CblasNoTrans, Bij->ncol, y0_0->ncol, Bij->nrow, 
                    1.0, Bij->data, Bij->ld, y0_0->data, y0_0->ld, 1.0, Yk_mat_blk1, Yk_mat_ld
                );
            }  // End of "if (level0 > level1)"

            // C. node0 is a leaf node and its level is larger than node1,
            //    only compress on node1's side
            if (level0 < level1)
            {
                // Yk_mat(idx1, col_idx) = Yk_mat(idx1, col_idx) + Bij * y0{c2};
                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, Bij->nrow, y0_1->ncol, Bij->ncol, 
                    1.0, Bij->data, Bij->ld, y0_1->data, y0_1->nrow, 1.0, Yk_mat_blk0, Yk_mat_ld
                );
                // Yk_mat(idx2, col_idx) = Yk_mat(idx2, col_idx) + exU{c2} * (Bij' * vec(idx1, :));
                H2P_dense_mat_resize(tmpM, Bij->ncol, n_vec);
                CBLAS_GEMM(
                    CblasRowMajor, CblasTrans, CblasNoTrans, Bij->ncol, n_vec, Bij->nrow, 
                    1.0, Bij->data, Bij->ld, vec_blk0, n_vec, 0.0, tmpM->data, tmpM->ld
                );
                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, exU_1->nrow, tmpM->ncol, exU_1->ncol,
                    1.0, exU_1->data, exU_1->ld, tmpM->data, tmpM->ld, 1.0, Yk_mat_blk1, Yk_mat_ld
                );
            }  // End of "if (level0 < level1)"
        }  // End of i loop 
    }  // End of "#pragma omp parallel"

    // 5. Accumulate the results in Yk_mat to lead nodes
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        int s_row, n_row;
        calc_block_spos_len(kms, n_thread, tid, &s_row, &n_row);
        for (int level = 1; level < max_level; level++)
        {
            int s_col = n_vec * (level - 1);
            for (int i = s_row; i < s_row + n_row; i++)
            {
                DTYPE *Yk_mat_i = Yk_mat + i * Yk_mat_ld;
                #pragma omp simd
                for (int j = s_col; j < s_col + n_vec; j++)
                    Yk_mat_i[j + n_vec] += Yk_mat_i[j];
            }
        }
    }  // End of "#pragma omp parallel"

    // 6. Repack Yk_mat into Yk
    H2P_dense_mat_t *Yk = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_leaf_node * max_level);
    assert(Yk != NULL);
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_leaf_node; i++)
        {
            int node  = leaf_nodes[i];
            int level = node_level[node];
            int s_row = mat_cluster[2 * node];
            int e_row = mat_cluster[2 * node + 1];
            int n_row = e_row - s_row + 1;
            for (int j = level - 1; j >= 0; j--)
            {
                int s_col = (j - 1) * n_vec;
                DTYPE *Yk_mat_blk = Yk_mat + s_row * Yk_mat_ld + s_col;
                H2P_dense_mat_init(&Yk[i * max_level + j], n_row, n_vec);
                H2P_dense_mat_t Yk_ij = Yk[i * max_level + j];
                H2P_copy_matrix_block(n_row, n_vec, Yk_mat_blk, Yk_mat_ld, Yk_ij->data, Yk_ij->ld);
            }
        }  // End of i loop
    }  // End of "#pragma omp parallel"

    // 7. Free intermediate arrays 
    for (int i = 0; i < n_node; i++)
    {
        H2P_dense_mat_destroy(exU[i]);
        H2P_dense_mat_destroy(y0[i]);
    }
    free(exU);
    free(y0);
    free(vec);
    free(Yk_mat);
    *Yk_ = Yk;
}

void H2P_SPDHSS_H2_build(
    const int max_rank, const DTYPE shift, 
    H2Pack_t h2mat, H2Pack_t *spdhss_
)
{
    if (h2mat == NULL || h2mat->U == NULL || h2mat->is_HSS)
    {
        printf("[FATAL] %s: h2mat not constructed or configured as HSS!\n", __FUNCTION__);
        return;
    }
}
