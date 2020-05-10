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

// Accumulate partial H2 matvec results for H2P_SPDHSS_H2_build()
// Input parameters:
//   h2mat    : Source H2 matrix structure
//   max_rank : Maximum HSS approximation rank, will use max_rank Gaussian random vectors
// Output parameter:
//   *Yk_ : Matrix, size h2mat->n_node * h2mat->max_level, each non-empty element is 
//          a matrix of max_rank columns
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

// TODO: HSS_B_pair2idx should be replaced by a CSR matrix

// Gather matrices in HSS_B into a large matrix tmpB s.t. the i-th row j-th column 
// block of tmpB is HSS_B[HSS_B_pair2idx(blk0[i], blk1[j])]
// Input parameters:
//   n_blk{0, 1}    : Number of row & column blocks
//   blk{0, 1}      : Node indices of row & column blocks
//   HSS_B          : Source HSS_B matrices
//   n_node         : Number of nodes
//   HSS_B_pair2idx : Size n_node * n_node, convert (blk0[i], blk1[j]) to an index
// Output parameter:
//   tmpB : Result matrix
void H2P_SPDHSS_H2_gather_HSS_B(
    const int n_blk0, const int n_blk1, const int *blk0, const int *blk1, 
    H2P_dense_mat_t *HSS_B, const int n_node, const int *HSS_B_pair2idx, 
    H2P_dense_mat_t tmpB
)
{
    int nrow = 0, ncol = 0;
    int B_idx_00 = HSS_B_pair2idx[blk0[0] * n_node + blk1[0]];
    int nrow0 = HSS_B[B_idx_00]->nrow;
    int ncol0 = HSS_B[B_idx_00]->ncol;
    
    // Count the total number of rows
    for (int i = 0; i < n_blk0; i++)
    {
        int B_idx_i0 = HSS_B_pair2idx[blk0[i] * n_node + blk1[0]];
        if (B_idx_i0 == 0)
        {
            printf("[FATAL] %s error: pair (%d, %d) B_idx_i0 == 0\n", __FUNCTION__, blk0[i], blk1[0]);
            return;
        }
        if (HSS_B[B_idx_i0]->ncol != ncol0)
        {
            printf(
                "[FATAL] %s error: pair (%d, %d) ncol = %d, expected %d\n", 
                __FUNCTION__, blk0[i], blk1[0], HSS_B[B_idx_i0]->ncol, ncol0
            );
            return;
        }
        nrow += HSS_B[B_idx_i0]->nrow;
    }  // End of i loop

    // Count the total number of columns
    for (int j = 0; j < n_blk1; j++)
    {
        int B_idx_0j = HSS_B_pair2idx[blk0[0] * n_node + blk1[j]];
        if (B_idx_0j == 0)
        {
            printf("[FATAL] %s error: pair (%d, %d) B_idx_i0 == 0\n", __FUNCTION__, blk0[0], blk1[j]);
            return;
        }
        if (HSS_B[B_idx_0j]->nrow != nrow0)
        {
            printf(
                "[FATAL] %s error: pair (%d, %d) nrow = %d, expected %d\n", 
                __FUNCTION__, blk0[0], blk1[j], HSS_B[B_idx_0j]->nrow, nrow0
            );
            return;
        }
        ncol += HSS_B[B_idx_0j]->ncol;
    }  // End of j loop

    // Gather each block
    H2P_dense_mat_resize(tmpB, nrow, ncol);
    int s_row = 0;
    for (int i = 0; i < n_blk0; i++)
    {
        int B_idx_i0 = HSS_B_pair2idx[blk0[i] * n_node + blk1[0]];
        int nrow_i = HSS_B[B_idx_i0]->nrow;
        s_row += HSS_B[B_idx_i0]->nrow;
        
        int s_col = 0;
        for (int j = 0; j < n_blk1; j++)
        {
            int B_idx_ij = HSS_B_pair2idx[blk0[i] * n_node + blk1[j]];
            if (B_idx_ij == 0)
            {
                printf("[FATAL] %s error: pair (%d, %d) B_idx_ij == 0\n", __FUNCTION__, blk0[i], blk1[j]);
                return;
            }
            DTYPE *tmpB_ij = tmpB->data + s_row * tmpB->ld + s_col;
            H2P_dense_mat_t HSS_Bij = HSS_B[B_idx_ij];
            int ncol_j = HSS_Bij->ncol;
            H2P_copy_matrix_block(nrow_i, ncol_j, HSS_Bij->data, HSS_Bij->ld, tmpB_ij, tmpB->ld);
            s_col += ncol_j;
        }  // End of j loop
    }  // End of i loop
}

// Remove unused HSS_B matrices
// Input parameters:
//   n_blk{0, 1}    : Number of row & column blocks
//   blk{0, 1}      : Node indices of row & column blocks
//   HSS_B          : Source HSS_B matrices
//   n_node         : Number of nodes
//   HSS_B_pair2idx : Size n_node * n_node, convert (i, j) pair to an index for HSS_B
// Output parameter:
//   HSS_B          : Updated HSS_B matrices
//   HSS_B_pair2idx : Updated HSS_B_pair2idx
void H2P_SPDHSS_H2_clean_HSS_B(
    const int n_blk0, const int n_blk1, const int *blk0, const int *blk1, 
    H2P_dense_mat_t *HSS_B, const int n_node, int *HSS_B_pair2idx
)
{
    for (int i = 0; i < n_blk0; i++)
    {
        for (int j = 0; j < n_blk1; j++)
        {
            int pair_ij = blk0[i] * n_node + blk1[j];
            int B_idx_ij = HSS_B_pair2idx[pair_ij];
            HSS_B_pair2idx[pair_ij] = 0;
            H2P_dense_mat_destroy(HSS_B[B_idx_ij]);
        }
    }
}

// Calculate a new HSS Bij matrix for pair (node0, node1)
// Input parameters:
//   h2mat          : Source H2 matrix structure
//   node{0, 1}     : Node pair
//   S, V, W, Minv  : Arrays, size h2mat->n_node, intermediate matrices used in H2P_SPDHSS_H2_build()
//   HSS_B          : New HSS Bij matrices
//   HSS_B_pair2idx : h2mat->n_node * h2mat->n_node, convert (i, j) pair to an index for HSS_B
// Output parameters:
//   HSS_B          : Updated HSS_B matrices (some unused Bij matrices will be deleted)
//   HSS_B_pair2idx : Updated HSS_B_pair2idx
void H2P_SPDHSS_H2_calc_HSS_Bij(
    H2Pack_t h2mat, const int node0, const int node1,
    H2P_dense_mat_t *S, H2P_dense_mat_t *V, H2P_dense_mat_t *W, 
    H2P_dense_mat_t *Minv, H2P_dense_mat_t *HSS_B, int *HSS_B_pair2idx
)
{
    int   pt_dim      = h2mat->pt_dim;
    int   max_child   = h2mat->max_child;
    int   n_node      = h2mat->n_node;
    int   *node_level = h2mat->node_level;
    int   *n_child    = h2mat->n_child;
    int   *children   = h2mat->children;
    DTYPE *enbox      = h2mat->enbox;

    int   level0   = node_level[node0];
    int   level1   = node_level[node1];
    int   n_child0 = n_child[node0];
    int   n_child1 = n_child[node1];
    int   *child0  = children + node0 * max_child;
    int   *child1  = children + node1 * max_child;
    DTYPE *enbox0  = enbox + node0 * 2 * pt_dim;
    DTYPE *enbox1  = enbox + node1 * 2 * pt_dim;
    int   is_adm   = H2P_check_box_admissible(enbox0, enbox1, pt_dim, ALPHA_H2);

    int B_idx = HSS_B_pair2idx[node0 * n_node + node1];
    H2P_dense_mat_t HSS_Bij = HSS_B[B_idx];

    if (level0 == level1)
    {
        // 1.1: node0 and node1 are admissible
        if (is_adm)
        {
            H2P_dense_mat_t H2_Bij, tmpM;
            H2P_dense_mat_init(&H2_Bij, 128, 128);
            H2P_dense_mat_init(&tmpM,   128, 128);
            H2_Bij->nrow = 0;
            H2P_get_Bij_block(h2mat, node0, node1, H2_Bij);
            if (H2_Bij->nrow == 0)
            {
                printf("[FATAL] %s bug in case 1.1\n", __FUNCTION__);
                H2P_dense_mat_destroy(H2_Bij);
                H2P_dense_mat_destroy(tmpM);
                return;
            }
            H2P_dense_mat_t W0 = W[node0];
            H2P_dense_mat_t W1 = W[node1];
            // Bij = W{node1} * H2_B{H2_B_idx} * W{node2}';
            H2P_dense_mat_resize(tmpM, H2_Bij->nrow, W1->nrow);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasTrans, H2_Bij->nrow, W1->nrow, H2_Bij->ncol,
                1.0, H2_Bij->data, H2_Bij->ld, W1->data, W1->ld, 0.0, tmpM->data, tmpM->ld
            );
            H2P_dense_mat_resize(HSS_Bij, W0->nrow, tmpM->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, W0->nrow, tmpM->ncol, W0->ncol,
                1.0, W0->data, W0->ld, tmpM->data, tmpM->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_dense_mat_destroy(H2_Bij);
            H2P_dense_mat_destroy(tmpM);
            return;
        }  // End of "if (is_adm)"

        // Otherwise: node0 and node1 are inadmissible

        // 1.2: Both nodes are leaf nodes
        if (n_child0 == 0 && n_child1 == 0)
        {
            H2P_dense_mat_t H2_Dij, tmpM;
            H2P_dense_mat_init(&H2_Dij, 128, 128);
            H2P_dense_mat_init(&tmpM,   128, 128);
            H2_Dij->nrow = 0;
            H2P_get_Dij_block(h2mat, node0, node1, H2_Dij);
            if (H2_Dij->nrow == 0)
            {
                printf("[FATAL] %s bug in case 1.2\n", __FUNCTION__);
                H2P_dense_mat_destroy(H2_Dij);
                H2P_dense_mat_destroy(tmpM);
                return;
            }
            H2P_dense_mat_t S0 = S[node0];
            H2P_dense_mat_t S1 = S[node1];
            H2P_dense_mat_t V0 = V[node0];
            H2P_dense_mat_t V1 = V[node1];
            // tmpM = V{node1}' * linsolve(S{node1}, H2_D{H2_D_idx}, struct('LT', true));
            CBLAS_TRSM(
                CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 
                H2_Dij->nrow, H2_Dij->ncol, 1.0, S0->data, S0->ld, H2_Dij->data, H2_Dij->ld
            );
            H2P_dense_mat_resize(tmpM, V0->ncol, H2_Dij->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, H2_Dij->ncol, V0->nrow,
                1.0, V0->data, V0->ld, H2_Dij->data, H2_Dij->ld, 0.0, tmpM->data, tmpM->ld
            );
            // Bij = linsolve(S{node2}, tmpM', struct('LT', true))' * V{node2};
            // S{node2} * X = tmpM', we need Bij = X' * V{node2}
            // Solve X' * S{node2}' = tmpM to obtain X' directly
            CBLAS_TRSM(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 
                tmpM->nrow, tmpM->ncol, 1.0, S1->data, S1->ld, tmpM->data, tmpM->ld
            );
            H2P_dense_mat_resize(HSS_Bij, tmpM->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpM->nrow, V1->ncol, tmpM->ncol,
                1.0, tmpM->data, tmpM->ld, V1->data, V1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_dense_mat_destroy(H2_Dij);
            H2P_dense_mat_destroy(tmpM);
            return;
        }  // End of "if (n_child0 == 0 && n_child1 == 0)"

        // 1.3: Both nodes are non-leaf nodes
        if (n_child0 > 0 && n_child1 > 0)
        {
            H2P_dense_mat_t tmpB, tmpM0, tmpM1;
            H2P_dense_mat_init(&tmpB,  128, 128);
            H2P_dense_mat_init(&tmpM0, 128, 128);
            H2P_dense_mat_init(&tmpM1, 128, 128);
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                n_child0, n_child1, child0, child1, 
                HSS_B, n_node, HSS_B_pair2idx, tmpB
            );
            if (tmpB->nrow == 0)
            {
                printf("[FATAL] %s bug in case 1.3\n", __FUNCTION__);
                H2P_dense_mat_destroy(tmpB);
                H2P_dense_mat_destroy(tmpM0);
                H2P_dense_mat_destroy(tmpM1);
                return;
            }
            H2P_dense_mat_t V0    = V[node0];
            H2P_dense_mat_t V1    = V[node1];
            H2P_dense_mat_t Minv0 = Minv[node0];
            H2P_dense_mat_t Minv1 = Minv[node1];
            // Bij = V{node1}' * Minv{node1} * tmpB * Minv{node2} * V{node2};
            H2P_dense_mat_resize(tmpM0, Minv1->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv1->nrow, V1->ncol, Minv1->ncol, 
                1.0, Minv1->data, Minv1->ld, V1->data, V1->ld, 0.0, tmpM0->data, tmpM0->ld
            );
            H2P_dense_mat_resize(tmpM1, tmpB->nrow, tmpM0->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpB->nrow, tmpM0->ncol, tmpB->ncol,
                1.0, tmpB->data, tmpB->ld, tmpM0->data, tmpM0->ld, 0.0, tmpM1->data, tmpM1->ld
            );
            H2P_dense_mat_resize(tmpM0, Minv0->nrow, tmpM1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv0->nrow, tmpM1->ncol, Minv0->ncol,
                1.0, Minv0->data, Minv0->ld, tmpM1->data, tmpM1->ld, 0.0, tmpM0->data, tmpM0->ld
            );
            H2P_dense_mat_resize(HSS_Bij, V0->ncol, tmpM0->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, tmpM0->ncol, V0->nrow, 
                1.0, V0->data, V0->ld, tmpM0->data, tmpM0->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(n_child0, n_child1, child0, child1, HSS_B, n_node, HSS_B_pair2idx);
            H2P_dense_mat_destroy(tmpB);
            H2P_dense_mat_destroy(tmpM0);
            H2P_dense_mat_destroy(tmpM1);
            return;
        }  // End of "if (n_child0 > 0 && n_child1 > 0)"

        // 1.4: node0 is non-leaf, node1 is leaf
        if (n_child0 > 0 && n_child1 == 0)
        {
            H2P_dense_mat_t tmpB, tmpM0, tmpM1;
            H2P_dense_mat_init(&tmpB,  128, 128);
            H2P_dense_mat_init(&tmpM0, 128, 128);
            H2P_dense_mat_init(&tmpM1, 128, 128);
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                n_child0, 1, child0, &node1, 
                HSS_B, n_node, HSS_B_pair2idx, tmpB
            );
            if (tmpB->nrow == 0)
            {
                printf("[FATAL] %s bug in case 1.4\n", __FUNCTION__);
                H2P_dense_mat_destroy(tmpB);
                H2P_dense_mat_destroy(tmpM0);
                H2P_dense_mat_destroy(tmpM1);
                return;
            }
            H2P_dense_mat_t V0    = V[node0];
            H2P_dense_mat_t V1    = V[node1];
            H2P_dense_mat_t S1    = S[node1];
            H2P_dense_mat_t Minv0 = Minv[node0];
            // tmpM1 = V{node1}' * Minv{node1} * tmpB;
            H2P_dense_mat_resize(tmpM0, Minv0->nrow, tmpB->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv0->nrow, tmpB->ncol, Minv0->ncol,
                1.0, Minv0->data, Minv0->ld, tmpB->data, tmpB->ld, 0.0, tmpM0->data, tmpM0->ld
            );
            H2P_dense_mat_resize(tmpM1, V0->ncol, tmpM0->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, tmpM0->ncol, V0->nrow,
                1.0, V0->data, V0->ld, tmpM0->data, tmpM0->ld, 0.0, tmpM1->data, tmpM1->ld
            );
            // Bij = linsolve(S{node2}, tmpM1', struct('LT', true))' * V{node2};
            // S{node2} * X = tmpM1', we need Bij = X' * V{node2}
            // Solve X' * S{node2}' = tmpM1 to obtain X' directly
            CBLAS_TRSM(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 
                tmpM1->nrow, tmpM1->ncol, 1.0, S1->data, S1->ld, tmpM1->data, tmpM1->ld
            );
            H2P_dense_mat_resize(HSS_Bij, tmpM1->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpM1->nrow, V1->ncol, tmpM1->ncol,
                1.0, tmpM1->data, tmpM1->ld, V1->data, V1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(n_child0, 1, child0, &node1, HSS_B, n_node, HSS_B_pair2idx);
            H2P_dense_mat_destroy(tmpB);
            H2P_dense_mat_destroy(tmpM0);
            H2P_dense_mat_destroy(tmpM1);
            return;
        }  // End of "if (n_child0 > 0 && n_child1 == 0)"

        // 1.5: node0 is leaf, node1 is non-leaf
        if (n_child0 == 0 && n_child1 > 0)
        {
            H2P_dense_mat_t tmpB, tmpM0, tmpM1;
            H2P_dense_mat_init(&tmpB,  128, 128);
            H2P_dense_mat_init(&tmpM0, 128, 128);
            H2P_dense_mat_init(&tmpM1, 128, 128);
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                1, n_child1, &node0, child1, 
                HSS_B, n_node, HSS_B_pair2idx, tmpB
            );
            if (tmpB->nrow == 0)
            {
                printf("[FATAL] %s bug in case 1.5\n", __FUNCTION__);
                H2P_dense_mat_destroy(tmpB);
                H2P_dense_mat_destroy(tmpM0);
                H2P_dense_mat_destroy(tmpM1);
                return;
            }
            H2P_dense_mat_t V0    = V[node0];
            H2P_dense_mat_t V1    = V[node1];
            H2P_dense_mat_t S0    = S[node0];
            H2P_dense_mat_t Minv1 = Minv[node1];
            // tmpM1 = tmpB * Minv{node2} * V{node2};
            H2P_dense_mat_resize(tmpM0, Minv1->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv1->nrow, V1->ncol, Minv1->ncol,
                1.0, Minv1->data, Minv1->ld, V1->data, V1->ld, 0.0, tmpM0->data, tmpM0->ld
            );
            H2P_dense_mat_resize(tmpM1, tmpB->nrow, tmpM0->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpB->nrow, tmpM0->ncol, tmpB->ncol,
                1.0, tmpB->data, tmpB->ld, tmpM0->data, tmpM0->ld, 0.0, tmpM1->data, tmpM1->ld
            );
            // Bij = V{node1}' * linsolve(S{node1}, tmpM1, struct('LT', true));
            CBLAS_TRSM(
                CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 
                tmpM1->nrow, tmpM1->ncol, 1.0, S0->data, S0->ld, tmpM1->data, tmpM1->ld
            );
            H2P_dense_mat_resize(HSS_Bij, V0->ncol, tmpM1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, tmpM1->ncol, V0->nrow, 
                1.0, V0->data, V0->ld, tmpM1->data, tmpM1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(1, n_child1, &node0, child1, HSS_B, n_node, HSS_B_pair2idx);
            H2P_dense_mat_destroy(tmpB);
            H2P_dense_mat_destroy(tmpM0);
            H2P_dense_mat_destroy(tmpM1);
            return;
        }  // End of "if (n_child0 == 0 && n_child1 > 0)"
    }  // End of "if (level0 == level1)"

    if (level0 > level1)
    {
        // Note: node1 must be a leaf node
        if (n_child1 > 0)
        {
            printf("[FATAL] %s bug in case 2\n", __FUNCTION__);
            return;
        }  // End of "if (n_child1 > 0)"

        // 2.1: node0 and node1 are admissible
        if (is_adm)
        {
            H2P_dense_mat_t H2_Bij;
            H2P_dense_mat_init(&H2_Bij, 128, 128);
            H2_Bij->nrow = 0;
            H2P_get_Bij_block(h2mat, node0, node1, H2_Bij);
            if (H2_Bij->nrow == 0)
            {
                printf("[FATAL] %s bug in case 2.1\n", __FUNCTION__);
                H2P_dense_mat_destroy(H2_Bij);
                return;
            }
            H2P_dense_mat_t W0 = W[node0];
            // Bij = W{node1} * H2_B{H2_B_idx};
            H2P_dense_mat_resize(HSS_Bij, W0->nrow, H2_Bij->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, W0->nrow, H2_Bij->ncol, W0->ncol,
                1.0, W0->data, W0->ld, H2_Bij->data, H2_Bij->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_dense_mat_destroy(H2_Bij);
            return;
        }  // End of "if (is_adm)"

        // Otherwise: node0 and node1 are inadmissible

        // 2.2: node0 is a leaf node
        if (n_child0 == 0)
        {
            H2P_dense_mat_t H2_Dij;
            H2P_dense_mat_init(&H2_Dij, 128, 128);
            H2_Dij->nrow = 0;
            H2P_get_Dij_block(h2mat, node0, node1, H2_Dij);
            if (H2_Dij->nrow == 0)
            {
                printf("[FATAL] %s bug in case 2.2\n", __FUNCTION__);
                H2P_dense_mat_destroy(H2_Dij);
                return;
            }
            H2P_dense_mat_t V0 = V[node0];
            H2P_dense_mat_t S0 = S[node0];
            // Bij = V{node1}' * linsolve(S{node1}, H2_Dij, struct('LT', true));
            CBLAS_TRSM(
                CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                H2_Dij->nrow, H2_Dij->ncol, 1.0, S0->data, S0->ld, H2_Dij->data, H2_Dij->ld
            );
            H2P_dense_mat_resize(HSS_Bij, V0->ncol, H2_Dij->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, H2_Dij->ncol, V0->nrow,
                1.0, V0->data, V0->ld, H2_Dij->data, H2_Dij->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_dense_mat_destroy(H2_Dij);
            return;
        }  // End of "if (n_child0 == 0)"

        // 2.3: node0 is a non-leaf node
        if (n_child0 > 0)
        {
            H2P_dense_mat_t tmpB, tmpM;
            H2P_dense_mat_init(&tmpB, 128, 128);
            H2P_dense_mat_init(&tmpM, 128, 128);
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                n_child0, 1, child0, &node1, 
                HSS_B, n_node, HSS_B_pair2idx, tmpB
            );
            if (tmpB->nrow == 0)
            {
                printf("[FATAL] %s bug in case 2.3\n", __FUNCTION__);
                H2P_dense_mat_destroy(tmpB);
                H2P_dense_mat_destroy(tmpM);
                return;
            }
            H2P_dense_mat_t V0    = V[node0];
            H2P_dense_mat_t Minv0 = Minv[node0];
            // Bij = V{node1}' * Minv{node1} * tmpB;
            H2P_dense_mat_resize(tmpM, Minv0->nrow, tmpB->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv0->nrow, tmpB->ncol, Minv0->ncol,
                1.0, Minv0->data, Minv0->ld, tmpB->data, tmpB->ld, 0.0, tmpM->data, tmpM->ld
            );
            H2P_dense_mat_resize(HSS_Bij, V0->ncol, tmpM->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, tmpM->ncol, V0->nrow,
                1.0, V0->data, V0->ld, tmpM->data, tmpM->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(n_child0, 1, child0, &node1, HSS_B, n_node, HSS_B_pair2idx);
            H2P_dense_mat_destroy(tmpB);
            H2P_dense_mat_destroy(tmpM);
            return;
        }  // End of "if (n_child0 > 0)"
    }  // End of "if (level0 > level1)"

    if (level0 < level1)
    {
        // Note: node0 must be a leaf node
        if (n_child0 > 0)
        {
            printf("[FATAL] %s bug in case 3\n", __FUNCTION__);
            return;
        }

        // 3.1: node0 and node1 are admissable
        if (is_adm)
        {
            H2P_dense_mat_t H2_Bij;
            H2P_dense_mat_init(&H2_Bij, 128, 128);
            H2_Bij->nrow = 0;
            H2P_get_Bij_block(h2mat, node0, node1, H2_Bij);
            if (H2_Bij->nrow == 0)
            {
                printf("[FATAL] %s bug in case 3.1\n", __FUNCTION__);
                H2P_dense_mat_destroy(H2_Bij);
                return;
            }
            H2P_dense_mat_t W1 = W[node1];
            // Bij = H2_B{H2_B_idx} * W{node2}';
            H2P_dense_mat_resize(HSS_Bij, H2_Bij->nrow, W1->nrow);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasTrans, H2_Bij->nrow, W1->nrow, H2_Bij->ncol,
                1.0, H2_Bij->data, H2_Bij->ld, W1->data, W1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_dense_mat_destroy(H2_Bij);
            return;
        }  // End of "if (is_adm)"

        // Otherwise: node0 and node1 are inadmissible

        // 3.2: node1 is a leaf node
        if (n_child1 == 0)
        {
            H2P_dense_mat_t H2_Dij;
            H2P_dense_mat_init(&H2_Dij, 128, 128);
            H2_Dij->nrow = 0;
            H2P_get_Dij_block(h2mat, node0, node1, H2_Dij);
            if (H2_Dij->nrow == 0)
            {
                printf("[FATAL] %s bug in case 3.2\n", __FUNCTION__);
                H2P_dense_mat_destroy(H2_Dij);
                return;
            }
            H2P_dense_mat_t V1 = V[node1];
            H2P_dense_mat_t S1 = S[node1];
            // Bij = linsolve(S{node2}, H2_Dij', struct('LT', true))' * V{node2};
            // S{node2} * X = H2_Dij', we need Bij = X' * V{node2}
            // Solve X' * S{node2}' = H2_Dij to obtain X' directly
            CBLAS_TRSM(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 
                H2_Dij->nrow, H2_Dij->ncol, 1.0, S1->data, S1->ld, H2_Dij->data, H2_Dij->ld
            );
            H2P_dense_mat_resize(HSS_Bij, H2_Dij->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, H2_Dij->nrow, V1->ncol, H2_Dij->ncol,
                1.0, H2_Dij->data, H2_Dij->ld, V1->data, V1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_dense_mat_destroy(H2_Dij);
            return;
        }  // End of "if (n_child1 == 0)"

        // 3.3: node1 is a non-leaf node
        if (n_child1 > 0)
        {
            H2P_dense_mat_t tmpB, tmpM;
            H2P_dense_mat_init(&tmpB, 128, 128);
            H2P_dense_mat_init(&tmpM, 128, 128);
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                1, n_child1, &node0, child1, 
                HSS_B, n_node, HSS_B_pair2idx, tmpB
            );
            if (tmpB->nrow == 0)
            {
                printf("[FATAL] %s bug in case 3.3\n", __FUNCTION__);
                H2P_dense_mat_destroy(tmpB);
                H2P_dense_mat_destroy(tmpM);
                return;
            }
            H2P_dense_mat_t V1    = V[node1];
            H2P_dense_mat_t Minv1 = Minv[node1];
            // Bij = tmpB * Minv{node2} * V{node2};
            H2P_dense_mat_resize(tmpM, Minv1->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv1->nrow, V1->ncol, Minv1->ncol,
                1.0, Minv1->data, Minv1->ld, V1->data, V1->ld, 0.0, tmpM->data, tmpM->ld
            );
            H2P_dense_mat_resize(HSS_Bij, tmpB->nrow, tmpM->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpB->nrow, tmpM->ncol, tmpB->ncol,
                1.0, tmpB->data, tmpB->ld, tmpM->data, tmpM->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(1, n_child1, &node0, child1, HSS_B, n_node, HSS_B_pair2idx);
            H2P_dense_mat_destroy(tmpB);
            H2P_dense_mat_destroy(tmpM);
            return;
        }  // End of "if (n_child1 > 0)"
    }  // End of "if (level0 < level1)"
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
