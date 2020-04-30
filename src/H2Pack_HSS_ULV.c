#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "utils.h"
#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"

// In H2Pack_build.c, I don't want to add the declaration of this function 
// to H2Pack_build.h and expose it. 
extern void H2P_copy_matrix_block(
    const int nrow, const int ncol, DTYPE *src_mat, const int src_ld, 
    DTYPE *dst_mat, const int dst_ld
);

// Transpose a nrow-by-ncol row-major matrix
// Input parameters:
//   nrow, ncol : Source matrix size
//   src_mat    : Size >= nrow * src_ld, source matrix
//   src_ld     : Leading dimension of the source matrix, >= ncol
//   dst_ld     : Leading dimension of the destination matrix, >= nrow
// Output parameter:
//   dst_mat : Size >= ncol * dst_ld, destination matrix
void H2P_transpose_mat(
    const int nrow, const int ncol, DTYPE *src_mat, const int src_ld, 
    DTYPE *dst_mat, const int dst_ld
)
{
    for (int j = 0; j < ncol; j++)
    {
        DTYPE *dst_row = dst_mat + j * dst_ld;
        DTYPE *src_col = src_mat + j;
        for (int i = 0; i < nrow; i++)
            dst_row[i] = src_col[i * src_ld];
    }
}

// Get element A(row, col) from CSR matrix A
// Input parameters:
//   row_ptr, col_idx, val : CSR matrix arrays
//   row, col : Target position
// Output parameter:
//   <return> : A(row, col) if exists, 0 if not
int H2P_get_CSR_elem(
    const int *row_ptr, const int *col_idx, const int *val,
    const int row, const int col
)
{
    int res = 0;
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++)
    {
        if (col_idx[i] == col) 
        {
            res = val[i];
            break;
        }
    }
    return res;
}

// Get D{node0, node1} from H2Pack
// Input parameters:
//   h2pack     : H2Pack structure
//   node{0, 1} : Target node pair
// Output parameter:
//   Dij : Target D{node0, node1} block
void H2P_get_Dij_block(H2Pack_t h2pack, const int node0, const int node1, H2P_dense_mat_t Dij)
{
    int *D_p2i_rowptr = h2pack->D_p2i_rowptr;
    int *D_p2i_colidx = h2pack->D_p2i_colidx;
    int *D_p2i_val    = h2pack->D_p2i_val;
    int D_idx = H2P_get_CSR_elem(D_p2i_rowptr, D_p2i_colidx, D_p2i_val, node0, node1);
    if (D_idx == 0)
    {
        printf("[FATAL] D{%d, %d} does not exist!\n", node0, node1);
        return;
    }
    D_idx--;
    int D_nrow = h2pack->D_nrow[D_idx];
    int D_ncol = h2pack->D_ncol[D_idx];
    H2P_dense_mat_resize(Dij, D_nrow, D_ncol);
    if (h2pack->BD_JIT == 0)
    {
        H2P_copy_matrix_block(D_nrow, D_ncol, h2pack->D_data + h2pack->D_ptr[D_idx], D_ncol, Dij->data, D_ncol);
    } else {
        int   n_point     = h2pack->n_point;
        int   krnl_dim    = h2pack->krnl_dim;
        int   *pt_cluster = h2pack->pt_cluster;
        int   pt_s0       = pt_cluster[2 * node0];
        int   pt_s1       = pt_cluster[2 * node1];
        int   pt_e0       = pt_cluster[2 * node0 + 1];
        int   pt_e1       = pt_cluster[2 * node1 + 1];
        int   node0_npt   = pt_e0 - pt_s0 + 1;
        int   node1_npt   = pt_e1 - pt_s1 + 1;
        DTYPE *coord      = h2pack->coord;
        h2pack->krnl_eval(
            coord + pt_s0, n_point, node0_npt,
            coord + pt_s1, n_point, node1_npt,
            h2pack->krnl_param, Dij->data, node1_npt * krnl_dim
        );
    }
}

// Get B{node0, node1} from H2Pack
// Input parameters:
//   h2pack     : H2Pack structure
//   node{0, 1} : Target node pair
// Output parameter:
//   Bij : Target B{node0, node1} block
void H2P_get_Bij_block(H2Pack_t h2pack, const int node0, const int node1, H2P_dense_mat_t Bij)
{
    int *B_p2i_rowptr = h2pack->B_p2i_rowptr;
    int *B_p2i_colidx = h2pack->B_p2i_colidx;
    int *B_p2i_val    = h2pack->B_p2i_val;
    int B_idx = H2P_get_CSR_elem(B_p2i_rowptr, B_p2i_colidx, B_p2i_val, node0, node1);
    if (B_idx == 0)
    {
        printf("[FATAL] B{%d, %d} does not exist!\n", node0, node1);
        return;
    }
    B_idx--;
    int B_nrow = h2pack->B_nrow[B_idx];
    int B_ncol = h2pack->B_ncol[B_idx];
    H2P_dense_mat_resize(Bij, B_nrow, B_ncol);
    if (h2pack->BD_JIT == 0)
    {
        H2P_copy_matrix_block(B_nrow, B_ncol, h2pack->B_data + h2pack->B_ptr[B_idx], B_ncol, Bij->data, B_ncol);
    } else {
        int   n_point     = h2pack->n_point;
        int   krnl_dim    = h2pack->krnl_dim;
        int   *pt_cluster = h2pack->pt_cluster;
        int   *node_level = h2pack->node_level;
        int   level0      = node_level[node0];
        int   level1      = node_level[node1];
        DTYPE *coord      = h2pack->coord;
        void  *krnl_param = h2pack->krnl_param;
        kernel_eval_fptr krnl_eval = h2pack->krnl_eval;
        H2P_dense_mat_t  *J_coord  = h2pack->J_coord;
        // (1) Two nodes are of the same level, compress on both sides
        if (level0 == level1)
        {
            krnl_eval(
                J_coord[node0]->data, J_coord[node0]->ncol, J_coord[node0]->ncol,
                J_coord[node1]->data, J_coord[node1]->ncol, J_coord[node1]->ncol,
                krnl_param, Bij->data, J_coord[node1]->ncol * krnl_dim
            );
        }
        // (2) node1 is a leaf node and its level is higher than node0's level, 
        //     only compress on node0's side
        if (level0 > level1)
        {
            int pt_s1 = pt_cluster[2 * node1];
            int pt_e1 = pt_cluster[2 * node1 + 1];
            int node1_npt = pt_e1 - pt_s1 + 1;
            krnl_eval(
                J_coord[node0]->data, J_coord[node0]->ncol, J_coord[node0]->ncol,
                coord + pt_s1, n_point, node1_npt, 
                krnl_param, Bij->data, node1_npt * krnl_dim
            );
        }
        // (3) node0 is a leaf node and its level is higher than node1's level, 
        //     only compress on node1's side
        if (level0 < level1)
        {
            int pt_s0 = pt_cluster[2 * node0];
            int pt_e0 = pt_cluster[2 * node0 + 1];
            int node0_npt = pt_e0 - pt_s0 + 1;
            krnl_eval(
                coord + pt_s0, n_point, node0_npt, 
                J_coord[node1]->data, J_coord[node1]->ncol, J_coord[node1]->ncol,
                krnl_param, Bij->data, J_coord[node1]->ncol * krnl_dim
            );
        }
    }
}

// Construct the ULV Cholesky factorization for a HSS matrix
void H2P_HSS_ULV_Cholesky_factorize(H2Pack_t h2pack, const DTYPE shift)
{
    if (!h2pack->is_HSS)
    {
        printf("[FATAL] %s failed: H2Pack is not running in HSS mode!\n", __FUNCTION__);
        return;
    }

    int n_node          = h2pack->n_node;
    int n_thread        = h2pack->n_thread;
    int n_leaf_node     = h2pack->n_leaf_node;
    int max_child       = h2pack->max_child;
    int max_level       = h2pack->max_level;
    int *children       = h2pack->children;
    int *n_child        = h2pack->n_child;
    int *level_n_node   = h2pack->level_n_node;
    int *level_nodes    = h2pack->level_nodes;
    int *mat_cluster    = h2pack->mat_cluster;
    H2P_dense_mat_t *U  = h2pack->U;
    H2P_thread_buf_t *thread_buf = h2pack->tb;

    int *ULV_Ls;
    H2P_int_vec_t   *ULV_idx;
    H2P_dense_mat_t *ULV_Q, *ULV_L, *U_mid, *D_mid;
    ULV_Ls  = (int*)             malloc(sizeof(int)             * n_node);
    ULV_idx = (H2P_int_vec_t*)   malloc(sizeof(H2P_int_vec_t)   * n_node);
    ULV_Q   = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    ULV_L   = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    U_mid   = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    D_mid   = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    assert(ULV_Ls != NULL && ULV_idx != NULL && ULV_Q != NULL && ULV_L != NULL);
    assert(U_mid != NULL && D_mid != NULL);
    for (int i = 0; i < n_node; i++)
    {
        ULV_idx[i] = NULL;
        ULV_Q[i]   = NULL;
        ULV_L[i]   = NULL;
        U_mid[i]   = NULL;
        D_mid[i]   = NULL;
    }

    // Non-root node construction
    int is_SPD = 1;
    for (int i = max_level; i >= 0; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);
        if (!is_SPD) continue;
        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            H2P_int_vec_t   tmpidx = thread_buf[tid]->idx0;
            H2P_dense_mat_t tmpU   = thread_buf[tid]->mat0;
            H2P_dense_mat_t tmpD   = thread_buf[tid]->mat1;
            H2P_dense_mat_t tmpB   = thread_buf[tid]->mat2;
            H2P_dense_mat_t tmpM   = thread_buf[tid]->mat2;

            #pragma omp for
            for (int j = 0; j < level_i_n_node; j++)
            {
                if (!is_SPD) continue;
                int node = level_i_nodes[j];
                int node_n_child = n_child[node];
                int *node_children = children + node * max_child;
                // 1. Construct tmpU and tmpD for factorization
                int U_nrow, U_ncol, U_diff;
                if (node_n_child == 0)
                {
                    // Leaf node, use the original U and D
                    H2P_dense_mat_resize(tmpU, U[node]->nrow, U[node]->ncol);
                    H2P_copy_matrix_block(U[node]->nrow, U[node]->ncol, U[node]->data, U[node]->ld, tmpU->data, tmpU->ld);
                    H2P_get_Dij_block(h2pack, node, node, tmpD);
                    for (int k = 0; k < tmpD->nrow; k++)
                        tmpD->data[k * tmpD->ld + k] += shift;
                } else {
                    // Non-leaf node, assemble tmpU and tmpD from mid_U and mid_D
                    // (1) Accumulate the dimension of each compressed child's diagonal block
                    H2P_int_vec_set_capacity(tmpidx, node_n_child + 1);
                    int *offset = tmpidx->data;
                    offset[0] = 0;
                    U_nrow = 0;
                    for (int k = 0; k < node_n_child; k++)
                    {
                        int child_k = node_children[k];
                        offset[k + 1] = offset[k] + D_mid[child_k]->nrow;
                        U_nrow += D_mid[child_k]->nrow;
                    }
                    // (2) Build the compressed diagonal block
                    H2P_dense_mat_resize(tmpU, U_nrow, U_nrow);
                    H2P_dense_mat_resize(tmpD, U_nrow, U_nrow);
                    memset(tmpU->data, 0, sizeof(DTYPE) * U_nrow * U_nrow);
                    memset(tmpD->data, 0, sizeof(DTYPE) * U_nrow * U_nrow);
                    for (int k = 0; k < node_n_child; k++)
                    {
                        int child_k = node_children[k];
                        // idx_k = offset(k) : offset(k+1)-1;
                        int idx_k_s = offset[k];
                        int idx_k_len = offset[k + 1] - idx_k_s;
                        // Diagonal blocks
                        // tmpD(idx_k, idx_k) = D_mid{child_k};
                        H2P_copy_matrix_block(
                            D_mid[child_k]->nrow, D_mid[child_k]->ncol, D_mid[child_k]->data, 
                            D_mid[child_k]->ld, tmpD->data + idx_k_s * (tmpD->ld + 1), tmpD->ld
                        );
                        // tmpU(idx_k, idx_k) = U_mid{child_k};
                        H2P_copy_matrix_block(
                            U_mid[child_k]->nrow, U_mid[child_k]->ncol, U_mid[child_k]->data, 
                            U_mid[child_k]->ld, tmpU->data + idx_k_s * (tmpU->ld + 1), tmpU->ld
                        );
                        // Off-diagonal blocks
                        for (int l = k + 1; l < node_n_child; l++)
                        {
                            int child_l = node_children[l];
                            // idx_l = offset(l) : offset(l+1)-1;
                            int idx_l_s = offset[l];
                            int idx_l_len = offset[l + 1] - idx_l_s;
                            // B_idx = B_pair2idx(child_k, child_l);
                            H2P_get_Bij_block(h2pack, child_k, child_l, tmpB);
                            // tmpD(idx_k, idx_l) = U_mid{child_k} * B{B_idx} * U_mid{child_l}';
                            H2P_dense_mat_resize(tmpM, tmpB->nrow, U_mid[child_l]->nrow);
                            CBLAS_GEMM(
                                CblasRowMajor, CblasNoTrans, CblasTrans, tmpM->nrow, tmpM->ncol, tmpB->ncol,
                                1.0, tmpB->data, tmpB->ld, U_mid[child_l]->data, U_mid[child_l]->ld, 
                                0.0, tmpM->data, tmpM->ld
                            );
                            DTYPE *tmpD_kl = tmpD->data + idx_k_s * tmpD->ld + idx_l_s;
                            DTYPE *tmpD_lk = tmpD->data + idx_l_s * tmpD->ld + idx_k_s;
                            CBLAS_GEMM(
                                CblasRowMajor, CblasNoTrans, CblasNoTrans, U_mid[child_k]->nrow, tmpM->ncol, tmpM->nrow,
                                1.0, U_mid[child_k]->data, U_mid[child_k]->ld, tmpM->data, tmpM->ld, 
                                0.0, tmpD_kl, tmpD->ld
                            );
                            // tmpD(idx_l, idx_k) = tmpD(idx_k, idx_l)';
                            H2P_transpose_mat(idx_k_len, idx_l_len, tmpD_kl, tmpD->ld, tmpD_lk, tmpD->ld);
                        }
                    }  // End of k loop
                    // if (level > 1), tmpU = tmpU * U{node}; end
                    if (i > 0)
                    {
                        H2P_dense_mat_resize(tmpM, tmpU->nrow, U[node]->ncol);
                        CBLAS_GEMM(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpM->nrow, tmpM->ncol, tmpU->ncol,
                            1.0, tmpU->data, tmpU->ld, U[node]->data, U[node]->ld, 0.0, tmpM->data, tmpM->ld
                        );
                        H2P_dense_mat_resize(tmpU, tmpM->nrow, tmpM->ncol);
                        H2P_copy_matrix_block(tmpM->nrow, tmpM->ncol, tmpM->data, tmpM->ld, tmpU->data, tmpU->ld);
                    }
                }  // End of "if (node_n_child == 0)"
                U_nrow = tmpU->nrow;
                U_ncol = tmpU->ncol;
                U_diff = U_nrow - U_ncol;

                // 2. Cholesky factorization
                // size(tmpU) = [U_nrow, U_ncol]
                // size(tmpD) = [U_nrow, U_nrow]
                // U_nrow >= U_ncol always holds
                int info;
                if (i > 0)
                {
                    H2P_dense_mat_init(&ULV_Q[node], U_nrow + 1, U_ncol);
                    H2P_dense_mat_init(&U_mid[node], U_ncol, U_ncol);
                    // [Q{node}, R] = qr(tmpU);
                    H2P_copy_matrix_block(U_nrow, U_ncol, tmpU->data, tmpU->ld, ULV_Q[node]->data, ULV_Q[node]->ld);
                    DTYPE *A   = ULV_Q[node]->data;
                    DTYPE *tau = ULV_Q[node]->data + U_nrow * U_ncol;
                    info = LAPACK_GEQRF(LAPACK_ROW_MAJOR, U_nrow, U_ncol, A, U_ncol, tau);
                    // U_mid{node} = R(1 : U_ncol, :);
                    for (int k = 0; k < U_ncol; k++)
                    {
                        DTYPE *A_k = A + k * U_ncol;
                        DTYPE *R_k = U_mid[node]->data + k * U_ncol;
                        if (k > 0) memset(R_k, 0, sizeof(DTYPE) * k);
                        memcpy(R_k + k, A_k + k, sizeof(DTYPE) * (U_ncol - k));
                    }
                    ULV_Ls[node] = U_ncol;
                    // tmpD = Q{node}' * tmpD * Q{node};
                    info = LAPACK_ORMQR(LAPACK_ROW_MAJOR, 'L', 'T', U_nrow, U_nrow, U_ncol, A, U_ncol, tau, tmpD->data, U_nrow);
                    info = LAPACK_ORMQR(LAPACK_ROW_MAJOR, 'R', 'N', U_nrow, U_nrow, U_ncol, A, U_ncol, tau, tmpD->data, U_nrow);
                    // tmpD11 = tmpD(1 : U_ncol, 1 : U_ncol);
                    // tmpD21 = tmpD((U_ncol+1) : end, 1 : U_ncol);
                    // tmpD22 = tmpD((U_ncol+1) : end, (U_ncol+1) : end);
                    DTYPE *tmpD11 = tmpD->data;
                    DTYPE *tmpD21 = tmpD->data + U_ncol * U_nrow;
                    DTYPE *tmpD22 = tmpD->data + U_ncol * (U_nrow + 1);
                    if (U_diff > 0)
                    {
                        // [tmpL, chol_flag] = chol(tmpD22, 'lower');
                        // Here tmpL is stored in tmpD22
                        info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', U_diff, tmpD22, U_nrow);
                        for (int k = 0; k < U_diff - 1; k++)
                        {
                            DTYPE *tmpL_kk1 = tmpD22 + k * U_nrow + (k + 1);
                            int n_zero_row = U_diff - 1 - k;
                            memset(tmpL_kk1, 0, sizeof(DTYPE) * n_zero_row);
                        }
                        if (info != 0)
                        {
                            printf("[FATAL] Node %d potrf() return flag = %d, the target matrix is not SPD, consider adding a shift\n", node, info);
                            is_SPD = 0;
                        }
                        // LD21 = tmpL \ tmpD21;
                        // Here LD21 is stored in tmpD21
                        CBLAS_TRSM(
                            CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                            U_diff, U_ncol, 1.0, tmpD22, U_nrow, tmpD21, U_nrow
                        );
                    }
                    // L{node} = [eye(U_ncol), LD21'; zeros(U_diff, U_ncol), tmpL];
                    H2P_dense_mat_init(&ULV_L[node], U_nrow, U_nrow);
                    for (int k = 0; k < U_ncol; k++)
                    {
                        DTYPE *L_k = ULV_L[node]->data + k * U_nrow;
                        memset(L_k, 0, sizeof(DTYPE) * U_nrow);
                        L_k[k] = 1.0;
                    }
                    if (U_diff > 0)
                    {
                        DTYPE *L12 = ULV_L[node]->data + U_ncol;
                        DTYPE *L21 = ULV_L[node]->data + U_ncol * U_nrow;
                        DTYPE *L22 = ULV_L[node]->data + U_ncol * (U_nrow + 1);
                        H2P_transpose_mat(U_diff, U_ncol, tmpD21, U_nrow, L12, U_nrow);
                        for (int k = 0; k < U_diff; k++)
                            memset(L21 + k * U_nrow, 0, sizeof(DTYPE) * U_ncol);
                        H2P_copy_matrix_block(U_diff, U_diff, tmpD22, U_nrow, L22, U_nrow);
                    }
                    // D_mid{node} = tmpD11 - LD21' * LD21;
                    H2P_dense_mat_init(&D_mid[node], U_ncol, U_ncol);
                    H2P_copy_matrix_block(U_ncol, U_ncol, tmpD11, U_nrow, D_mid[node]->data, U_ncol);
                    if (U_diff > 0)
                    {
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasNoTrans, U_ncol, U_ncol, U_diff,
                            -1.0, tmpD21, U_nrow, tmpD21, U_nrow, 1.0, D_mid[node]->data, U_ncol
                        );
                    }
                } else {  // Else of "if (i > 0)"
                    // Q{node} = eye(size(tmpD)); 
                    // We don't actually need Q{node} when node == root, just make a placeholder here
                    H2P_dense_mat_init(&ULV_Q[node], 1, 1);
                    H2P_dense_mat_init(&ULV_L[node], U_nrow, U_nrow);
                    ULV_Ls[node] = 0;
                    // [L{node}, chol_flag] = chol(tmpD, 'lower');
                    H2P_copy_matrix_block(U_nrow, U_nrow, tmpD->data, U_nrow, ULV_L[node]->data, U_nrow);
                    info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', U_nrow, ULV_L[node]->data, U_nrow);
                    for (int k = 0; k < U_nrow; k++)
                    {
                        DTYPE *L_kk1 = ULV_L[node]->data + k * U_nrow + (k + 1);
                        int n_zero_row = U_nrow - 1 - k;
                        memset(L_kk1, 0, sizeof(DTYPE) * n_zero_row);
                    }
                    if (info != 0)
                    {
                        printf("[FATAL] Node %d potrf() return flag = %d, the target matrix is not SPD, consider adding a shift\n", node, info);
                        is_SPD = 0;
                    }
                }  // End of "if (i > 0)"

                // 3. Construct ULV_idx{node}, row indices where ULV_Q{node} and ULV_L{node} are applied to
                if (!is_SPD) continue;
                if (node_n_child == 0)
                {
                    int cluster_s   = mat_cluster[2 * node];
                    int cluster_e   = mat_cluster[2 * node + 1];
                    int cluster_len = cluster_e - cluster_s + 1;
                    H2P_int_vec_init(&ULV_idx[node], cluster_len);
                    for (int k = 0; k < cluster_len; k++)
                        ULV_idx[node]->data[k] = cluster_s + k;
                    ULV_idx[node]->length = cluster_len;
                    assert(ULV_idx[node]->length == ULV_L[node]->nrow);
                } else {
                    int idx_size = 0;
                    for (int k = 0; k < node_n_child; k++)
                    {
                        int child_k = node_children[k];
                        idx_size += D_mid[child_k]->nrow;
                    }
                    H2P_int_vec_init(&ULV_idx[node], idx_size);
                    idx_size = 0;
                    for (int k = 0; k < node_n_child; k++)
                    {
                        int child_k = node_children[k];
                        for (int l = 0; l < D_mid[child_k]->nrow; l++)
                            ULV_idx[node]->data[idx_size + l] = ULV_idx[child_k]->data[l];
                        idx_size += D_mid[child_k]->nrow;
                    }
                    ULV_idx[node]->length = idx_size;
                    assert(ULV_idx[node]->length == ULV_L[node]->nrow);
                }  // End of "if (node_n_child == 0)"

                // 4. Free U_mid{child_k} and D_mid{child_k} since we no longer need them
                if (node_n_child > 0)
                {
                    for (int k = 0; k < node_n_child; k++)
                    {
                        int child_k = node_children[k];
                        H2P_dense_mat_destroy(U_mid[child_k]);
                        H2P_dense_mat_destroy(D_mid[child_k]);
                        U_mid[child_k] = NULL;
                        D_mid[child_k] = NULL;
                    }
                }
            }  // End of j loop
        }  // End of "#pragma omp parallel"
    }  // End of i loop

    // Free intermediate matrices and set the output matrices
    for (int i = 0; i < n_node; i++)
    {
        if (U_mid[i] != NULL) H2P_dense_mat_destroy(U_mid[i]);
        if (D_mid[i] != NULL) H2P_dense_mat_destroy(D_mid[i]);
    }
    free(U_mid);
    free(D_mid);
    h2pack->ULV_Ls  = ULV_Ls;
    h2pack->ULV_idx = ULV_idx;
    h2pack->ULV_Q   = ULV_Q;
    h2pack->ULV_L   = ULV_L;
}

// Solve the linear system A_{HSS} * x = b using the HSS ULV Cholesky factorization
void H2P_HSS_ULV_Cholesky_solve(H2Pack_t h2pack, const int op, const DTYPE *b, DTYPE *x)
{
    if (!h2pack->is_HSS)
    {
        printf("[FATAL] %s failed: H2Pack is not running in HSS mode!\n", __FUNCTION__);
        return;
    }
    if (h2pack->ULV_idx == NULL)
    {
        printf("[FATAL] %s failed: need to call H2P_HSS_ULV_Cholesky_factorize() first!\n", __FUNCTION__);
        return;
    }
    if (op < 1 || op > 3) 
    {
        printf("[FATAL] %s failed: invalid <op> value\n", __FUNCTION__);
        return;
    }

    int n_leaf_node   = h2pack->n_leaf_node;
    int max_level     = h2pack->max_level;
    int n_thread      = h2pack->n_thread;
    int *level_n_node = h2pack->level_n_node;
    int *level_nodes  = h2pack->level_nodes;
    int *ULV_Ls       = h2pack->ULV_Ls;
    H2P_int_vec_t    *ULV_idx = h2pack->ULV_idx;
    H2P_dense_mat_t  *ULV_Q   = h2pack->ULV_Q;
    H2P_dense_mat_t  *ULV_L   = h2pack->ULV_L;
    H2P_thread_buf_t *thread_buf = h2pack->tb;

    memcpy(x, b, sizeof(DTYPE) * h2pack->krnl_mat_size);
    
    int solve_LT = op & 1;
    if (solve_LT)
    {
        // Level by level up sweep
        for (int i = max_level; i > 0; i--)
        {
            int *level_i_nodes = level_nodes + i * n_leaf_node;
            int level_i_n_node = level_n_node[i];
            int n_thread_i = MIN(level_i_n_node, n_thread);
            #pragma omp parallel num_threads(n_thread_i)
            {
                int tid = omp_get_thread_num();
                H2P_dense_mat_t x0 = thread_buf[tid]->mat0;

                #pragma omp for schedule(dynamic)
                for (int j = 0; j < level_i_n_node; j++)
                {
                    int node = level_i_nodes[j];
                    H2P_int_vec_t   idx = ULV_idx[node];
                    H2P_dense_mat_t Q   = ULV_Q[node];
                    H2P_dense_mat_t L   = ULV_L[node];
                    assert(idx->length == L->nrow);
                    int I_size = ULV_Ls[node];
                    int L_size = L->nrow - I_size;
                    // b0 = Q{node}' * x(idx);
                    H2P_dense_mat_resize(x0, idx->length, 1);
                    for (int k = 0; k < idx->length; k++) x0->data[k] = x[idx->data[k]];
                    DTYPE *b0  = x0->data;
                    DTYPE *A   = Q->data;
                    DTYPE *tau = Q->data + (Q->nrow - 1) * Q->ncol;
                    LAPACK_ORMQR(LAPACK_ROW_MAJOR, 'L', 'T', L->nrow, 1, Q->ncol, A, Q->ncol, tau, x0->data, 1);
                    // b1 = b0(1 : I_size);
                    // b2 = b0(I_size+1 : end);
                    DTYPE *b1 = b0;
                    DTYPE *b2 = b0 + I_size;
                    // L12 = L{node}(1 : I_size, I_size+1 : end);
                    // L22 = L{node}(I_size+1 : end, I_size+1 : end);
                    DTYPE *L12 = L->data + I_size;
                    DTYPE *L22 = L->data + I_size * (L->ld + 1);
                    // x2 = L22 \ b2;
                    DTYPE *x2 = b2;
                    CBLAS_TRSM(
                        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                        L_size, 1, 1.0, L22, L->ld, b2, 1
                    );
                    // x1 = b1 - L12 * x2;
                    DTYPE *x1 = b1;
                    CBLAS_GEMV(
                        CblasRowMajor, CblasNoTrans, I_size, L_size, 
                        -1.0, L12, L->ld, x2, 1, 1.0, x1, 1
                    );
                    // x(idx) = L{node} \ x(idx);
                    for (int k = 0; k < idx->length; k++) x[idx->data[k]] = x0->data[k];
                }  // End of j loop
            }  // End of "#pragma omp parallel"
        }  // End of i loop 

        // Root node, x(idx) = L{node} \ x(idx);
        int node = level_nodes[0];
        H2P_int_vec_t   idx = ULV_idx[node];
        H2P_dense_mat_t L   = ULV_L[node];
        H2P_dense_mat_t x0  = thread_buf[0]->mat0;
        H2P_dense_mat_resize(x0, idx->length, 1);
        for (int k = 0; k < idx->length; k++) x0->data[k] = x[idx->data[k]];
        CBLAS_TRSM(
            CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
            L->nrow, 1, 1.0, L->data, L->ld, x0->data, 1
        );
        for (int k = 0; k < idx->length; k++) x[idx->data[k]] = x0->data[k];
    }  // End of "if (solve_LT)"

    int solve_L = op & 2;
    if (solve_L)
    {
        // Root node, x(idx) = L{node}' \ x(idx);
        int node = level_nodes[0];
        H2P_int_vec_t   idx = ULV_idx[node];
        H2P_dense_mat_t L   = ULV_L[node];
        H2P_dense_mat_t x0  = thread_buf[0]->mat0;
        H2P_dense_mat_resize(x0, idx->length, 1);
        for (int k = 0; k < idx->length; k++) x0->data[k] = x[idx->data[k]];
        CBLAS_TRSM(
            CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
            L->nrow, 1, 1.0, L->data, L->ld, x0->data, 1
        );
        for (int k = 0; k < idx->length; k++) x[idx->data[k]] = x0->data[k];

        // Level by level down sweep
        for (int i = 1; i <= max_level; i++)
        {
            int *level_i_nodes = level_nodes + i * n_leaf_node;
            int level_i_n_node = level_n_node[i];
            int n_thread_i = MIN(level_i_n_node, n_thread);
            #pragma omp parallel num_threads(n_thread_i)
            {
                int tid = omp_get_thread_num();
                H2P_dense_mat_t x0 = thread_buf[tid]->mat0;

                #pragma omp for schedule(dynamic)
                for (int j = 0; j < level_i_n_node; j++)
                {
                    int node = level_i_nodes[j];
                    H2P_int_vec_t   idx = ULV_idx[node];
                    H2P_dense_mat_t Q   = ULV_Q[node];
                    H2P_dense_mat_t L   = ULV_L[node];
                    assert(idx->length == L->nrow);
                    int I_size = ULV_Ls[node];
                    int L_size = L->nrow - I_size;
                    // b0 = x(idx);
                    // b1 = b0(1 : I_size);
                    // b2 = b0(I_size+1 : end);
                    H2P_dense_mat_resize(x0, idx->length, 1);
                    for (int k = 0; k < idx->length; k++) x0->data[k] = x[idx->data[k]];
                    DTYPE *b0 = x0->data;
                    DTYPE *b1 = b0;
                    DTYPE *b2 = b0 + I_size;
                    // L12 = L{node}(1 : I_size, I_size+1 : end);
                    // L22 = L{node}(I_size+1 : end, I_size+1 : end);
                    DTYPE *L12 = L->data + I_size;
                    DTYPE *L22 = L->data + I_size * (L->ld + 1);
                    // b2 = b2 - L12' * b1;
                    CBLAS_GEMV(
                        CblasRowMajor, CblasTrans, I_size, L_size,
                        -1.0, L12, L->ld, b1, 1, 1.0, b2, 1
                    );
                    // b2 = L22' \ b2;
                    CBLAS_TRSM(
                        CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                        L_size, 1, 1.0, L22, L->ld, b2, 1
                    );
                    // x(idx) = Q{node} * [b1; b2];
                    DTYPE *b3  = x0->data;
                    DTYPE *A   = Q->data;
                    DTYPE *tau = Q->data + (Q->nrow - 1) * Q->ncol;
                    LAPACK_ORMQR(LAPACK_ROW_MAJOR, 'L', 'N', L->nrow, 1, Q->ncol, A, Q->ncol, tau, x0->data, 1);
                    for (int k = 0; k < idx->length; k++) x[idx->data[k]] = x0->data[k];
                }  // End of j loop
            }  // End of "#pragma omp parallel"
        }  // End of i loop 
    }  // End of "if (solve_L)"
}
