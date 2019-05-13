#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_ID_compress.h"

// Symmetry kernel function, should be an input parameter of H2P_build later
// Input parameters:
//   x, y : Coordinate of two points
//   dim  : Dimension of point coordinate
// Output parameter:
//   <return> : Output of kernel function
DTYPE kernel_func(const DTYPE *x, const DTYPE *y, const int dim)
{
    // Use the reciprocal kernel for testing
    DTYPE res = 0.0;
    for (int i = 0; i < dim; i++)
    {
        DTYPE delta = x[i] - y[i];
        res += delta * delta;
    }
    res = 1.0 / DSQRT(res);
    return res;
}

// Evaluate a kernel matrix
// Input parameters:
//   h2pack : H2Pack structure
//   nx     : Number of points in the x point set
//   ny     : Number of points in the y point set
//   idx_x  : Indices of points in the x point set
//   idx_y  : Indices of points in the y point set
// Output parameter:
//   kernel_mat : Obtained kernel matrix, nx-by-ny
void H2P_eval_kernel_matrix(
    H2Pack_t h2pack, const int nx, const int ny,
    const int *idx_x, const int *idx_y, H2P_dense_mat_t kernel_mat
)
{
    int   dim    = h2pack->dim;
    DTYPE *coord = h2pack->coord;
    H2P_dense_mat_resize(kernel_mat, nx, ny);
    for (int i = 0; i < nx; i++)
    {
        DTYPE *kernel_mat_row = kernel_mat->data + i * kernel_mat->ld;
        const DTYPE *x_i = coord + idx_x[i] * dim;
        for (int j = 0; j < ny; j++)
        {
            const DTYPE *y_j = coord + idx_y[j] * dim;
            kernel_mat_row[j] = kernel_func(x_i, y_j, dim);
        }
    }
}

void H2P_build_UJ_direct(H2Pack_t h2pack)
{
    int   dim            = h2pack->dim;
    int   n_node         = h2pack->n_node;
    int   n_leaf_node    = h2pack->n_leaf_node;
    int   max_child      = h2pack->max_child;
    int   max_level      = h2pack->max_level;
    int   min_adm_level  = h2pack->min_adm_level;
    int   *children      = h2pack->children;
    int   *n_child       = h2pack->n_child;
    int   *level_n_node  = h2pack->level_n_node;
    int   *level_nodes   = h2pack->level_nodes;
    int   *leaf_nodes    = h2pack->leaf_nodes;
    int   *cluster       = h2pack->cluster;
    int   *node_adm_list = h2pack->node_adm_list;
    int   *node_adm_cnt  = h2pack->node_adm_cnt;
    DTYPE *coord         = h2pack->coord;
    
    // 1. Allocate U and J
    h2pack->n_UJ = n_node;
    h2pack->U = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    h2pack->J = (H2P_int_vec_t*)   malloc(sizeof(H2P_int_vec_t)   * n_node);
    assert(h2pack->U != NULL && h2pack->J != NULL);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i] = NULL;
        h2pack->J[i] = NULL;
    }
    H2P_dense_mat_t *U = h2pack->U;
    H2P_int_vec_t   *J = h2pack->J;
    
    // 2. Skeleton row sets for leaf nodes: all points in that box
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int s_index = cluster[node * 2];
        int e_index = cluster[node * 2 + 1];
        int n_point = e_index - s_index + 1;
        H2P_int_vec_init(&J[node], n_point);
        for (int j = 0; j < n_point; j++)
            J[node]->data[j] = s_index + j;
        J[node]->length = n_point;
    }
    
    // 3. Hierarchical construction level by level. min_adm_level is the 
    //    highest level that still has admissible blocks, so we only need 
    //    to compress matrix blocks to that level since higher level blocks 
    //    are inadmissible and cannot be compressed.
    int flag = 0;
    DTYPE rel_tol = 1e-6;
    H2P_int_vec_t   col_idx, sub_idx;
    H2P_dense_mat_t A_block;
    H2P_int_vec_init(&col_idx, 0);
    H2P_int_vec_init(&sub_idx, 0);
    H2P_dense_mat_init(&A_block, 64, 64);
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        
        // (1) Update row indices associated with clusters at i-th level
        for (int j = 0; j < level_n_node[i]; j++)
        {
            int node = level_i_nodes[j];
            int n_child_node = n_child[node];
            int *child_nodes = children + node * max_child;
            int J_child_size = 0;
            for (int i_child = 0; i_child < n_child_node; i_child++)
            {
                int i_child_node = child_nodes[i_child];
                J_child_size += J[i_child_node]->length;
            }
            if (J[node] == NULL) H2P_int_vec_init(&J[node], J_child_size);
            else H2P_int_vec_set_capacity(J[node], J[node]->length + J_child_size);
            for (int i_child = 0; i_child < n_child_node; i_child++)
            {
                int i_child_node = child_nodes[i_child];
                H2P_int_vec_concatenate(J[node], J[i_child_node]);
            }
        }

        // (2) Compression at the i-th level
        for (int j = 0; j < level_n_node[i]; j++)
        {
            int node = level_i_nodes[j];
            if (node_adm_cnt[node] == 0)
            {
                H2P_int_vec_init(&J[node], 1);
                H2P_dense_mat_init(&U[node], 1, 1);
                U[node]->nrow = 0;
                U[node]->ncol = 0;
                U[node]->ld   = 0;
            } else {
                int n_col_idx = 0;
                int *adm_list = node_adm_list + node * n_node;
                for (int k = 0; k < node_adm_cnt[node]; k++)
                {
                    int adm_node_k = adm_list[k];
                    n_col_idx += J[adm_node_k]->length;
                }
                H2P_int_vec_set_capacity(col_idx, n_col_idx);
                col_idx->length = 0;
                for (int k = 0; k < node_adm_cnt[node]; k++)
                {
                    int adm_node_k = adm_list[k];
                    H2P_int_vec_concatenate(col_idx, J[adm_node_k]);
                }
                
                int nx = J[node]->length;
                int ny = n_col_idx;
                H2P_dense_mat_resize(A_block, nx, ny);
                H2P_int_vec_set_capacity(sub_idx, nx);
                H2P_eval_kernel_matrix(h2pack, nx, ny, J[node]->data, col_idx->data, A_block);
                H2P_ID_compress(A_block, QR_REL_NRM, &rel_tol, &U[node], sub_idx->data);
                for (int k = 0; k < U[node]->ncol; k++)
                    J[node]->data[k] = J[node]->data[sub_idx->data[k]];
                J[node]->length = U[node]->ncol;
            }
        }
    }
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        if (h2pack->U[i] == NULL)
        {
            H2P_dense_mat_init(&U[i], 1, 1);
            U[i]->nrow = 0;
            U[i]->ncol = 0;
            U[i]->ld   = 0;
        }
        if (h2pack->J[i] == NULL)
            H2P_int_vec_init(&J[i], 1);
    }
    H2P_int_vec_destroy(col_idx);
    H2P_int_vec_destroy(sub_idx);
    H2P_dense_mat_destroy(A_block);
}

void H2P_build_B(H2Pack_t h2pack)
{

}

void H2P_build_D(H2Pack_t h2pack)
{
    
}

// Build H2 representation with a kernel function
void H2P_build(H2Pack_t h2pack)
{
    double st, et;

    // 1. Build projection matrices and skeleton row sets
    st = H2P_get_wtime_sec();
    H2P_build_UJ_direct(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[1] = et - st;

    // 2. Build generator matrices
    st = H2P_get_wtime_sec();
    H2P_build_B(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[2] = et - st;
    
    // 3. Build dense blocks
    st = H2P_get_wtime_sec();
    H2P_build_D(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[3] = et - st;
}


