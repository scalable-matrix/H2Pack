#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"

// TODO: Check if we should replace CBLAS_GEMV with an inline function
//       to reduce the overhead of calling external library.

// H2 representation matvec upward sweep, calculate U_j^T * x_j
void H2P_matvec_upward_sweep(H2Pack_t h2pack, const DTYPE *x, DTYPE *y)
{
    int max_child      = h2pack->max_child;
    int n_node         = h2pack->n_node;
    int n_leaf_node    = h2pack->n_leaf_node;
    int max_level      = h2pack->max_level;
    int min_adm_level  = h2pack->min_adm_level;
    int max_adm_height = h2pack->max_adm_height;
    int *children      = h2pack->children;
    int *n_child       = h2pack->n_child;
    int *height_n_node = h2pack->height_n_node;
    int *node_level    = h2pack->node_level;
    int *height_nodes  = h2pack->height_nodes;
    int *cluster       = h2pack->cluster;
    
    // 1. Initialize y0 on the first run
    if (h2pack->y0 == NULL)
    {
        h2pack->y0 = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
        assert(h2pack->y0 != NULL);
        H2P_dense_mat_t *y0 = h2pack->y0;
        H2P_dense_mat_t *U  = h2pack->U;
        for (int node = 0; node < n_node; node++)
        {
            int ncol = U[node]->ncol;
            if (ncol > 0) 
            {
                H2P_dense_mat_init(&y0[node], ncol, 1);
            } else {
                H2P_dense_mat_init(&y0[node], 1, 1);
                y0[node]->nrow = 0;
                y0[node]->ncol = 0;
                y0[node]->ld   = 0;
            }
        }
    }
    
    // 2. Upward sweep
    H2P_dense_mat_t *y0 = h2pack->y0;
    H2P_dense_mat_t *U  = h2pack->U;
    for (int i = 0; i <= max_adm_height; i++)
    {
        int *height_i_nodes = height_nodes + i * n_leaf_node;
        for (int j = 0; j < height_n_node[i]; j++)
        {
            int node  = height_i_nodes[j];
            int level = node_level[node];
            if (level < min_adm_level) continue;
            int n_child_node = n_child[node];
            int *child_nodes = children + node * max_child;
            if (n_child_node == 0)
            {
                // Leaf node, directly calculate U_j^T * x_j
                const DTYPE *x_spos = x + cluster[node * 2];
                CBLAS_GEMV(
                    CblasRowMajor, CblasTrans, U[node]->nrow, U[node]->ncol, 
                    1.0, U[node]->data, U[node]->ld, 
                    x_spos, 1, 0.0, y0[node]->data, 1
                );
            } else {
                // Non-leaf node, concatenate children node's y0 and multiple
                // it with U_j^T
                int U_srow = 0;
                for (int k = 0; k < n_child_node; k++)
                {
                    int child_k = child_nodes[k];
                    int child_k_len = y0[child_k]->nrow; 
                    DTYPE *U_node_k = U[node]->data + U_srow * U[node]->ld;
                    DTYPE beta = (k == 0) ? 0.0 : 1.0;
                    CBLAS_GEMV(
                        CblasRowMajor, CblasTrans, child_k_len, U[node]->ncol, 
                        1.0, U_node_k, U[node]->ld, 
                        y0[child_k]->data, 1, beta, y0[node]->data, 1
                    );
                    U_srow += child_k_len;
                }
            }
        }
    }
}

// H2 representation matvec intermediate sweep, calculate B_{ij} * (U_j^T * x_j)
void H2P_matvec_intermediate_sweep(H2Pack_t h2pack, const DTYPE *x, DTYPE *y)
{
    int n_node       = h2pack->n_node;
    int n_r_adm_pair = h2pack->n_r_adm_pair;
    int *r_adm_pairs = h2pack->r_adm_pairs;
    int *node_level  = h2pack->node_level;
    int *cluster     = h2pack->cluster;
    
    // 1. Initialize y1 on the first run
    if (h2pack->y1 == NULL)
    {
        h2pack->y1 = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
        assert(h2pack->y1 != NULL);
        for (int i = 0; i < n_node; i++) h2pack->y1[i] = NULL;
    }
    H2P_dense_mat_t *y0 = h2pack->y0;
    H2P_dense_mat_t *y1 = h2pack->y1;
    H2P_dense_mat_t *U  = h2pack->U;
    H2P_dense_mat_t *B  = h2pack->B;
    
    // Use ld to mark if y1[i] is visited in this intermediate sweep
    for (int i = 0; i < n_node; i++) 
        if (y1[i] != NULL) y1[i]->ld = 1;

    // 2. Intermediate sweep
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int node0  = r_adm_pairs[2 * i];
        int node1  = r_adm_pairs[2 * i + 1];
        int level0 = node_level[node0];
        int level1 = node_level[node1];
        H2P_dense_mat_t Bi = B[i];
        
        // (1) Initialize y1[node0] and y1[node1] if necessary
        if (y1[node0] == NULL) H2P_dense_mat_init(&y1[node0], U[node0]->ncol, 1);
        if (y1[node1] == NULL) H2P_dense_mat_init(&y1[node1], U[node1]->ncol, 1);
        // Reset y1[node0] and y1[node1] to 0 to remove results from previous matvec
        if (y1[node0]->ld == 1)
        {
            memset(y1[node0]->data, 0, sizeof(DTYPE) * y1[node0]->nrow);
            y1[node0]->ld = 2;
        }
        if (y1[node1]->ld == 1)
        {
            memset(y1[node1]->data, 0, sizeof(DTYPE) * y1[node1]->nrow);
            y1[node1]->ld = 2;
        }
        
        // (2) Two nodes are of the same level, compress on both sides
        if (level0 == level1)
        {
            CBLAS_GEMV(
                CblasRowMajor, CblasNoTrans, Bi->nrow, Bi->ncol, 
                1.0, Bi->data, Bi->ld, 
                y0[node1]->data, 1, 1.0, y1[node0]->data, 1
            );
            CBLAS_GEMV(
                CblasRowMajor, CblasTrans, Bi->nrow, Bi->ncol, 
                1.0, Bi->data, Bi->ld, 
                y0[node0]->data, 1, 1.0, y1[node1]->data, 1
            );
        }
        
        // (3) node1 is a leaf node and its level is higher than node0's level, 
        //     only compressed on node0's side, node1's side don't need the 
        //     downward sweep and can directly accumulate result to output vector
        if (level0 > level1)
        {
            int s_index = cluster[node1 * 2];
            const DTYPE *x_spos = x + s_index;
            DTYPE *y_spos = y + s_index;
            CBLAS_GEMV(
                CblasRowMajor, CblasNoTrans, Bi->nrow, Bi->ncol, 
                1.0, Bi->data, Bi->ld, 
                x_spos, 1, 1.0, y1[node0]->data, 1
            );
            CBLAS_GEMV(
                CblasRowMajor, CblasTrans, Bi->nrow, Bi->ncol, 
                1.0, Bi->data, Bi->ld, 
                y0[node0]->data, 1, 1.0, y_spos, 1
            );
        }
        
        // (4) node0 is a leaf node and its level is higher than node1's level, 
        //     only compressed on node1's side, node0's side don't need the 
        //     downward sweep and can directly accumulate result to output vector
        if (level0 < level1)
        {
            int s_index = cluster[node0 * 2];
            const DTYPE *x_spos = x + s_index;
            DTYPE *y_spos = y + s_index;
            CBLAS_GEMV(
                CblasRowMajor, CblasNoTrans, Bi->nrow, Bi->ncol, 
                1.0, Bi->data, Bi->ld, 
                y0[node1]->data, 1, 1.0, y_spos, 1
            );
            CBLAS_GEMV(
                CblasRowMajor, CblasTrans, Bi->nrow, Bi->ncol, 
                1.0, Bi->data, Bi->ld, 
                x_spos, 1, 1.0, y1[node1]->data, 1
            );
        }
    }
    
    // 3. Initialize empty y1
    for (int node = 0; node < n_node; node++)
    {
        if (y1[node] == NULL)
        {
            H2P_dense_mat_init(&y1[node], 1, 1);
            y1[node]->nrow = 0;
            y1[node]->ncol = 0;
            y1[node]->ld   = 0;
        }
    }
}

// H2 representation matvec downward sweep, calculate U_i * (B_{ij} * (U_j^T * x_j))
void H2P_matvec_downward_sweep(H2Pack_t h2pack, const DTYPE *x, DTYPE *y)
{
    int max_child       = h2pack->max_child;
    int n_leaf_node     = h2pack->n_leaf_node;
    int max_level       = h2pack->max_level;
    int min_adm_level   = h2pack->min_adm_level;
    int *children       = h2pack->children;
    int *n_child        = h2pack->n_child;
    int *level_n_node   = h2pack->level_n_node;
    int *level_nodes    = h2pack->level_nodes;
    int *cluster        = h2pack->cluster;
    H2P_dense_mat_t *U  = h2pack->U;
    H2P_dense_mat_t *y1 = h2pack->y1;
    
    H2P_dense_mat_t y1_tmp;
    H2P_dense_mat_init(&y1_tmp, h2pack->n_point, 1);
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        for (int j = 0; j < level_n_node[i]; j++)
        {
            int node = level_i_nodes[j];
            int n_child_node = n_child[node];
            int *child_nodes = children + node * max_child;
            
            if (y1[node]->nrow == 0) continue;
            
            CBLAS_GEMV(
                CblasRowMajor, CblasNoTrans, U[node]->nrow, U[node]->ncol,
                1.0, U[node]->data, U[node]->ld, 
                y1[node]->data, 1, 0.0, y1_tmp->data, 1
            );
            
            if (n_child_node == 0)
            {
                // Leaf node, accumulate final results to output vector
                int s_index = cluster[2 * node];
                int e_index = cluster[2 * node + 1];
                int n_point = e_index - s_index + 1;
                DTYPE *y_spos = y + s_index;
                #pragma omp simd
                for (int k = 0; k < n_point; k++)
                    y_spos[k] += y1_tmp->data[k];
            } else {
                // Non-leaf node, push down y1 values
                int y1_tmp_idx = 0;
                for (int k = 0; k < n_child_node; k++)
                {
                    int child_k = child_nodes[k];
                    int child_k_len = U[child_k]->ncol;
                    DTYPE *y1_tmp_spos = y1_tmp->data + y1_tmp_idx;
                    if (y1[child_k]->nrow == 0)
                    {
                        H2P_dense_mat_resize(y1[child_k], child_k_len, 1);
                        memcpy(y1[child_k]->data, y1_tmp_spos, sizeof(DTYPE) * child_k_len);
                    } else {
                        #pragma omp simd
                        for (int l = 0; l < child_k_len; l++)
                            y1[child_k]->data[l] += y1_tmp_spos[l];
                    }
                    y1_tmp_idx += child_k_len;
                }
            }
        }
    }
    H2P_dense_mat_destroy(y1_tmp);
}

// H2 representation matvec dense blocks matvec, calculate D_i * x_i
void H2P_matvec_dense_blocks(H2Pack_t h2pack, const DTYPE *x, DTYPE *y)
{
    int n_leaf_node    = h2pack->n_leaf_node;
    int n_r_inadm_pair = h2pack->n_r_inadm_pair;
    int *r_inadm_pairs = h2pack->r_inadm_pairs;
    int *leaf_nodes    = h2pack->height_nodes;
    int *cluster       = h2pack->cluster;
    H2P_dense_mat_t *D = h2pack->D;
    
    // 1. Diagonal blocks matvec
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int s_index = cluster[node * 2];
        const DTYPE *x_spos = x + s_index;
        DTYPE *y_spos = y + s_index;
        H2P_dense_mat_t Di = D[i];
        CBLAS_GEMV(
            CblasRowMajor, CblasNoTrans, Di->nrow, Di->ncol,
            1.0, Di->data, Di->ld, 
            x_spos, 1, 1.0, y_spos, 1
        );
    }
    
    // 2. Off-diagonal blocks from inadmissible pairs matvec
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int s_index0 = cluster[2 * node0];
        int s_index1 = cluster[2 * node1];
        const DTYPE *x_spos0 = x + s_index0;
        const DTYPE *x_spos1 = x + s_index1;
        DTYPE *y_spos0 = y + s_index0;
        DTYPE *y_spos1 = y + s_index1;
        H2P_dense_mat_t Di = D[n_leaf_node + i];
        CBLAS_GEMV(
            CblasRowMajor, CblasNoTrans, Di->nrow, Di->ncol,
            1.0, Di->data, Di->ld, 
            x_spos1, 1, 1.0, y_spos0, 1
        );
        CBLAS_GEMV(
            CblasRowMajor, CblasTrans, Di->nrow, Di->ncol,
            1.0, Di->data, Di->ld, 
            x_spos0, 1, 1.0, y_spos1, 1
        );
    }
}

// H2 representation multiplies a column vector
void H2P_matvec(H2Pack_t h2pack, const DTYPE *x, DTYPE *y)
{
    double st, et;
    memset(y, 0, sizeof(DTYPE) * h2pack->n_point);
    
    // 1. Upward sweep, calculate U_j^T * x_j
    st = H2P_get_wtime_sec();
    H2P_matvec_upward_sweep(h2pack, x, y);
    et = H2P_get_wtime_sec();
    h2pack->timers[4] += et - st;
    
    // 2. Intermediate sweep, calculate B_{ij} * (U_j^T * x_j)
    st = H2P_get_wtime_sec();
    H2P_matvec_intermediate_sweep(h2pack, x, y);
    et = H2P_get_wtime_sec();
    h2pack->timers[5] += et - st;
    
    // 3. Downward sweep, calculate U_i * (B_{ij} * (U_j^T * x_j))
    st = H2P_get_wtime_sec();
    H2P_matvec_downward_sweep(h2pack, x, y);
    et = H2P_get_wtime_sec();
    h2pack->timers[6] += et - st;
    
    // 4. Dense blocks, calculate D_i * x_i
    st = H2P_get_wtime_sec();
    H2P_matvec_dense_blocks(h2pack, x, y);
    et = H2P_get_wtime_sec();
    h2pack->timers[7] += et - st;
    
    h2pack->n_matvec++;
}

