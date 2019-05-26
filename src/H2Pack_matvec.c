#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

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
void H2P_matvec_upward_sweep(H2Pack_t h2pack, const DTYPE *x)
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
        int height_i_n_node = height_n_node[i];
        int nthreads = MIN(height_i_n_node, h2pack->n_thread);
        
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (int j = 0; j < height_i_n_node; j++)
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
void H2P_matvec_intermediate_sweep(H2Pack_t h2pack, const DTYPE *x)
{
    int n_node        = h2pack->n_node;
    int n_thread      = h2pack->n_thread;
    int n_r_adm_pair  = h2pack->n_r_adm_pair;
    int *r_adm_pairs  = h2pack->r_adm_pairs;
    int *node_level   = h2pack->node_level;
    int *cluster      = h2pack->cluster;
    int *node_n_r_adm = h2pack->node_n_r_adm;
    int *B_nrow       = h2pack->B_nrow;
    int *B_ncol       = h2pack->B_ncol;
    int *B_ptr        = h2pack->B_ptr;
    DTYPE *B_data     = h2pack->B_data;
    H2P_int_vec_t B_blk = h2pack->B_blk;
    H2P_dense_mat_t *y0 = h2pack->y0;
    H2P_dense_mat_t *U  = h2pack->U;

    // 1. Initialize y1 
    if (h2pack->y1 == NULL)
    {
        h2pack->y1 = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
        assert(h2pack->y1 != NULL);
        for (int i = 0; i < n_node; i++) h2pack->y1[i] = NULL;
    }
    H2P_dense_mat_t *y1 = h2pack->y1;

    // Use ld to mark if y1[i] is visited in this intermediate sweep
    for (int i = 0; i < n_node; i++) 
    {
        if (y1[i] != NULL) y1[i]->ld = 0;
        if (node_n_r_adm[i])
        {
            // The first U[node{0, 1}]->ncol elements in y1[node{0, 1}] will be used in downward
            // sweep, store the final results in this part and use the positions behind this as
            // thread buffers. The last position of each row is used to mark if this row has data.
            if (y1[i] == NULL) H2P_dense_mat_init(&y1[i], n_thread, U[i]->ncol + 1);
            y1[i]->ld = 1;
        }
    }

    // 2. Intermediate sweep
    const int n_B_blk = B_blk->length;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        DTYPE *y = h2pack->tb[tid]->y;
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n_node; i++)
        {
            if (y1[i] == NULL) continue;
            if (y1[i]->ld == 0) continue;
            const int ncol = y1[i]->ncol;
            // Need not to reset all copies of y1 to be 0 here, use the last element in
            // each row as the beta value to rewrite / accumulate y1 results in GEMV
            memset(y1[i]->data, 0, sizeof(DTYPE) * ncol);
            for (int j = 1; j < n_thread; j++)
                y1[i]->data[(j + 1) * ncol - 1] = 0.0;
        }
        
        #pragma omp for schedule(dynamic)
        for (int i_blk = 0; i_blk < n_B_blk; i_blk++)
        {
            int s_index = B_blk->data[i_blk];
            int e_index = B_blk->data[i_blk + 1];
            for (int i = s_index; i < e_index; i++)
            {
                int node0  = r_adm_pairs[2 * i];
                int node1  = r_adm_pairs[2 * i + 1];
                int level0 = node_level[node0];
                int level1 = node_level[node1];
                
                DTYPE *Bi = B_data + B_ptr[i];
                int Bi_nrow = B_nrow[i];
                int Bi_ncol = B_ncol[i];
                
                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    int ncol0 = y1[node0]->ncol;
                    int ncol1 = y1[node1]->ncol;
                    DTYPE *y1_dst_0 = y1[node0]->data + tid * ncol0;
                    DTYPE *y1_dst_1 = y1[node1]->data + tid * ncol1;
                    DTYPE beta0 = y1_dst_0[ncol0 - 1];
                    DTYPE beta1 = y1_dst_1[ncol1 - 1];
                    y1_dst_0[ncol0 - 1] = 1.0;
                    y1_dst_1[ncol1 - 1] = 1.0;
                    CBLAS_GEMV(
                        CblasRowMajor, CblasNoTrans, Bi_nrow, Bi_ncol, 
                        1.0, Bi, Bi_ncol, 
                        y0[node1]->data, 1, beta0, y1_dst_0, 1
                    );
                    CBLAS_GEMV(
                        CblasRowMajor, CblasTrans, Bi_nrow, Bi_ncol, 
                        1.0, Bi, Bi_ncol, 
                        y0[node0]->data, 1, beta1, y1_dst_1, 1
                    );
                }
                
                // (2) node1 is a leaf node and its level is higher than node0's level, 
                //     only compressed on node0's side, node1's side don't need the 
                //     downward sweep and can directly accumulate result to output vector
                if (level0 > level1)
                {
                    int s_index = cluster[node1 * 2];
                    const DTYPE *x_spos = x + s_index;
                    DTYPE *y_spos = y + s_index;
                    int ncol0 = y1[node0]->ncol;
                    DTYPE *y1_dst_0 = y1[node0]->data + tid * ncol0;
                    DTYPE beta0 = y1_dst_0[ncol0 - 1];
                    y1_dst_0[ncol0 - 1] = 1.0;
                    CBLAS_GEMV(
                        CblasRowMajor, CblasNoTrans, Bi_nrow, Bi_ncol, 
                        1.0, Bi, Bi_ncol, 
                        x_spos, 1, beta0, y1_dst_0, 1
                    );
                    CBLAS_GEMV(
                        CblasRowMajor, CblasTrans, Bi_nrow, Bi_ncol, 
                        1.0, Bi, Bi_ncol, 
                        y0[node0]->data, 1, 1.0, y_spos, 1
                    );
                }
                
                // (3) node0 is a leaf node and its level is higher than node1's level, 
                //     only compressed on node1's side, node0's side don't need the 
                //     downward sweep and can directly accumulate result to output vector
                if (level0 < level1)
                {
                    int s_index = cluster[node0 * 2];
                    const DTYPE *x_spos = x + s_index;
                    DTYPE *y_spos = y + s_index;
                    int ncol1 = y1[node1]->ncol;
                    DTYPE *y1_dst_1 = y1[node1]->data + tid * ncol1;
                    DTYPE beta1 = y1_dst_1[ncol1 - 1];
                    y1_dst_1[ncol1 - 1] = 1.0;
                    CBLAS_GEMV(
                        CblasRowMajor, CblasNoTrans, Bi_nrow, Bi_ncol, 
                        1.0, Bi, Bi_ncol, 
                        y0[node1]->data, 1, 1.0, y_spos, 1
                    );
                    CBLAS_GEMV(
                        CblasRowMajor, CblasTrans, Bi_nrow, Bi_ncol, 
                        1.0, Bi, Bi_ncol, 
                        x_spos, 1, beta1, y1_dst_1, 1
                    );
                }
            }
        }
    }
    
    // 3. Sum thread-local buffers
    #pragma omp parallel for num_threads(n_thread) schedule(dynamic)
    for (int i = 0; i < n_node; i++)
    {
        if (y1[i] == NULL) continue;
        if (y1[i]->ld == 0) continue;
        
        int ncol = y1[i]->ncol;
        DTYPE *dst_row = y1[i]->data;
        for (int j = 1; j < n_thread; j++)
        {
            DTYPE *src_row = y1[i]->data + j * ncol;
            if (src_row[ncol - 1] != 1.0) continue;
            #pragma omp simd
            for (int k = 0; k < ncol - 1; k++)
                dst_row[k] += src_row[k];
        }
    }
    
    // 4. Initialize empty y1
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
void H2P_matvec_downward_sweep(H2Pack_t h2pack, const DTYPE *x)
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
    
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int nthreads = MIN(level_i_n_node, h2pack->n_thread);
        
        #pragma omp parallel num_threads(nthreads) 
        {
            int tid = omp_get_thread_num();
            DTYPE *y = h2pack->tb[tid]->y;
            H2P_dense_mat_t y1_tmp = h2pack->tb[tid]->mat0;
            
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                int *child_nodes = children + node * max_child;
                
                if (y1[node]->ld == 0) continue;
                
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
                        if (y1[child_k]->ld == 0)
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
    }
}

// H2 representation matvec dense blocks matvec, calculate D_i * x_i
void H2P_matvec_dense_blocks(H2Pack_t h2pack, const DTYPE *x)
{
    int n_leaf_node    = h2pack->n_leaf_node;
    int n_r_inadm_pair = h2pack->n_r_inadm_pair;
    int *r_inadm_pairs = h2pack->r_inadm_pairs;
    int *leaf_nodes    = h2pack->height_nodes;
    int *cluster       = h2pack->cluster;
    //H2P_dense_mat_t *D = h2pack->D;
    int *D_nrow  = h2pack->D_nrow;
    int *D_ncol  = h2pack->D_ncol;
    int *D_ptr   = h2pack->D_ptr;
    DTYPE *D_data = h2pack->D_data;
    
    H2P_int_vec_t D_blk0 = h2pack->D_blk0;
    H2P_int_vec_t D_blk1 = h2pack->D_blk1;
    
    const int n_D0_blk = D_blk0->length;
    const int n_D1_blk = D_blk1->length;
    #pragma omp parallel num_threads(h2pack->n_thread)
    {
        int tid = omp_get_thread_num();
        DTYPE *y = h2pack->tb[tid]->y;
        
        // 1. Diagonal blocks matvec
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk0 = 0; i_blk0 < n_D0_blk; i_blk0++)
        {
            int s_index = D_blk0->data[i_blk0];
            int e_index = D_blk0->data[i_blk0 + 1];
            for (int i = s_index; i < e_index; i++)
            {
                int node = leaf_nodes[i];
                int s_index = cluster[node * 2];
                const DTYPE *x_spos = x + s_index;
                DTYPE *y_spos = y + s_index;
                DTYPE *Di = D_data + D_ptr[i];
                int Di_nrow = D_nrow[i];
                int Di_ncol = D_ncol[i];
                CBLAS_GEMV(
                    CblasRowMajor, CblasNoTrans, Di_nrow, Di_ncol,
                    1.0, Di, Di_ncol, 
                    x_spos, 1, 1.0, y_spos, 1
                );
            }
        }
        
        // 2. Off-diagonal blocks from inadmissible pairs matvec
        #pragma omp for schedule(dynamic) 
        for (int i_blk1 = 0; i_blk1 < n_D1_blk; i_blk1++)
        {
            int s_index = D_blk1->data[i_blk1];
            int e_index = D_blk1->data[i_blk1 + 1];
            for (int i = s_index; i < e_index; i++)
            {
                int node0 = r_inadm_pairs[2 * i];
                int node1 = r_inadm_pairs[2 * i + 1];
                int s_index0 = cluster[2 * node0];
                int s_index1 = cluster[2 * node1];
                const DTYPE *x_spos0 = x + s_index0;
                const DTYPE *x_spos1 = x + s_index1;
                DTYPE *y_spos0 = y + s_index0;
                DTYPE *y_spos1 = y + s_index1;
                DTYPE *Di = D_data + D_ptr[n_leaf_node + i];
                int Di_nrow = D_nrow[n_leaf_node + i];
                int Di_ncol = D_ncol[n_leaf_node + i];
                CBLAS_GEMV(
                    CblasRowMajor, CblasNoTrans, Di_nrow, Di_ncol,
                    1.0, Di, Di_ncol, 
                    x_spos1, 1, 1.0, y_spos0, 1
                );
                CBLAS_GEMV(
                    CblasRowMajor, CblasTrans, Di_nrow, Di_ncol,
                    1.0, Di, Di_ncol, 
                    x_spos0, 1, 1.0, y_spos1, 1
                );
            }
        }
    }
}

// H2 representation multiplies a column vector
void H2P_matvec(H2Pack_t h2pack, const DTYPE *x, DTYPE *y)
{
    double st, et;
    int n_point  = h2pack->n_point;
    int n_thread = h2pack->n_thread;
    
    // 1. Reset partial y result in each thread-local buffer to 0
    st = H2P_get_wtime_sec();
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        DTYPE *tid_y = h2pack->tb[tid]->y;
        memset(tid_y, 0, sizeof(DTYPE) * n_point);
    }
    et = H2P_get_wtime_sec();
    h2pack->timers[8] += et - st;
    
    // 2. Upward sweep, calculate U_j^T * x_j
    st = H2P_get_wtime_sec();
    H2P_matvec_upward_sweep(h2pack, x);
    et = H2P_get_wtime_sec();
    h2pack->timers[4] += et - st;
    
    // 3. Intermediate sweep, calculate B_{ij} * (U_j^T * x_j)
    st = H2P_get_wtime_sec();
    H2P_matvec_intermediate_sweep(h2pack, x);
    et = H2P_get_wtime_sec();
    h2pack->timers[5] += et - st;
    
    // 4. Downward sweep, calculate U_i * (B_{ij} * (U_j^T * x_j))
    st = H2P_get_wtime_sec();
    H2P_matvec_downward_sweep(h2pack, x);
    et = H2P_get_wtime_sec();
    h2pack->timers[6] += et - st;
    
    // 5. Dense blocks, calculate D_i * x_i
    st = H2P_get_wtime_sec();
    H2P_matvec_dense_blocks(h2pack, x);
    et = H2P_get_wtime_sec();
    h2pack->timers[7] += et - st;
    
    // 6. Reduce sum partial y results
    st = H2P_get_wtime_sec();
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        int rem = n_point % n_thread;
        int bs0 = n_point / n_thread;
        int bs1 = bs0 + 1;
        int spos, epos;
        if (tid < rem)
        {
            spos = bs1 * tid;
            epos = spos + bs1;
        } else {
            spos = tid * bs0 + rem;
            epos = spos + bs0;
        }
        
        DTYPE *y_src = h2pack->tb[0]->y;
        memcpy(y + spos, y_src + spos, sizeof(DTYPE) * (epos - spos));
        for (int tid = 1; tid < n_thread; tid++)
        {
            y_src = h2pack->tb[tid]->y;
            #pragma omp simd
            for (int i = spos; i < epos; i++) y[i] += y_src[i];
        }
    }
    et = H2P_get_wtime_sec();
    h2pack->mat_size[7] = (n_thread + 1) * h2pack->n_point;
    h2pack->timers[8] += et - st;
    
    h2pack->n_matvec++;
}

