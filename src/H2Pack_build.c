#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

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
    if (res < 1e-20) res = 1.0;
    res = 1.0 / DSQRT(res);
    return res;
}

// Evaluate a kernel matrix
// Input parameters:
//   coord  : Point coordinates
//   dim    : Dimension of point coordinate
//   idx_x  : Indices of points in the x point set
//   idx_y  : Indices of points in the y point set
// Output parameter:
//   kernel_mat : Obtained kernel matrix, nx-by-ny
void H2P_eval_kernel_matrix(
    DTYPE *coord, const int dim, H2P_int_vec_t idx_x, 
    H2P_int_vec_t idx_y, H2P_dense_mat_t kernel_mat
)
{
    H2P_dense_mat_resize(kernel_mat, idx_x->length, idx_y->length);
    for (int i = 0; i < idx_x->length; i++)
    {
        DTYPE *kernel_mat_row = kernel_mat->data + i * kernel_mat->ld;
        const DTYPE *x_i = coord + idx_x->data[i] * dim;
        for (int j = 0; j < idx_y->length; j++)
        {
            const DTYPE *y_j = coord + idx_y->data[j] * dim;
            kernel_mat_row[j] = kernel_func(x_i, y_j, dim);
        }
    }
}

// Evaluate a kernel matrix
// Input parameters:
//   dim     : Dimension of point coordinate
//   x_coord : X point set coordinates, size nx-by-dim
//   y_coord : Y point set coordinates, size ny-by-dim
// Output parameter:
//   kernel_mat : Obtained kernel matrix, nx-by-ny
void H2P_eval_kernel_matrix_direct(
    const int dim, H2P_dense_mat_t x_coord, 
    H2P_dense_mat_t y_coord, H2P_dense_mat_t kernel_mat
)

{
    H2P_dense_mat_resize(kernel_mat, x_coord->nrow, y_coord->nrow);
    for (int i = 0; i < x_coord->nrow; i++)
    {
        DTYPE *kernel_mat_row = kernel_mat->data + i * kernel_mat->ld;
        const DTYPE *x_i = x_coord->data + i * dim;
        for (int j = 0; j < y_coord->nrow; j++)
        {
            const DTYPE *y_j = y_coord->data + j * dim;
            kernel_mat_row[j] = kernel_func(x_i, y_j, dim);
        }
    }
}

// Evaluate a kernel matrix
// Input parameters:
//   dim        : Dimension of point coordinate
//   coord      : X point coordinates (not all used)
//   idx_x      : Indices of points in the x point set, length nx
//   box_center : X point box center coordinate, for shifting x points
//   pp_coord   : Proxy point set coordinates, size ny-by-dim
// Output parameter:
//   kernel_mat : Obtained kernel matrix, nx-by-ny
void H2P_eval_kernel_matrix_UJ_proxy(
    const int dim, DTYPE *coord, H2P_int_vec_t idx_x, 
    H2P_dense_mat_t box_center, H2P_dense_mat_t pp_coord, H2P_dense_mat_t kernel_mat
)
{
    H2P_dense_mat_t shift_x_i;
    H2P_dense_mat_init(&shift_x_i, dim, 1);
    H2P_dense_mat_resize(kernel_mat, idx_x->length, pp_coord->nrow);
    for (int i = 0; i < idx_x->length; i++)
    {
        DTYPE *kernel_mat_row = kernel_mat->data + i * kernel_mat->ld;
        const DTYPE *x_i = coord + idx_x->data[i] * dim;
        for (int j = 0; j < dim; j++)
            shift_x_i->data[j] = x_i[j] - box_center->data[j];
        for (int j = 0; j < pp_coord->nrow; j++)
        {
            const DTYPE *y_j = pp_coord->data + j * dim;
            kernel_mat_row[j] = kernel_func(shift_x_i->data, y_j, dim);
        }
    }
}

// Build H2 projection matrices using the direct approach
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with H2 projection matrices
void H2P_build_UJ_direct(H2Pack_t h2pack)
{
    int   dim            = h2pack->dim;
    int   n_node         = h2pack->n_node;
    int   n_leaf_node    = h2pack->n_leaf_node;
    int   max_child      = h2pack->max_child;
    int   max_level      = h2pack->max_level;
    int   min_adm_level  = h2pack->min_adm_level;
    int   stop_type      = h2pack->QR_stop_type;
    int   *children      = h2pack->children;
    int   *n_child       = h2pack->n_child;
    int   *level_n_node  = h2pack->level_n_node;
    int   *level_nodes   = h2pack->level_nodes;
    int   *leaf_nodes    = h2pack->leaf_nodes;
    int   *cluster       = h2pack->cluster;
    int   *node_adm_list = h2pack->node_adm_list;
    int   *node_adm_cnt  = h2pack->node_adm_cnt;
    DTYPE *coord         = h2pack->coord;
    void  *stop_param;
    if (stop_type == QR_RANK) 
        stop_param = &h2pack->QR_stop_rank;
    if ((stop_type == QR_REL_NRM) || (stop_type == QR_ABS_NRM))
        stop_param = &h2pack->QR_stop_tol;
    
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
                
                H2P_eval_kernel_matrix(coord, dim, J[node], col_idx, A_block);
                H2P_ID_compress(A_block, stop_type, stop_param, &U[node], sub_idx);
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
        } else {
            h2pack->mat_size[0] += U[i]->nrow * U[i]->ncol;
        }
        if (h2pack->J[i] == NULL) H2P_int_vec_init(&J[i], 1);
    }
    H2P_int_vec_destroy(col_idx);
    H2P_int_vec_destroy(sub_idx);
    H2P_dense_mat_destroy(A_block);
}

// Check if a coordinate is in box [-L/2, L/2]^dim
// Input parameters:
//   dim   : Dimension of coordinate
//   coord : Coordinate
//   L     : Box size
// Output parameter:
//   <return> : If the coordinate is in the box
int point_in_box(const int dim, DTYPE *coord, DTYPE L)
{
    int res = 1;
    DTYPE semi_L = L * 0.5;
    for (int i = 0; i < dim; i++)
    {
        DTYPE coord_i = coord[i];
        if ((coord_i < -semi_L) || (coord_i > semi_L))
        {
            res = 0;
            break;
        }
    }
    return res;
}

// Build H2 generator matrices
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with proxy points for building UJ
void H2P_generate_proxy_point_ID(H2Pack_t h2pack)
{
    int   dim          = h2pack->dim;
    int   root         = h2pack->n_node - 1;
    int   max_level    = h2pack->max_level;
    int   n_leaf_node  = h2pack->n_leaf_node;
    int   *level_nodes = h2pack->level_nodes;
    DTYPE *enbox       = h2pack->enbox;
    DTYPE  max_L       = enbox[root * dim * 2 + dim];
    
    // 1. Initialize proxy point arrays
    h2pack->pp = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * (max_level + 1));
    assert(h2pack->pp != NULL);
    for (int i = 0; i < max_level; i++) h2pack->pp[i] = NULL;
    H2P_dense_mat_t *pp = h2pack->pp;
    
    // The numbers of Nx and Ny points are empirical values
    int Nx_size = 1500;
    int Ny_size = 10000;
    H2P_dense_mat_t tmpA, Nx_points, Ny_points, min_dist;
    H2P_int_vec_t skel_idx;
    H2P_dense_mat_init(&tmpA, Nx_size, Ny_size);
    H2P_dense_mat_init(&Nx_points, Nx_size, dim);
    H2P_dense_mat_init(&Ny_points, Ny_size, dim);
    H2P_dense_mat_init(&min_dist, Nx_size, 1);
    H2P_int_vec_init(&skel_idx, Nx_size);
    srand48(time(NULL));

    // 2. Construct proxy points on each level
    for (int level = 2; level <= max_level; level++)
    {
        int node = level_nodes[level * n_leaf_node];
        
        // (1) Decide box sizes: Nx points are in box1, Ny points are in box3
        //     but not in box2 (points in box2 are inadmissible to Nx points)
        DTYPE L1 = enbox[node * 2 * dim + dim];
        DTYPE L2 = (1.0 + 2.0 * ALPHA_H2) * L1;
        DTYPE semi_L1   = L1 * 0.5;
        DTYPE semi_L3_0 = max_L - L1;
        DTYPE semi_L3_1 = (1.0 + 8.0 * ALPHA_H2) * L1;
        DTYPE semi_L3   = MIN(semi_L3_0, semi_L3_1);
        DTYPE L3 = 2.0 * semi_L3;
        
        // (2) Generate Nx and Ny points
        H2P_dense_mat_resize(Nx_points, Nx_size, dim);
        for (int i = 0; i < Nx_size * dim; i++)
        {
            DTYPE val = drand48();
            Nx_points->data[i] = L1 * val - semi_L1;
        }
        H2P_dense_mat_resize(Ny_points, Ny_size, dim);
        for (int i = 0; i < Ny_size; i++)
        {
            DTYPE *Ny_i = Ny_points->data + i * dim;
            int flag = 1;
            while (flag == 1)
            {
                for (int j = 0; j < dim; j++)
                {
                    DTYPE val = drand48();
                    Ny_i[j] = L3 * val - semi_L3;
                }
                flag = point_in_box(dim, Ny_i, L2);
            }
        }
        H2P_eval_kernel_matrix_direct(dim, Nx_points, Ny_points, tmpA);

        // (3) Use ID to select skeleton points in Nx first, then use the
        //     skeleton Nx points to select skeleton Ny points
        DTYPE rel_tol = 1e-15;
        H2P_ID_compress(tmpA, QR_REL_NRM, &rel_tol, NULL, skel_idx);
        H2P_dense_mat_select_rows(tmpA, skel_idx);
        H2P_dense_mat_transpose(tmpA);
        H2P_ID_compress(tmpA, QR_REL_NRM, &rel_tol, NULL, skel_idx);
        H2P_dense_mat_select_rows(Ny_points, skel_idx);
        
        // (4) Make the skeleton Ny points dense and then use them as proxy points
        int ny = skel_idx->length;
        H2P_dense_mat_resize(min_dist, ny, 1);
        for (int i = 0; i < ny; i++) min_dist->data[i] = 1e20;
        for (int i = 0; i < ny; i++)
        {
            DTYPE *coord_i = Ny_points->data + i * dim;
            for (int j = 0; j < i; j++)
            {
                DTYPE *coord_j = Ny_points->data + j * dim;
                DTYPE dist_ij = 0.0;
                for (int k = 0; k < dim; k++)
                {
                    DTYPE diff = coord_i[k] - coord_j[k];
                    dist_ij += diff * diff;
                }
                dist_ij = DSQRT(dist_ij);
                min_dist->data[i] = MIN(min_dist->data[i], dist_ij);
                min_dist->data[j] = MIN(min_dist->data[j], dist_ij);
            }
        }
        H2P_dense_mat_init(&pp[level], 2 * ny, dim);
        H2P_dense_mat_t pp_level = pp[level];

        for (int i = 0; i < ny; i++)
        {
            DTYPE *coord_0 = pp_level->data + dim * (2 * i);
            DTYPE *coord_1 = pp_level->data + dim * (2 * i + 1);
            memcpy(coord_0, Ny_points->data + dim * i, sizeof(DTYPE) * dim);
            int flag = 1;
            while (flag == 1)
            {
                DTYPE radius = 0.0;
                for (int j = 0; j < dim; j++)
                {
                    coord_1[j] = drand48() - 0.5;
                    radius += coord_1[j] * coord_1[j];
                }
                DTYPE inv_radius_scale = 0.33 / DSQRT(radius);
                for (int j = 0; j < dim; j++) 
                {
                    coord_1[j] *= inv_radius_scale;
                    coord_1[j] *= min_dist->data[i];
                    coord_1[j] += coord_0[j];
                }
                if ((point_in_box(dim, coord_1, L2) == 0) &&
                    (point_in_box(dim, coord_1, L3) == 1))
                    flag = 0;
            }
        }
    }
    
    H2P_int_vec_destroy(skel_idx);
    H2P_dense_mat_destroy(tmpA);
    H2P_dense_mat_destroy(Nx_points);
    H2P_dense_mat_destroy(Ny_points);
    H2P_dense_mat_destroy(min_dist);
}

// Build H2 projection matrices using the direct approach
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with H2 projection matrices
void H2P_build_UJ_proxy(H2Pack_t h2pack)
{
    int   dim            = h2pack->dim;
    int   n_node         = h2pack->n_node;
    int   n_leaf_node    = h2pack->n_leaf_node;
    int   max_child      = h2pack->max_child;
    int   max_level      = h2pack->max_level;
    int   min_adm_level  = h2pack->min_adm_level;
    int   stop_type      = h2pack->QR_stop_type;
    int   *children      = h2pack->children;
    int   *n_child       = h2pack->n_child;
    int   *level_n_node  = h2pack->level_n_node;
    int   *level_nodes   = h2pack->level_nodes;
    int   *leaf_nodes    = h2pack->leaf_nodes;
    int   *cluster       = h2pack->cluster;
    int   *node_adm_list = h2pack->node_adm_list;
    int   *node_adm_cnt  = h2pack->node_adm_cnt;
    DTYPE *coord         = h2pack->coord;
    DTYPE *enbox         = h2pack->enbox;
    H2P_dense_mat_t *pp  = h2pack->pp;
    void  *stop_param;
    if (stop_type == QR_RANK) 
        stop_param = &h2pack->QR_stop_rank;
    if ((stop_type == QR_REL_NRM) || (stop_type == QR_ABS_NRM))
        stop_param = &h2pack->QR_stop_tol;
    
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
    H2P_int_vec_t   col_idx, sub_idx;
    H2P_dense_mat_t A_block, box_center;
    H2P_int_vec_init(&col_idx, 0);
    H2P_int_vec_init(&sub_idx, 0);
    H2P_dense_mat_init(&A_block, 64, 64);
    H2P_dense_mat_init(&box_center, dim, 1);
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
            DTYPE *node_box = enbox + node * 2 * dim;
            for (int k = 0; k < dim; k++)
                box_center->data[k] = node_box[k] + 0.5 * node_box[dim + k];
            H2P_eval_kernel_matrix_UJ_proxy(dim, coord, J[node], box_center, pp[i], A_block);
            H2P_ID_compress(A_block, stop_type, stop_param, &U[node], sub_idx);
            for (int k = 0; k < U[node]->ncol; k++)
                J[node]->data[k] = J[node]->data[sub_idx->data[k]];
            J[node]->length = U[node]->ncol;
        }
    }

    // 4. Initialize other not touched U J
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        if (h2pack->U[i] == NULL)
        {
            H2P_dense_mat_init(&U[i], 1, 1);
            U[i]->nrow = 0;
            U[i]->ncol = 0;
            U[i]->ld   = 0;
        } else {
            h2pack->mat_size[0] += U[i]->nrow * U[i]->ncol;
        }
        if (h2pack->J[i] == NULL) H2P_int_vec_init(&J[i], 1);
    }

    H2P_int_vec_destroy(col_idx);
    H2P_int_vec_destroy(sub_idx);
    H2P_dense_mat_destroy(A_block);
    H2P_dense_mat_destroy(box_center);
}


// Build H2 generator matrices
// Input parameter:
//   h2pack : H2Pack structure with H2 projection matrices
// Output parameter:
//   h2pack : H2Pack structure with H2 generator matrices
void H2P_build_B(H2Pack_t h2pack)
{
    int   dim          = h2pack->dim;
    int   n_r_adm_pair = h2pack->n_r_adm_pair;
    int   *r_adm_pairs = h2pack->r_adm_pairs;
    int   *node_level  = h2pack->node_level;
    int   *cluster     = h2pack->cluster;
    DTYPE *coord       = h2pack->coord;

    h2pack->n_B = n_r_adm_pair;
    h2pack->B = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * h2pack->n_B);
    assert(h2pack->B != NULL);
    int B_idx = 0;
    H2P_dense_mat_t *B = h2pack->B;
    H2P_int_vec_t   *J = h2pack->J;
    H2P_int_vec_t   idx;
    H2P_int_vec_init(&idx, 0);
    
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int node0  = r_adm_pairs[2 * i];
        int node1  = r_adm_pairs[2 * i + 1];
        int level0 = node_level[node0];
        int level1 = node_level[node1];

        // (1) Two nodes are of the same level, compress on both sides
        if (level0 == level1)
        {
            int n_point0 = J[node0]->length;
            int n_point1 = J[node1]->length;
            H2P_dense_mat_init(&B[B_idx], n_point0, n_point1);
            H2P_eval_kernel_matrix(coord, dim, J[node0], J[node1], B[B_idx]);
            B_idx++;
            h2pack->mat_size[1] += n_point0 * n_point1;
        }
        
        // (2) node1 is a leaf node and its level is higher than node0's level, 
        //     only compress on node0's side
        if (level0 > level1)
        {
            int n_point0 = J[node0]->length;
            int s_index1 = cluster[2 * node1];
            int e_index1 = cluster[2 * node1 + 1];
            int n_point1 = e_index1 - s_index1 + 1;
            H2P_int_vec_set_capacity(idx, n_point1);
            idx->length = n_point1;
            for (int j = 0; j < n_point1; j++)
                idx->data[j] = s_index1 + j;
            H2P_dense_mat_init(&B[B_idx], n_point0, n_point1);
            H2P_eval_kernel_matrix(coord, dim, J[node0], idx, B[B_idx]);
            B_idx++;
            h2pack->mat_size[1] += n_point0 * n_point1;
        }
        
        // (3) node0 is a leaf node and its level is higher than node1's level, 
        //     only compress on node1's side
        if (level0 < level1)
        {
            int s_index0 = cluster[2 * node0];
            int e_index0 = cluster[2 * node0 + 1];
            int n_point0 = e_index0 - s_index0 + 1;
            int n_point1 = J[node1]->length;
            H2P_int_vec_set_capacity(idx, n_point0);
            idx->length = n_point0;
            for (int j = 0; j < n_point0; j++)
                idx->data[j] = s_index0 + j;
            H2P_dense_mat_init(&B[B_idx], n_point0, n_point1);
            H2P_eval_kernel_matrix(coord, dim, idx, J[node1], B[B_idx]);
            B_idx++;
            h2pack->mat_size[1] += n_point0 * n_point1;
        }
    }
    
    H2P_int_vec_destroy(idx);
}

// Build dense blocks in the original matrices
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with dense blocks
void H2P_build_D(H2Pack_t h2pack)
{
    int   dim            = h2pack->dim;
    int   n_leaf_node    = h2pack->n_leaf_node;
    int   n_r_inadm_pair = h2pack->n_r_inadm_pair;
    int   *leaf_nodes    = h2pack->leaf_nodes;
    int   *cluster       = h2pack->cluster;
    int   *r_inadm_pairs = h2pack->r_inadm_pairs;
    DTYPE *coord         = h2pack->coord;
    
    // 1. Allocate D
    h2pack->n_D = n_leaf_node + n_r_inadm_pair;
    h2pack->D = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * h2pack->n_D);
    assert(h2pack->D != NULL);
    int D_idx = 0;
    H2P_dense_mat_t *D = h2pack->D;
    H2P_int_vec_t idx0, idx1;
    H2P_int_vec_init(&idx0, 0);
    H2P_int_vec_init(&idx1, 0);
    
    // 2. Generate diagonal blocks (leaf node self interaction)
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int s_index = cluster[2 * node];
        int e_index = cluster[2 * node + 1];
        int n_point = e_index - s_index + 1;
        H2P_int_vec_set_capacity(idx0, n_point);
        idx0->length = n_point;
        for (int j = 0; j < n_point; j++)
            idx0->data[j] = s_index + j;
        H2P_dense_mat_init(&D[D_idx], n_point, n_point);
        H2P_eval_kernel_matrix(coord, dim, idx0, idx0, D[D_idx]);
        D_idx++;
        h2pack->mat_size[2] += n_point * n_point;
    }
    
    // 3. Generate off-diagonal blocks from inadmissible pairs
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int s_index0 = cluster[2 * node0];
        int s_index1 = cluster[2 * node1];
        int e_index0 = cluster[2 * node0 + 1];
        int e_index1 = cluster[2 * node1 + 1];
        int n_point0 = e_index0 - s_index0 + 1;
        int n_point1 = e_index1 - s_index1 + 1;
        H2P_int_vec_set_capacity(idx0, n_point0);
        H2P_int_vec_set_capacity(idx1, n_point1);
        idx0->length = n_point0;
        idx1->length = n_point1;
        for (int j = 0; j < n_point0; j++)
            idx0->data[j] = s_index0 + j;
        for (int j = 0; j < n_point1; j++)
            idx1->data[j] = s_index1 + j;
        H2P_dense_mat_init(&D[D_idx], n_point0, n_point1);
        H2P_eval_kernel_matrix(coord, dim, idx0, idx1, D[D_idx]);
        D_idx++;
        h2pack->mat_size[2] += n_point0 * n_point1;
    }
    
    H2P_int_vec_destroy(idx1);
    H2P_int_vec_destroy(idx0);
}

// Build H2 representation with a kernel function
void H2P_build(H2Pack_t h2pack)
{
    double st, et;

    // 1. Generate proxy points for building U and J
    st = H2P_get_wtime_sec();
    H2P_generate_proxy_point_ID(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[1] = et - st;

    // 2. Build projection matrices and skeleton row sets
    st = H2P_get_wtime_sec();
    H2P_build_UJ_proxy(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[2] = et - st;

    // 3. Build generator matrices
    st = H2P_get_wtime_sec();
    H2P_build_B(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[3] = et - st;
    
    // 4. Build dense blocks
    st = H2P_get_wtime_sec();
    H2P_build_D(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[4] = et - st;
}

