#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_ID_compress.h"
#include "H2Pack_build.h"
#include "DAG_task_queue.h"
#include "utils.h"

// Gather some columns from a matrix to another matrix
// Input parameters:
//   src_mat : Source matrix with required columns
//   src_ld  : Leading dimension of the source matrix, should >= max(col_idx(:))+1
//   dst_ld  : Leading dimension of the destination matrix, should >= ncol
//   nrow    : Number of rows in the source and destination matrices
//   col_idx : Indices of required columns
//   ncol    : Number of required columns
// Output parameter:
//   dst_mat : Destination matrix with required columns only
void H2P_gather_matrix_columns(
    DTYPE *src_mat, const int src_ld, DTYPE *dst_mat, const int dst_ld, 
    const int nrow, int *col_idx, const int ncol
)
{
    for (int irow = 0; irow < nrow; irow++)
    {
        DTYPE *src_row = src_mat + src_ld * irow;
        DTYPE *dst_row = dst_mat + dst_ld * irow;
        for (int icol = 0; icol < ncol; icol++)
            dst_row[icol] = src_row[col_idx[icol]];
    }
}

// Copy a block from ta matrix to another matrix
// Input parameters:
//   nrow    : Number of rows to be copied 
//   ncol    : Number of columns to be copied
//   src_mat : Source matrix 
//   src_ld  : Leading dimension of the source matrix
//   dst_ld  : Leading dimension of the destination matrix
// Output parameter:
//   dst_mat : Destination matrix
void H2P_copy_matrix_block(
    const int nrow, const int ncol, DTYPE *src_mat, const int src_ld, 
    DTYPE *dst_mat, const int dst_ld
)
{
    size_t row_msize = sizeof(DTYPE) * ncol;
    for (int irow = 0; irow < nrow; irow ++)
    {
        DTYPE *src_row = src_mat + irow * src_ld;
        DTYPE *dst_row = dst_mat + irow * dst_ld;
        memcpy(dst_row, src_row, row_msize);
    }
}

// Evaluate a kernel matrix with OpenMP parallelization
// Input parameters:
//   krnl_param : Pointer to kernel function parameter array
//   krnl_eval  : Kernel matrix evaluation function
//   krnl_dim   : Dimension of tensor kernel's return
//   x_coord    : X point set coordinates, size nx-by-pt_dim
//   y_coord    : Y point set coordinates, size ny-by-pt_dim
// Output parameter:
//   kernel_mat : Obtained kernel matrix, nx-by-ny
void H2P_eval_kernel_matrix_OMP(
    const void *krnl_param, kernel_eval_fptr krnl_eval, const int krnl_dim, 
    H2P_dense_mat_t x_coord, H2P_dense_mat_t y_coord, H2P_dense_mat_t kernel_mat
)
{
    const int nx = x_coord->ncol;
    const int ny = y_coord->ncol;
    const int nrow = nx * krnl_dim;
    const int ncol = ny * krnl_dim;
    H2P_dense_mat_resize(kernel_mat, nrow, ncol);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        int nx_blk_start, nx_blk_len;
        calc_block_spos_len(nx, nt, tid, &nx_blk_start, &nx_blk_len);
        
        DTYPE *kernel_mat_srow = kernel_mat->data + nx_blk_start * krnl_dim * kernel_mat->ld;
        DTYPE *x_coord_spos = x_coord->data + nx_blk_start;
        
        krnl_eval(
            x_coord_spos,  x_coord->ncol, nx_blk_len, 
            y_coord->data, y_coord->ncol, ny, 
            krnl_param, kernel_mat_srow, kernel_mat->ld
        );
    }
}

// Check if a coordinate is in box [-L/2, L/2]^pt_dim
// Input parameters:
//   pt_dim : Dimension of point coordinate
//   coord  : Coordinate
//   L      : Box size
// Output parameter:
//   <return> : If the coordinate is in the box
int point_in_box(const int pt_dim, DTYPE *coord, DTYPE L)
{
    int res = 1;
    DTYPE semi_L = L * 0.5;
    for (int i = 0; i < pt_dim; i++)
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

// Generate proxy points for constructing H2 projection and skeleton matrices
// using ID compress for any kernel function
void H2P_generate_proxy_point_ID(
    const int pt_dim, const int krnl_dim, const DTYPE tol_norm, const int max_level, const int start_level,
    DTYPE max_L, const void *krnl_param, kernel_eval_fptr krnl_eval, H2P_dense_mat_t **pp_
)
{   
    // 1. Initialize proxy point arrays
    H2P_dense_mat_t *pp = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * (max_level + 1));
    assert(pp != NULL);
    for (int i = 0; i <= max_level; i++) pp[i] = NULL;
    
    // 2. Initialize temporary arrays. The numbers of Nx and Ny points are empirical values
    int Nx_size = 1000;
    int Ny_size = 15000;
    H2P_dense_mat_t tmpA, Nx_points, Ny_points, min_dist;
    H2P_int_vec_t skel_idx;
    H2P_dense_mat_init(&Nx_points, pt_dim,  Nx_size);
    H2P_dense_mat_init(&Ny_points, pt_dim,  Ny_size);
    H2P_dense_mat_init(&tmpA,      Nx_size, Ny_size);
    H2P_dense_mat_init(&min_dist,  Nx_size, 1);
    H2P_int_vec_init(&skel_idx, Nx_size);
    srand48(time(NULL));
    int n_thread = omp_get_max_threads();

    // 3. Construct proxy points on each level
    DTYPE pow_2_level = 0.5;
    H2P_dense_mat_t QR_buff;
    H2P_int_vec_t   ID_buff;
    H2P_dense_mat_init(&QR_buff, 2 * Ny_size, 1);
    H2P_int_vec_init(&ID_buff, 4 * Ny_size);
    for (int level = 0; level < start_level; level++) pow_2_level *= 2.0;
    for (int level = start_level; level <= max_level; level++)
    {
        if (level < 2)
        {
            H2P_dense_mat_init(&pp[level], pt_dim, 1);
            pp[level]->ncol = 0;
            pow_2_level *= 2.0;
            continue;
        }

        // (1) Decide box sizes: Nx points are in box1, Ny points are in box3
        //     but not in box2 (points in box2 are inadmissible to Nx points)
        pow_2_level *= 2.0;
        DTYPE L1 = max_L / pow_2_level;
        DTYPE L2 = (1.0 + 2.0 * ALPHA_H2) * L1;
        DTYPE semi_L1   = L1 * 0.5;
        DTYPE semi_L3_0 = max_L - L1;
        DTYPE semi_L3_1 = (1.0 + 8.0 * ALPHA_H2) * L1;
        DTYPE semi_L3   = MIN(semi_L3_0, semi_L3_1);
        DTYPE L3 = 2.0 * semi_L3;
        
        // (2) Generate Nx and Ny points
        H2P_dense_mat_resize(Nx_points, pt_dim, Nx_size);
        for (int i = 0; i < Nx_size * pt_dim; i++)
        {
            DTYPE val = drand48();
            Nx_points->data[i] = L1 * val - semi_L1;
        }
        H2P_dense_mat_resize(Ny_points, pt_dim, Ny_size);
        for (int i = 0; i < Ny_size; i++)
        {
            DTYPE *tmp_coord = tmpA->data;
            int flag = 1;
            while (flag == 1)
            {
                for (int j = 0; j < pt_dim; j++)
                {
                    DTYPE val = drand48();
                    tmp_coord[j] = L3 * val - semi_L3;
                }
                flag = point_in_box(pt_dim, tmp_coord, L2);
            }
            DTYPE *Ny_i = Ny_points->data + i;
            for (int j = 0; j < pt_dim; j++)
                Ny_i[j * Ny_size] = tmp_coord[j];
        }

        // (3) Use ID to select skeleton points in Nx first, then use the
        //     skeleton Nx points to select skeleton Ny points

        DTYPE rel_tol = tol_norm < 1e-11 ? 1e-14 : tol_norm * 1e-2;
        
        H2P_eval_kernel_matrix_OMP(krnl_param, krnl_eval, krnl_dim, Nx_points, Ny_points, tmpA);
        if (krnl_dim == 1)
        {
            H2P_dense_mat_resize(QR_buff, tmpA->nrow, 1);
        } else {
            int QR_buff_size = (2 * krnl_dim + 2) * tmpA->ncol + (krnl_dim + 1) * tmpA->nrow;
            H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
        }
        H2P_int_vec_set_capacity(ID_buff, 4 * tmpA->nrow);
        H2P_ID_compress(
            tmpA, QR_REL_NRM, &rel_tol, NULL, skel_idx, 
            n_thread, QR_buff->data, ID_buff->data, krnl_dim
        );
        H2P_dense_mat_select_columns(Nx_points, skel_idx);
        
        H2P_eval_kernel_matrix_OMP(krnl_param, krnl_eval, krnl_dim, Ny_points, Nx_points, tmpA);
        if (krnl_dim == 1)
        {
            H2P_dense_mat_resize(QR_buff, tmpA->nrow, 1);
        } else {
            int QR_buff_size = (2 * krnl_dim + 2) * tmpA->ncol + (krnl_dim + 1) * tmpA->nrow;
            H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
        }
        H2P_int_vec_set_capacity(ID_buff, 4 * tmpA->nrow);
        H2P_ID_compress(
            tmpA, QR_REL_NRM, &rel_tol, NULL, skel_idx, 
            n_thread, QR_buff->data, ID_buff->data, krnl_dim
        );
        H2P_dense_mat_select_columns(Ny_points, skel_idx);
        
        // (4) Set up the proxy points
        int ny = skel_idx->length;
        if (tol_norm > 1e-11)
        {
            //  Case 1: when tol_norm is LARGE, directly use the Ny_points as the proxy points. 
            const int Ny_size2 = ny;
            H2P_dense_mat_init(&pp[level], pt_dim, Ny_size2);
            H2P_dense_mat_t pp_level = pp[level];
            // Also transpose the coordinate array for vectorizing kernel evaluation here
            for (int i = 0; i < ny; i++)
            {
                DTYPE *tmp_coord0 = tmpA->data;
                DTYPE *Ny_point_i = Ny_points->data + i;
                for (int j = 0; j < pt_dim; j++)
                    tmp_coord0[j] = Ny_point_i[j * Ny_points->ncol];
                DTYPE *coord_0 = pp_level->data + i;
                for (int j = 0; j < pt_dim; j++)
                    coord_0[j * Ny_size2] = tmp_coord0[j];
            }
        } else {
            //  Case 2: when tol_norm is SMALL, densify the Ny_points as the proxy points. 
            H2P_dense_mat_resize(min_dist, ny, 1);
            DTYPE *coord_i = tmpA->data;
            for (int i = 0; i < ny; i++) min_dist->data[i] = 1e20;
            for (int i = 0; i < ny; i++)
            {
                for (int k = 0; k < pt_dim; k++)
                    coord_i[k] = Ny_points->data[i + k * Ny_points->ncol];
                
                for (int j = 0; j < i; j++)
                {
                    DTYPE dist_ij = 0.0;
                    for (int k = 0; k < pt_dim; k++)
                    {
                        DTYPE diff = coord_i[k] - Ny_points->data[j + k * Ny_points->ncol];
                        dist_ij += diff * diff;
                    }
                    dist_ij = DSQRT(dist_ij);
                    min_dist->data[i] = MIN(min_dist->data[i], dist_ij);
                    min_dist->data[j] = MIN(min_dist->data[j], dist_ij);
                }
            }

            const int Ny_size2 = ny * 2;
            H2P_dense_mat_init(&pp[level], pt_dim, Ny_size2);
            H2P_dense_mat_t pp_level = pp[level];
            for (int i = 0; i < ny; i++)
            {
                DTYPE *tmp_coord0 = tmpA->data;
                DTYPE *Ny_point_i = Ny_points->data + i;
                for (int j = 0; j < pt_dim; j++)
                    tmp_coord0[j] = Ny_point_i[j * Ny_points->ncol];
                DTYPE *tmp_coord1 = tmpA->data + pt_dim;
                DTYPE radius_i_scale = min_dist->data[i] * 0.33;
                int flag = 1;
                while (flag == 1)
                {
                    DTYPE radius_1 = 0.0;
                    for (int j = 0; j < pt_dim; j++)
                    {
                        tmp_coord1[j] = drand48() - 0.5;
                        radius_1 += tmp_coord1[j] * tmp_coord1[j];
                    }
                    DTYPE inv_radius_1 = 1.0 / DSQRT(radius_1);
                    for (int j = 0; j < pt_dim; j++) 
                    {
                        tmp_coord1[j] *= inv_radius_1;
                        tmp_coord1[j] *= radius_i_scale;
                        tmp_coord1[j] += tmp_coord0[j];
                    }
                    if ((point_in_box(pt_dim, tmp_coord1, L2) == 0) &&
                        (point_in_box(pt_dim, tmp_coord1, L3) == 1))
                        flag = 0;
                }  // End of "while (flag == 1)"
                DTYPE *coord_0 = pp_level->data + (2 * i);
                DTYPE *coord_1 = pp_level->data + (2 * i + 1);
                for (int j = 0; j < pt_dim; j++)
                {
                    coord_0[j * Ny_size2] = tmp_coord0[j];
                    coord_1[j * Ny_size2] = tmp_coord1[j];
                }
            }  // End of "for (int i = 0; i < ny; i++)"
        }  // End of "if (tol_norm > 1e-11)"
    }  // End of "for (int level = start_level; level <= max_level; level++)"
    
    *pp_ = pp;
    H2P_int_vec_destroy(skel_idx);
    H2P_int_vec_destroy(ID_buff);
    H2P_dense_mat_destroy(QR_buff);
    H2P_dense_mat_destroy(tmpA);
    H2P_dense_mat_destroy(Nx_points);
    H2P_dense_mat_destroy(Ny_points);
    H2P_dense_mat_destroy(min_dist);
}

// Generate uniformly distributed proxy points on a box surface for constructing
// H2 projection and skeleton matrices for SOME kernel function
void H2P_generate_proxy_point_surface(
    const int pt_dim, const int min_npts, const int max_level, 
    const int start_level, DTYPE max_L, H2P_dense_mat_t **pp_
)
{
    if (pt_dim < 2 || pt_dim > 3)
    {
        fprintf(stderr, "[ERROR] H2P_generate_proxy_point_surface() needs pt_dim = 2 or 3!\n");
        fprintf(stderr, "[ERROR] H2P_generate_proxy_point_surface() exits without doing anything.\n");
        return;
    }
    
    H2P_dense_mat_t *pp = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * (max_level + 1));
    assert(pp != NULL);
    for (int i = 0; i <= max_level; i++) pp[i] = NULL;
    
    int npts_axis, npts;
    if (pt_dim == 2)
    {
        npts_axis = (min_npts + 3) / 4;
        npts = npts_axis * 4;
    } else {
        double n_point_face = (double) min_npts / 6.0;
        npts_axis = (int) ceil(sqrt(n_point_face));
        npts = npts_axis * npts_axis * 6;
    }
    double h = 2.0 / (double) (npts_axis + 1);
    
    // Generate proxy points on the surface of [-1,1]^pt_dim box
    H2P_dense_mat_t unit_pp;
    H2P_dense_mat_init(&unit_pp, pt_dim, npts);
    int index = 0;
    if (pt_dim == 3)
    {
        DTYPE *x = unit_pp->data;
        DTYPE *y = unit_pp->data + npts;
        DTYPE *z = unit_pp->data + npts * 2;
        for (int i = 0; i < npts_axis; i++)
        {
            double h_i = h * (i + 1) - 1.0;
            for (int j = 0; j < npts_axis; j++)
            {
                double h_j = h * (j + 1) - 1.0;
                
                x[index + 0] = h_i;
                y[index + 0] = h_j;
                z[index + 0] = -1.0;
                
                x[index + 1] = h_i;
                y[index + 1] = h_j;
                z[index + 1] = 1.0;
                
                x[index + 2] = h_i;
                y[index + 2] = -1.0;
                z[index + 2] = h_j;
                
                x[index + 3] = h_i;
                y[index + 3] = 1.0;
                z[index + 3] = h_j;
                
                x[index + 4] = -1.0;
                y[index + 4] = h_i;
                z[index + 4] = h_j;
                
                x[index + 5] = 1.0;
                y[index + 5] = h_i;
                z[index + 5] = h_j;
                
                index += 6;
            }
        }
    }  // End of "if (pt_dim == 3)"
    if (pt_dim == 2)
    {
        DTYPE *x = unit_pp->data;
        DTYPE *y = unit_pp->data + npts;
        for (int i = 0; i < npts_axis; i++)
        {
            double h_i = h * (i + 1) - 1.0;
            
            x[index + 0] = h_i;
            y[index + 0] = -1.0;
            
            x[index + 1] = h_i;
            y[index + 1] = 1.0;
            
            x[index + 2] = -1.0;
            y[index + 2] = h_i;
            
            x[index + 3] = 1.0;
            y[index + 3] = h_i;
        }
    }  // End of "if (pt_dim == 2)"
    
    // Scale proxy points on unit box surface to different size as
    // proxy points on different levels
    DTYPE pow_2_level = 0.5;
    for (int level = 0; level < start_level; level++) pow_2_level *= 2.0;
    for (int level = start_level; level <= max_level; level++)
    {
        pow_2_level *= 2.0;
        H2P_dense_mat_init(&pp[level], pt_dim, npts);
        DTYPE box_width = max_L / pow_2_level * 0.5;
        DTYPE adm_width = (1.0 + 2.0 * ALPHA_H2) * box_width;
        DTYPE *pp_level = pp[level]->data;
        #pragma omp simd
        for (int i = 0; i < pt_dim * npts; i++)
            pp_level[i] = adm_width * unit_pp->data[i];
    }
    
    H2P_dense_mat_destroy(unit_pp);
    *pp_ = pp;
}

// Build H2 projection matrices using proxy points
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with H2 projection matrices
void H2P_build_H2_UJ_proxy(H2Pack_t h2pack)
{
    int    pt_dim         = h2pack->pt_dim;
    int    krnl_dim       = h2pack->krnl_dim;
    int    n_node         = h2pack->n_node;
    int    n_point        = h2pack->n_point;
    int    n_thread       = h2pack->n_thread;
    int    max_child      = h2pack->max_child;
    int    stop_type      = h2pack->QR_stop_type;
    int    *children      = h2pack->children;
    int    *n_child       = h2pack->n_child;
    int    *node_level    = h2pack->node_level;
    int    *node_height   = h2pack->node_height;
    int    *pt_cluster    = h2pack->pt_cluster;
    DTYPE  *coord         = h2pack->coord;
    DTYPE  *enbox         = h2pack->enbox;
    size_t *mat_size      = h2pack->mat_size;
    void   *krnl_param    = h2pack->krnl_param;
    H2P_dense_mat_t  *pp  = h2pack->pp;
    H2P_thread_buf_t *thread_buf = h2pack->tb;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    DAG_task_queue_t upward_tq   = h2pack->upward_tq;
    void *stop_param = NULL;
    if (stop_type == QR_RANK) 
        stop_param = &h2pack->QR_stop_rank;
    if ((stop_type == QR_REL_NRM) || (stop_type == QR_ABS_NRM))
        stop_param = &h2pack->QR_stop_tol;
    
    // 1. Allocate U and J
    h2pack->n_UJ = n_node;
    h2pack->U       = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    h2pack->J       = (H2P_int_vec_t*)   malloc(sizeof(H2P_int_vec_t)   * n_node);
    h2pack->J_coord = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    assert(h2pack->U != NULL && h2pack->J != NULL && h2pack->J_coord != NULL);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]       = NULL;
        h2pack->J[i]       = NULL;
        h2pack->J_coord[i] = NULL;
    }
    H2P_dense_mat_t *U       = h2pack->U;
    H2P_int_vec_t   *J       = h2pack->J;
    H2P_dense_mat_t *J_coord = h2pack->J_coord;
    
    // 2. Construct U for nodes whose height is not larger than max_adm_height
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_t A_block         = thread_buf[tid]->mat0;
        H2P_dense_mat_t node_skel_coord = thread_buf[tid]->mat1;
        H2P_dense_mat_t QR_buff         = thread_buf[tid]->mat1;
        H2P_int_vec_t   sub_idx         = thread_buf[tid]->idx0;
        H2P_int_vec_t   ID_buff         = thread_buf[tid]->idx1;

        thread_buf[tid]->timer = -get_wtime_sec();
        int node = DAG_task_queue_get_task(upward_tq);
        while (node != -1)
        {
            int height = node_height[node];
            int level  = node_level[node];
            
            // (1) Update row indices associated with clusters for current node
            if (height == 0)
            {
                // Leaf nodes, use all points
                int pt_s = pt_cluster[node * 2];
                int pt_e = pt_cluster[node * 2 + 1];
                int node_npt = pt_e - pt_s + 1;
                H2P_int_vec_init(&J[node], node_npt);
                for (int k = 0; k < node_npt; k++)
                    J[node]->data[k] = pt_s + k;
                J[node]->length = node_npt;
                H2P_dense_mat_init(&J_coord[node], pt_dim, node_npt);
                H2P_copy_matrix_block(pt_dim, node_npt, coord + pt_s, n_point, J_coord[node]->data, node_npt);
            } else {
                // Non-leaf nodes, gather row indices from children nodes
                int n_child_node = n_child[node];
                int *child_nodes = children + node * max_child;
                int J_child_size = 0;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    J_child_size += J[i_child_node]->length;
                }
                H2P_int_vec_init(&J[node], J_child_size);
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    H2P_int_vec_concatenate(J[node], J[i_child_node]);
                }
            }  // End of "if (height == 0)"

            // (2) Gather current node's skeleton points (== all children nodes' skeleton points)
            H2P_dense_mat_resize(node_skel_coord, pt_dim, J[node]->length);
            if (height == 0)
            {
                node_skel_coord = J_coord[node];
            } else {
                int n_child_node = n_child[node];
                int *child_nodes = children + node * max_child;
                int J_child_size = 0;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    int src_ld = J_coord[i_child_node]->ncol;
                    int dst_ld = node_skel_coord->ncol;
                    DTYPE *src_mat = J_coord[i_child_node]->data;
                    DTYPE *dst_mat = node_skel_coord->data + J_child_size; 
                    H2P_copy_matrix_block(pt_dim, src_ld, src_mat, src_ld, dst_mat, dst_ld);
                    J_child_size += J[i_child_node]->length;
                }
            }  // End of "if (level == 0)"
            
            // (3) Shift current node's skeleton points so their center is at the original point
            DTYPE *node_box = enbox + node * 2 * pt_dim;
            int node_skel_npt = J[node]->length;
            int node_pp_npt   = pp[level]->ncol;
            for (int k = 0; k < pt_dim; k++)
            {
                DTYPE box_center_k = node_box[k] + 0.5 * node_box[pt_dim + k];
                DTYPE *node_skel_coord_k = node_skel_coord->data + k * node_skel_npt;
                #pragma omp simd
                for (int l = 0; l < node_skel_npt; l++)
                    node_skel_coord_k[l] -= box_center_k;
            }
            int A_blk_nrow = node_skel_npt * krnl_dim;
            int A_blk_ncol = node_pp_npt   * krnl_dim;
            H2P_dense_mat_resize(A_block, A_blk_nrow, A_blk_ncol);
            krnl_eval(
                node_skel_coord->data, node_skel_npt, node_skel_npt,
                pp[level]->data,       node_pp_npt,   node_pp_npt, 
                krnl_param, A_block->data, A_block->ld
            );

            // (4) ID compress 
            // Note: A is transposed in ID compress, be careful when calculating the buffer size
            if (krnl_dim == 1)
            {
                H2P_dense_mat_resize(QR_buff, A_block->nrow, 1);
            } else {
                int QR_buff_size = (2 * krnl_dim + 2) * A_block->ncol + (krnl_dim + 1) * A_block->nrow;
                H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
            }
            H2P_int_vec_set_capacity(ID_buff, 4 * A_block->nrow);
            H2P_ID_compress(
                A_block, stop_type, stop_param, &U[node], sub_idx, 
                1, QR_buff->data, ID_buff->data, krnl_dim
            );
            
            // (5) Choose the skeleton points of this node
            for (int k = 0; k < sub_idx->length; k++)
                J[node]->data[k] = J[node]->data[sub_idx->data[k]];
            J[node]->length = sub_idx->length;
            H2P_dense_mat_init(&J_coord[node], pt_dim, sub_idx->length);
            H2P_gather_matrix_columns(
                coord, n_point, J_coord[node]->data, J[node]->length, 
                pt_dim, J[node]->data, J[node]->length
            );

            // (6) Tell DAG_task_queue that this node is finished, and get next available node
            DAG_task_queue_finish_task(upward_tq, node);
            node = DAG_task_queue_get_task(upward_tq);
        }  // End of "while (node != -1)"
        thread_buf[tid]->timer += get_wtime_sec();
    }  // End of "#pragma omp parallel num_thread(n_thread)"
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
    printf("[PROFILING] Build U: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif

    // 3. Initialize other not touched U J & add statistic info
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        if (U[i] == NULL)
        {
            H2P_dense_mat_init(&U[i], 1, 1);
            U[i]->nrow = 0;
            U[i]->ncol = 0;
            U[i]->ld   = 0;
        } else {
            mat_size[0] += U[i]->nrow * U[i]->ncol;
            mat_size[3] += U[i]->nrow * U[i]->ncol;
            mat_size[3] += U[i]->nrow + U[i]->ncol;
            mat_size[5] += U[i]->nrow * U[i]->ncol;
            mat_size[5] += U[i]->nrow + U[i]->ncol;
        }
        if (J[i] == NULL) H2P_int_vec_init(&J[i], 1);
        if (J_coord[i] == NULL)
        {
            H2P_dense_mat_init(&J_coord[i], 1, 1);
            J_coord[i]->nrow = 0;
            J_coord[i]->ncol = 0;
            J_coord[i]->ld   = 0;
        }
    }  // End of "for (int i = 0; i < h2pack->n_UJ; i++)"
    
    for (int i = 0; i < n_thread; i++)
        H2P_thread_buf_reset(thread_buf[i]);
    BLAS_SET_NUM_THREADS(n_thread);
}

// Build HSS projection matrices using proxy points and skeleton points from H2 inadmissible nodes
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with HSS projection matrices
void H2P_build_HSS_UJ_hybrid(H2Pack_t h2pack)
{
    int    pt_dim            = h2pack->pt_dim;
    int    max_neighbor      = h2pack->max_neighbor;
    int    krnl_dim          = h2pack->krnl_dim;
    int    n_node            = h2pack->n_node;
    int    n_point           = h2pack->n_point;
    int    n_leaf_node       = h2pack->n_leaf_node;
    int    n_thread          = h2pack->n_thread;
    int    max_child         = h2pack->max_child;
    int    max_level         = h2pack->max_level;
    int    min_adm_level     = h2pack->min_adm_level;
    int    stop_type         = h2pack->QR_stop_type;
    int    *children         = h2pack->children;
    int    *n_child          = h2pack->n_child;
    int    *node_height      = h2pack->node_height;
    int    *level_n_node     = h2pack->level_n_node;
    int    *level_nodes      = h2pack->level_nodes;
    int    *leaf_nodes       = h2pack->height_nodes;
    int    *node_level       = h2pack->node_level;
    int    *pt_cluster       = h2pack->pt_cluster;
    int    *node_n_r_inadm   = h2pack->node_n_r_inadm;
    int    *node_inadm_lists = h2pack->node_inadm_lists;
    DTYPE  *coord            = h2pack->coord;
    DTYPE  *enbox            = h2pack->enbox;
    size_t *mat_size         = h2pack->mat_size;
    void   *krnl_param       = h2pack->krnl_param;
    H2P_dense_mat_t  *pp         = h2pack->pp;
    H2P_thread_buf_t *thread_buf = h2pack->tb;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    void *stop_param = NULL;
    if (stop_type == QR_RANK) 
        stop_param = &h2pack->QR_stop_rank;
    if ((stop_type == QR_REL_NRM) || (stop_type == QR_ABS_NRM))
        stop_param = &h2pack->QR_stop_tol;
    
    // 1. Allocate U and J
    h2pack->n_UJ = n_node;
    h2pack->U       = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    h2pack->J       = (H2P_int_vec_t*)   malloc(sizeof(H2P_int_vec_t)   * n_node);
    h2pack->J_coord = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    assert(h2pack->U != NULL && h2pack->J != NULL && h2pack->J_coord != NULL);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]       = NULL;
        h2pack->J[i]       = NULL;
        h2pack->J_coord[i] = NULL;
    }
    H2P_dense_mat_t *U       = h2pack->U;
    H2P_int_vec_t   *J       = h2pack->J;
    H2P_dense_mat_t *J_coord = h2pack->J_coord;
    
    // 2. Initialize the row indices for leaf nodes: all points in that box
    for (int j = 0; j < n_leaf_node; j++)
    {
        int node = leaf_nodes[j];
        int pt_s = pt_cluster[node * 2];
        int pt_e = pt_cluster[node * 2 + 1];
        int node_npt = pt_e - pt_s + 1;
        H2P_int_vec_init(&J[node], node_npt);
        for (int k = 0; k < node_npt; k++)
            J[node]->data[k] = pt_s + k;
        J[node]->length = node_npt;
        H2P_dense_mat_init(&J_coord[node], pt_dim, node_npt);
        H2P_copy_matrix_block(pt_dim, node_npt, coord + pt_s, n_point, J_coord[node]->data, node_npt);
    }

    // 3. Hierarchical construction level by level. min_adm_level is the 
    //    highest level that still has admissible blocks, so we only need 
    //    to compress matrix blocks to that level since higher blocks 
    //    are inadmissible and cannot be compressed.
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);
        int level = i;

        // (1) Update row indices associated with clusters at level i
        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            thread_buf[tid]->timer = -get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                if (n_child_node == 0) continue;  // J[node] has already been prepared for leaf node
                int *child_nodes = children + node * max_child;
                int J_child_size = 0;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    J_child_size += J[i_child_node]->length;
                }
                H2P_int_vec_init(&J[node], J_child_size);
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    H2P_int_vec_concatenate(J[node], J[i_child_node]);
                }
            }
            thread_buf[tid]->timer += get_wtime_sec();
        }
        
        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            H2P_int_vec_t   inadm_skel_idx   = thread_buf[tid]->idx0;
            H2P_int_vec_t   sub_idx          = thread_buf[tid]->idx0;
            H2P_int_vec_t   ID_buff          = thread_buf[tid]->idx1;
            H2P_dense_mat_t node_skel_coord  = thread_buf[tid]->mat0;
            H2P_dense_mat_t inadm_skel_coord = thread_buf[tid]->mat1;
            H2P_dense_mat_t A_block          = thread_buf[tid]->mat2;
            H2P_dense_mat_t QR_buff          = thread_buf[tid]->mat1;
            
            thread_buf[tid]->timer -= get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int height = node_height[node];

                // (2) Calculate the number of skeleton points from all inadmissible nodes
                int n_r_inadm = node_n_r_inadm[node];
                int inadm_skel_npt = 0;
                inadm_skel_idx->length = 0;
                if (n_r_inadm > 0)
                {
                    int *inadm_list = node_inadm_lists + node * max_neighbor;
                    for (int k = 0; k < n_r_inadm; k++)
                    {
                        int inadm_node_k = inadm_list[k];
                        if (J[inadm_node_k] == NULL)
                        {
                            int k_level = node_level[k];
                            printf(
                                "[FATAL] Node %3d (lvl %2d): near field node %3d (lvl %2d) skeleton point not ready!\n",
                                node, level, inadm_node_k, k_level
                            );
                            fflush(stdout);
                        }
                        assert(J[inadm_node_k] != NULL);
                        H2P_int_vec_concatenate(inadm_skel_idx, J[inadm_node_k]);
                    }
                    inadm_skel_npt = inadm_skel_idx->length;
                }
                
                // (3) Gather current node's skeleton points (== all children nodes' skeleton points)
                H2P_dense_mat_resize(node_skel_coord, pt_dim, J[node]->length);
                if (height == 0)
                {
                    node_skel_coord = J_coord[node];
                } else {
                    int n_child_node = n_child[node];
                    int *child_nodes = children + node * max_child;
                    int J_child_size = 0;
                    for (int i_child = 0; i_child < n_child_node; i_child++)
                    {
                        int i_child_node = child_nodes[i_child];
                        int src_ld = J_coord[i_child_node]->ncol;
                        int dst_ld = node_skel_coord->ncol;
                        DTYPE *src_mat = J_coord[i_child_node]->data;
                        DTYPE *dst_mat = node_skel_coord->data + J_child_size; 
                        H2P_copy_matrix_block(pt_dim, src_ld, src_mat, src_ld, dst_mat, dst_ld);
                        J_child_size += J[i_child_node]->length;
                    }
                }
                
                // (4) Shift points in this node so their center is at the original point
                DTYPE *node_box = enbox + node * 2 * pt_dim;
                int node_skel_npt = J[node]->length;
                int node_pp_npt = pp[level]->ncol;
                for (int k = 0; k < pt_dim; k++)
                {
                    DTYPE box_center_k = node_box[k] + 0.5 * node_box[pt_dim + k];
                    DTYPE *node_skel_coord_k = node_skel_coord->data + k * node_skel_npt;
                    #pragma omp simd
                    for (int l = 0; l < node_skel_npt; l++)
                        node_skel_coord_k[l] -= box_center_k;
                }

                // (5) Gather skeleton points from all inadmissible nodes and shift their centers to 
                //     the original point. We also put proxy points into inadm_skel_coord.
                H2P_dense_mat_resize(inadm_skel_coord, pt_dim, inadm_skel_npt + node_pp_npt);
                if (inadm_skel_npt > 0)
                {
                    for (int k = 0; k < pt_dim; k++)
                    {
                        DTYPE box_center_k = node_box[k] + 0.5 * node_box[pt_dim + k];
                        DTYPE *coord_k = coord + k * n_point;
                        DTYPE *inadm_skel_coord_k = inadm_skel_coord->data + k * inadm_skel_coord->ld;
                        #pragma omp simd 
                        for (int l = 0; l < inadm_skel_npt; l++)
                            inadm_skel_coord_k[l] = coord_k[inadm_skel_idx->data[l]] - box_center_k;
                    }
                }
                if (node_pp_npt > 0)
                {
                    H2P_copy_matrix_block(
                        pt_dim, node_pp_npt, pp[level]->data, pp[level]->ld, 
                        inadm_skel_coord->data + inadm_skel_npt, inadm_skel_coord->ld
                    );
                }

                // (6) Build the hybrid matrix block
                int A_blk_nrow = node_skel_npt * krnl_dim;
                int A_blk_ncol = inadm_skel_coord->ncol * krnl_dim;
                H2P_dense_mat_resize(A_block, A_blk_nrow, A_blk_ncol);
                krnl_eval(
                    node_skel_coord->data,  node_skel_coord->ncol,  node_skel_coord->ld,
                    inadm_skel_coord->data, inadm_skel_coord->ncol, inadm_skel_coord->ld, 
                    krnl_param, A_block->data, A_block->ld
                );
                H2P_dense_mat_normalize_columns(A_block, inadm_skel_coord);

                // (7) ID compress
                // Note: A is transposed in ID compress, be careful when calculating the buffer size
                if (krnl_dim == 1)
                {
                    H2P_dense_mat_resize(QR_buff, A_block->nrow, 1);
                } else {
                    int QR_buff_size = (2 * krnl_dim + 2) * A_block->ncol + (krnl_dim + 1) * A_block->nrow;
                    H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
                }
                H2P_int_vec_set_capacity(ID_buff, 4 * A_block->nrow);
                H2P_ID_compress(
                    A_block, stop_type, stop_param, &U[node], sub_idx, 
                    1, QR_buff->data, ID_buff->data, krnl_dim
                );
                
                // (8) Choose the skeleton points of this node
                for (int k = 0; k < sub_idx->length; k++)
                    J[node]->data[k] = J[node]->data[sub_idx->data[k]];
                J[node]->length = sub_idx->length;
                H2P_dense_mat_init(&J_coord[node], pt_dim, sub_idx->length);
                H2P_gather_matrix_columns(
                    coord, n_point, J_coord[node]->data, J[node]->length, 
                    pt_dim, J[node]->data, J[node]->length
                );
            }  // End of j loop
            thread_buf[tid]->timer += get_wtime_sec();
        }  // End of "pragma omp parallel"
        
        #ifdef PROFILING_OUTPUT
        double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
        for (int i = 0; i < n_thread_i; i++)
        {
            double thread_i_timer = thread_buf[i]->timer;
            avg_t += thread_i_timer;
            max_t = MAX(max_t, thread_i_timer);
            min_t = MIN(min_t, thread_i_timer);
        }
        avg_t /= (double) n_thread_i;
        printf("[PROFILING] Build U: height %d, %d/%d threads, %d nodes, ", i, n_thread_i, n_thread, height_n_node[i]);
        printf("min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
        #endif
    }  // End of i loop

    // 3. Initialize other not touched U J & add statistic info
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        if (U[i] == NULL)
        {
            H2P_dense_mat_init(&U[i], 1, 1);
            U[i]->nrow = 0;
            U[i]->ncol = 0;
            U[i]->ld   = 0;
        } else {
            mat_size[0] += U[i]->nrow * U[i]->ncol;
            mat_size[3] += U[i]->nrow * U[i]->ncol;
            mat_size[3] += U[i]->nrow + U[i]->ncol;
            mat_size[5] += U[i]->nrow * U[i]->ncol;
            mat_size[5] += U[i]->nrow + U[i]->ncol;
        }
        if (J[i] == NULL) H2P_int_vec_init(&J[i], 1);
        if (J_coord[i] == NULL)
        {
            H2P_dense_mat_init(&J_coord[i], 1, 1);
            J_coord[i]->nrow = 0;
            J_coord[i]->ncol = 0;
            J_coord[i]->ld   = 0;
        }
        //printf("Node %3d: %d skeleton points\n", i, J[i]->length);
    }
    
    for (int i = 0; i < n_thread; i++)
        H2P_thread_buf_reset(thread_buf[i]);
    BLAS_SET_NUM_THREADS(n_thread);
}

// Partition work units into multiple blocks s.t. each block has 
// approximately the same amount of work
void H2P_partition_workload(
    const int n_work,  const size_t *work_sizes, const size_t total_size, 
    const int n_block, H2P_int_vec_t blk_displs
)
{
    blk_displs->data[0] = 0;
    for (int i = 1; i < blk_displs->capacity; i++) 
        blk_displs->data[i] = n_work;
    size_t blk_size = total_size / n_block + 1;
    size_t curr_blk_size = 0;
    int idx = 1;
    for (int i = 0; i < n_work; i++)
    {
        curr_blk_size += work_sizes[i];
        if (curr_blk_size >= blk_size)
        {
            blk_displs->data[idx] = i + 1;
            curr_blk_size = 0;
            idx++;
        }
    }
    if (curr_blk_size > 0)
    {
        blk_displs->data[idx] = n_work;
        idx++;
    }
    blk_displs->length = idx;
}

// Build H2 generator matrices
// Input parameter:
//   h2pack : H2Pack structure with H2 projection matrices
// Output parameter:
//   h2pack : H2Pack structure with H2 generator matrices
void H2P_build_B(H2Pack_t h2pack)
{
    int    BD_JIT          = h2pack->BD_JIT;
    int    krnl_dim        = h2pack->krnl_dim;
    int    n_node          = h2pack->n_node;
    int    n_point         = h2pack->n_point;
    int    n_thread        = h2pack->n_thread;
    int    krnl_bimv_flops = h2pack->krnl_bimv_flops;
    int    *node_level     = h2pack->node_level;
    int    *pt_cluster     = h2pack->pt_cluster;
    DTYPE  *coord          = h2pack->coord;
    size_t *mat_size       = h2pack->mat_size;
    void   *krnl_param     = h2pack->krnl_param;
    double *JIT_flops      = h2pack->JIT_flops;
    kernel_eval_fptr krnl_eval = h2pack->krnl_eval;
    H2P_int_vec_t    B_blk     = h2pack->B_blk;
    H2P_dense_mat_t  *J_coord  = h2pack->J_coord;
    H2P_thread_buf_t *thread_buf = h2pack->tb;

    int n_r_adm_pair, *r_adm_pairs; 
    if (h2pack->is_HSS)
    {
        n_r_adm_pair = h2pack->HSS_n_r_adm_pair;
        r_adm_pairs  = h2pack->HSS_r_adm_pairs;
    } else {
        n_r_adm_pair = h2pack->n_r_adm_pair;
        r_adm_pairs  = h2pack->r_adm_pairs;
    }

    // 1. Allocate B
    h2pack->n_B = n_r_adm_pair;
    h2pack->B_nrow = (int*)    malloc(sizeof(int)    * n_r_adm_pair);
    h2pack->B_ncol = (int*)    malloc(sizeof(int)    * n_r_adm_pair);
    h2pack->B_ptr  = (size_t*) malloc(sizeof(size_t) * (n_r_adm_pair + 1));
    int    *B_nrow = h2pack->B_nrow;
    int    *B_ncol = h2pack->B_ncol;
    size_t *B_ptr  = h2pack->B_ptr;
    assert(h2pack->B_nrow != NULL && h2pack->B_ncol != NULL && h2pack->B_ptr != NULL);
    
    // 2. Partition B matrices into multiple blocks s.t. each block has approximately
    //    the same workload (total size of B matrices in a block)
    B_ptr[0] = 0;
    size_t B_total_size = 0;
    H2P_int_vec_t *J = h2pack->J;
    h2pack->node_n_r_adm = (int*) malloc(sizeof(int) * n_node);
    assert(h2pack->node_n_r_adm != NULL);
    int *node_n_r_adm = h2pack->node_n_r_adm;
    memset(node_n_r_adm, 0, sizeof(int) * n_node);
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int node0  = r_adm_pairs[2 * i];
        int node1  = r_adm_pairs[2 * i + 1];
        int level0 = node_level[node0];
        int level1 = node_level[node1];
        node_n_r_adm[node0]++;
        node_n_r_adm[node1]++;
        int node0_npt = 0, node1_npt = 0;
        if (level0 == level1)
        {
            node0_npt = J[node0]->length;
            node1_npt = J[node1]->length;
        }
        if (level0 > level1)
        {
            int pt_s1 = pt_cluster[2 * node1];
            int pt_e1 = pt_cluster[2 * node1 + 1];
            node0_npt = J[node0]->length;
            node1_npt = pt_e1 - pt_s1 + 1;
        }
        if (level0 < level1)
        {
            int pt_s0 = pt_cluster[2 * node0];
            int pt_e0 = pt_cluster[2 * node0 + 1];
            node0_npt = pt_e0 - pt_s0 + 1;
            node1_npt = J[node1]->length;
        }
        B_nrow[i] = node0_npt * krnl_dim;
        B_ncol[i] = node1_npt * krnl_dim;
        size_t Bi_size = (size_t) B_nrow[i] * (size_t) B_ncol[i];
        //Bi_size = (Bi_size + N_DTYPE_64B - 1) / N_DTYPE_64B * N_DTYPE_64B;
        B_total_size += Bi_size;
        B_ptr[i + 1] = Bi_size;
        // Add Statistic info
        mat_size[4]  += B_nrow[i] * B_ncol[i];
        mat_size[4]  += 2 * (B_nrow[i] + B_ncol[i]);
        JIT_flops[0] += (double)(krnl_bimv_flops) * (double)(node0_npt * node1_npt);
    }
    int BD_ntask_thread = (BD_JIT == 1) ? BD_NTASK_THREAD : 1;
    H2P_partition_workload(n_r_adm_pair, B_ptr + 1, B_total_size, n_thread * BD_ntask_thread, B_blk);
    for (int i = 1; i <= n_r_adm_pair; i++) B_ptr[i] += B_ptr[i - 1];
    mat_size[1] = B_total_size;

    if (BD_JIT == 1) return;

    // 3. Generate B matrices
    h2pack->B_data = (DTYPE*) malloc_aligned(sizeof(DTYPE) * B_total_size, 64);
    assert(h2pack->B_data != NULL);
    DTYPE *B_data = h2pack->B_data;
    const int n_B_blk = B_blk->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        
        thread_buf[tid]->timer = -get_wtime_sec();
        //#pragma omp for schedule(dynamic) nowait
        //for (int i_blk = 0; i_blk < n_B_blk; i_blk++)
        int i_blk = tid;    // Use first-touch policy for better NUMA memory access performance
        {
            int B_blk_s = B_blk->data[i_blk];
            int B_blk_e = B_blk->data[i_blk + 1];
            if (i_blk >= n_B_blk)
            {
                B_blk_s = 0; 
                B_blk_e = 0;
            }
            for (int i = B_blk_s; i < B_blk_e; i++)
            {
                int node0  = r_adm_pairs[2 * i];
                int node1  = r_adm_pairs[2 * i + 1];
                int level0 = node_level[node0];
                int level1 = node_level[node1];
                DTYPE *Bi  = B_data + B_ptr[i];

                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    krnl_eval(
                        J_coord[node0]->data, J_coord[node0]->ncol, J_coord[node0]->ncol,
                        J_coord[node1]->data, J_coord[node1]->ncol, J_coord[node1]->ncol,
                        krnl_param, Bi, J_coord[node1]->ncol * krnl_dim
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
                        krnl_param, Bi, node1_npt * krnl_dim
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
                        krnl_param, Bi, J_coord[node1]->ncol * krnl_dim
                    );
                }
            }  // End of i loop
        }  // End of i_blk loop
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
    printf("[PROFILING] Build B: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
}

// Build dense blocks in the original matrices
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with dense blocks
void H2P_build_D(H2Pack_t h2pack)
{
    int    BD_JIT          = h2pack->BD_JIT;
    int    krnl_dim        = h2pack->krnl_dim;
    int    n_thread        = h2pack->n_thread;
    int    n_point         = h2pack->n_point;
    int    n_leaf_node     = h2pack->n_leaf_node;
    int    krnl_bimv_flops = h2pack->krnl_bimv_flops;
    int    *leaf_nodes     = h2pack->height_nodes;
    int    *pt_cluster     = h2pack->pt_cluster;
    DTYPE  *coord          = h2pack->coord;
    size_t *mat_size       = h2pack->mat_size;
    void   *krnl_param     = h2pack->krnl_param;
    double *JIT_flops      = h2pack->JIT_flops;
    kernel_eval_fptr krnl_eval = h2pack->krnl_eval;
    H2P_int_vec_t    D_blk0    = h2pack->D_blk0;
    H2P_int_vec_t    D_blk1    = h2pack->D_blk1;
    H2P_thread_buf_t *thread_buf = h2pack->tb;
    
    int n_r_inadm_pair, *r_inadm_pairs;
    if (h2pack->is_HSS)
    {
        n_r_inadm_pair = h2pack->HSS_n_r_inadm_pair;
        r_inadm_pairs  = h2pack->HSS_r_inadm_pairs;
    } else {
        n_r_inadm_pair = h2pack->n_r_inadm_pair;
        r_inadm_pairs  = h2pack->r_inadm_pairs;
    }

    // 1. Allocate D
    h2pack->n_D = n_leaf_node + n_r_inadm_pair;
    h2pack->D_nrow = (int*)    malloc(sizeof(int)    * h2pack->n_D);
    h2pack->D_ncol = (int*)    malloc(sizeof(int)    * h2pack->n_D);
    h2pack->D_ptr  = (size_t*) malloc(sizeof(size_t) * (h2pack->n_D + 1));
    int    *D_nrow = h2pack->D_nrow;
    int    *D_ncol = h2pack->D_ncol;
    size_t *D_ptr  = h2pack->D_ptr;
    assert(h2pack->D_nrow != NULL && h2pack->D_ncol != NULL && h2pack->D_ptr != NULL);
    
    // 2. Partition D matrices into multiple blocks s.t. each block has approximately
    //    the same workload (total size of D matrices in a block)
    D_ptr[0] = 0;
    size_t D0_total_size = 0;
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int pt_s = pt_cluster[2 * node];
        int pt_e = pt_cluster[2 * node + 1];
        int node_npt = pt_e - pt_s + 1;
        D_nrow[i] = node_npt * krnl_dim;
        D_ncol[i] = node_npt * krnl_dim;
        size_t Di_size = (size_t) D_nrow[i] * (size_t) D_ncol[i];
        //Di_size = (Di_size + N_DTYPE_64B - 1) / N_DTYPE_64B * N_DTYPE_64B;
        D_ptr[i + 1] = Di_size;
        D0_total_size += Di_size;
        // Add statistic info
        mat_size[6] += D_nrow[i] * D_ncol[i];
        mat_size[6] += D_nrow[i] + D_ncol[i];
        JIT_flops[1] += (double)(krnl_bimv_flops) * (double)(node_npt * node_npt);
    }
    int BD_ntask_thread = (BD_JIT == 1) ? BD_NTASK_THREAD : 1;
    H2P_partition_workload(n_leaf_node, D_ptr + 1, D0_total_size, n_thread * BD_ntask_thread, D_blk0);
    size_t D1_total_size = 0;
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int ii = i + n_leaf_node;
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int pt_s0 = pt_cluster[2 * node0];
        int pt_s1 = pt_cluster[2 * node1];
        int pt_e0 = pt_cluster[2 * node0 + 1];
        int pt_e1 = pt_cluster[2 * node1 + 1];
        int node0_npt = pt_e0 - pt_s0 + 1;
        int node1_npt = pt_e1 - pt_s1 + 1;
        D_nrow[ii] = node0_npt * krnl_dim;
        D_ncol[ii] = node1_npt * krnl_dim;
        size_t Di_size = (size_t) D_nrow[ii] * (size_t) D_ncol[ii];
        //Di_size = (Di_size + N_DTYPE_64B - 1) / N_DTYPE_64B * N_DTYPE_64B;
        D_ptr[ii + 1] = Di_size;
        D1_total_size += Di_size;
        // Add statistic info
        mat_size[6] += D_nrow[ii] * D_ncol[ii];
        mat_size[6] += 2 * (D_nrow[ii] + D_ncol[ii]);
        JIT_flops[1] += (double)(krnl_bimv_flops) * (double)(node0_npt * node1_npt);
    }
    H2P_partition_workload(n_r_inadm_pair, D_ptr + n_leaf_node + 1, D1_total_size, n_thread * BD_ntask_thread, D_blk1);
    for (int i = 1; i <= n_leaf_node + n_r_inadm_pair; i++) D_ptr[i] += D_ptr[i - 1];
    mat_size[2] = D0_total_size + D1_total_size;
    
    if (BD_JIT == 1) return;
    
    h2pack->D_data = (DTYPE*) malloc_aligned(sizeof(DTYPE) * (D0_total_size + D1_total_size), 64);
    assert(h2pack->D_data != NULL);
    DTYPE *D_data = h2pack->D_data;
    const int n_D0_blk = D_blk0->length - 1;
    const int n_D1_blk = D_blk1->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        
        thread_buf[tid]->timer = -get_wtime_sec();
        
        // 3. Generate diagonal blocks (leaf node self interaction)
        //#pragma omp for schedule(dynamic) nowait
        //for (int i_blk0 = 0; i_blk0 < n_D0_blk; i_blk0++)
        int i_blk0 = tid;    // Use first-touch policy for better NUMA memory access performance
        {
            int D_blk0_s = D_blk0->data[i_blk0];
            int D_blk0_e = D_blk0->data[i_blk0 + 1];
            if (i_blk0 >= n_D0_blk)
            {
                D_blk0_s = 0;
                D_blk0_e = 0;
            }
            for (int i = D_blk0_s; i < D_blk0_e; i++)
            {
                int node = leaf_nodes[i];
                int pt_s = pt_cluster[2 * node];
                int pt_e = pt_cluster[2 * node + 1];
                int node_npt = pt_e - pt_s + 1;
                DTYPE *Di = D_data + D_ptr[i];
                krnl_eval(
                    coord + pt_s, n_point, node_npt,
                    coord + pt_s, n_point, node_npt,
                    krnl_param, Di, node_npt * krnl_dim
                );
            }
        }  // End of i_blk0 loop
        
        // 4. Generate off-diagonal blocks from inadmissible pairs
        //#pragma omp for schedule(dynamic) nowait
        //for (int i_blk1 = 0; i_blk1 < n_D1_blk; i_blk1++)
        int i_blk1 = tid;    // Use first-touch policy for better NUMA memory access performance
        {
            int D_blk1_s = D_blk1->data[i_blk1];
            int D_blk1_e = D_blk1->data[i_blk1 + 1];
            if (i_blk1 >= n_D1_blk)
            {
                D_blk1_s = 0;
                D_blk1_e = 0;
            }
            for (int i = D_blk1_s; i < D_blk1_e; i++)
            {
                int node0 = r_inadm_pairs[2 * i];
                int node1 = r_inadm_pairs[2 * i + 1];
                int pt_s0 = pt_cluster[2 * node0];
                int pt_s1 = pt_cluster[2 * node1];
                int pt_e0 = pt_cluster[2 * node0 + 1];
                int pt_e1 = pt_cluster[2 * node1 + 1];
                int node0_npt = pt_e0 - pt_s0 + 1;
                int node1_npt = pt_e1 - pt_s1 + 1;
                DTYPE *Di = D_data + D_ptr[i + n_leaf_node];
                krnl_eval(
                    coord + pt_s0, n_point, node0_npt,
                    coord + pt_s1, n_point, node1_npt,
                    krnl_param, Di, node1_npt * krnl_dim
                );
            }
        }  // End of i_blk1 loop
        
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
    printf("[PROFILING] Build D: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
}

// Build H2 representation with a kernel function
void H2P_build(
    H2Pack_t h2pack, H2P_dense_mat_t *pp, const int BD_JIT, void *krnl_param, 
    kernel_eval_fptr krnl_eval, kernel_bimv_fptr krnl_bimv, const int krnl_bimv_flops
)
{
    double st, et;

    if (pp == NULL)
    {
        fprintf(stderr, "[ERROR] You need to provide a set of proxy points for H2P_build()!\n");
        fprintf(stderr, "[ERROR] H2P_build() exits without doing anything.\n");
        return;
    }
    
    if (krnl_eval == NULL)
    {
        fprintf(stderr, "[ERROR] You need to provide a valid krnl_eval() for H2P_build()!\n");
        fprintf(stderr, "[ERROR] H2P_build() exits without doing anything.\n");
        return;
    }

    h2pack->pp = pp;
    h2pack->BD_JIT = BD_JIT;
    h2pack->krnl_param = krnl_param;
    h2pack->krnl_eval  = krnl_eval;
    h2pack->krnl_bimv  = krnl_bimv;
    h2pack->krnl_bimv_flops = krnl_bimv_flops;
    if (BD_JIT == 1 && krnl_bimv == NULL) 
        printf("[WARNING] krnl_eval() will be used in BD_JIT matvec. For better performance, krnl_bimv() should be provided. \n");

    // 1. Build projection matrices and skeleton row sets
    st = get_wtime_sec();
    if (h2pack->is_HSS) H2P_build_HSS_UJ_hybrid(h2pack);
    else H2P_build_H2_UJ_proxy(h2pack);
    et = get_wtime_sec();
    h2pack->timers[1] = et - st;

    // 2. Build generator matrices
    st = get_wtime_sec();
    H2P_build_B(h2pack);
    et = get_wtime_sec();
    h2pack->timers[2] = et - st;
    
    // 3. Build dense blocks
    st = get_wtime_sec();
    H2P_build_D(h2pack);
    et = get_wtime_sec();
    h2pack->timers[3] = et - st;
}

