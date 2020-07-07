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
#include "H2Pack_utils.h"
#include "DAG_task_queue.h"
#include "utils.h"

// Generate proxy points for constructing H2 projection and skeleton matrices
// using ID compress for any kernel function
void H2P_generate_proxy_point_ID(
    const int pt_dim, const int krnl_dim, const DTYPE tol_norm, const int max_level, const int min_level,
    DTYPE max_L, const void *krnl_param, kernel_eval_fptr krnl_eval, H2P_dense_mat_t **pp_
)
{   
    // 1. Initialize proxy point arrays
    H2P_dense_mat_t *pp = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * (max_level + 1));
    ASSERT_PRINTF(pp != NULL, "Failed to allocate %d H2P_dense_mat structues for storing proxy points", max_level + 1);
    for (int i = 0; i <= max_level; i++) pp[i] = NULL;
    
    // 2. Initialize temporary arrays. The numbers of Nx and Ny points are empirical values
    int Nx_size = 1000;
    int Ny_size = 15000;
    H2P_dense_mat_t tmpA, Nx_points, Ny_points, min_dist;
    H2P_int_vec_t skel_idx;
    H2P_dense_mat_init(&Nx_points, pt_dim,  Nx_size);
    H2P_dense_mat_init(&Ny_points, pt_dim,  Ny_size);
    H2P_dense_mat_init(&tmpA,      Nx_size * krnl_dim, Ny_size * krnl_dim);
    H2P_dense_mat_init(&min_dist,  Nx_size, 1);
    H2P_int_vec_init(&skel_idx, Nx_size);
    srand48(time(NULL));
    int n_thread = omp_get_max_threads();

    double other_t, krnl_t, spmm_t, ID_t, st, et;

    // 3. Construct proxy points on each level
    DTYPE pow_2_level = 0.5;
    H2P_dense_mat_t QR_buff;
    H2P_int_vec_t   ID_buff;
    H2P_dense_mat_init(&QR_buff, 2 * Ny_size, 1);
    H2P_int_vec_init(&ID_buff, 4 * Ny_size);
    for (int level = 0; level < min_level; level++) pow_2_level *= 2.0;
    for (int level = min_level; level <= max_level; level++)
    {
        if (level < 2)
        {
            H2P_dense_mat_init(&pp[level], pt_dim, 1);
            pp[level]->ncol = 0;
            pow_2_level *= 2.0;
            WARNING_PRINTF("Level %d: no proxy points are selected\n", level);
            continue;
        }
        other_t = 0.0;
        krnl_t  = 0.0;
        spmm_t  = 0.0;
        ID_t    = 0.0;

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
        st = get_wtime_sec();
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
        et = get_wtime_sec();
        other_t += et - st;

        // (3) Use ID to select skeleton points in Nx first, then use the
        //     skeleton Nx points to select skeleton Ny points
        DTYPE rel_tol = tol_norm < 1e-11 ? 1e-14 : tol_norm * 1e-2;
        // (3.1) Use sparsity + randomize to reduce the cost of Nx points selection
        //       Using a dense Gaussian random matrix is also OK, but might generate
        //       less proxy points. For accuracy, we use a sparse one. 
        st = get_wtime_sec();
        H2P_eval_kernel_matrix_OMP(krnl_param, krnl_eval, krnl_dim, Nx_points, Ny_points, tmpA);
        et = get_wtime_sec();
        krnl_t += et - st;
        H2P_int_vec_t   rndmat_idx = ID_buff;
        H2P_dense_mat_t rndmat_val = QR_buff;
        H2P_dense_mat_t tmpA1      = min_dist;
        st = get_wtime_sec();
        H2P_gen_rand_sparse_mat_trans(32, tmpA->ncol, tmpA->nrow, rndmat_val, rndmat_idx);
        H2P_dense_mat_resize(tmpA1, tmpA->nrow, tmpA->nrow);
        H2P_calc_sparse_mm_trans(
            tmpA->nrow, tmpA->nrow, tmpA->ncol, rndmat_val, rndmat_idx,
            tmpA->data, tmpA->ld, tmpA1->data, tmpA1->ld
        );
        et = get_wtime_sec();
        spmm_t += et - st;
        if (krnl_dim == 1)
        {
            H2P_dense_mat_resize(QR_buff, tmpA1->nrow, 1);
        } else {
            int QR_buff_size = (2 * krnl_dim + 2) * tmpA1->ncol + (krnl_dim + 1) * tmpA1->nrow;
            H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
        }
        H2P_int_vec_set_capacity(ID_buff, 4 * tmpA1->nrow);
        st = get_wtime_sec();
        H2P_ID_compress(
            tmpA1, QR_REL_NRM, &rel_tol, NULL, skel_idx, 
            n_thread, QR_buff->data, ID_buff->data, krnl_dim
        );
        et = get_wtime_sec();
        ID_t += et - st;
        st = get_wtime_sec();
        H2P_dense_mat_select_columns(Nx_points, skel_idx);
        et = get_wtime_sec();
        other_t += et - st;
        // (3.2) Use the skeleton points to select the proxy points Ny
        st = get_wtime_sec();
        H2P_eval_kernel_matrix_OMP(krnl_param, krnl_eval, krnl_dim, Ny_points, Nx_points, tmpA);
        et = get_wtime_sec();
        krnl_t += et - st;
        if (krnl_dim == 1)
        {
            H2P_dense_mat_resize(QR_buff, tmpA->nrow, 1);
        } else {
            int QR_buff_size = (2 * krnl_dim + 2) * tmpA->ncol + (krnl_dim + 1) * tmpA->nrow;
            H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
        }
        H2P_int_vec_set_capacity(ID_buff, 4 * tmpA->nrow);
        st = get_wtime_sec();
        H2P_ID_compress(
            tmpA, QR_REL_NRM, &rel_tol, NULL, skel_idx, 
            n_thread, QR_buff->data, ID_buff->data, krnl_dim
        );
        et = get_wtime_sec();
        ID_t += et - st;
        st = get_wtime_sec();
        H2P_dense_mat_select_columns(Ny_points, skel_idx);
        et = get_wtime_sec();
        other_t += et - st;
        
        // (4) Set up the proxy points
        st = get_wtime_sec();
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
        et = get_wtime_sec();
        other_t += et - st;
        #ifdef PROFILING_OUTPUT
        printf("%s: Level %d : Nx, Npp = %d, %d\n", __FUNCTION__, level, Ny_points->ncol, pp[level]->ncol);
        //printf("    kernel, SpMM, ID, other time = %.3lf, %.3lf, %.3lf, %.3lf s\n", krnl_t, spmm_t, ID_t, other_t);
        #endif
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
    const int pt_dim, const int xpt_dim, const int min_npt, const int max_level, 
    const int min_level, DTYPE max_L, H2P_dense_mat_t **pp_
)
{
    if (pt_dim < 2 || pt_dim > 3)
    {
        ERROR_PRINTF("Only 2D and 3D systems are supported in this function.\n");
        return;
    }
    
    H2P_dense_mat_t *pp = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * (max_level + 1));
    ASSERT_PRINTF(pp != NULL, "Failed to allocate %d H2P_dense_mat structues for storing proxy points", max_level + 1);
    for (int i = 0; i <= max_level; i++) pp[i] = NULL;
    
    int npt_axis, npt;
    if (pt_dim == 2)
    {
        npt_axis = (min_npt + 3) / 4;
        npt = npt_axis * 4;
    } else {
        DTYPE n_point_face = (DTYPE) min_npt / 6.0;
        npt_axis = (int) ceil(sqrt(n_point_face));
        npt = npt_axis * npt_axis * 6;
    }
    DTYPE h = 2.0 / (DTYPE) (npt_axis + 1);
    
    // Generate proxy points on the surface of [-1,1]^pt_dim box
    H2P_dense_mat_t unit_pp;
    H2P_dense_mat_init(&unit_pp, xpt_dim, npt);
    int index = 0;
    if (pt_dim == 3)
    {
        DTYPE *x = unit_pp->data;
        DTYPE *y = unit_pp->data + npt;
        DTYPE *z = unit_pp->data + npt * 2;
        DTYPE h_i = -1.0;
        for (int i = 0; i < npt_axis; i++)
        {
            h_i += h;
            DTYPE h_j = -1.0;
            for (int j = 0; j < npt_axis; j++)
            {
                h_j += h;
                
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
        DTYPE *y = unit_pp->data + npt;
        DTYPE h_i = -1.0;
        for (int i = 0; i < npt_axis; i++)
        {
            h_i += h;
            
            x[index + 0] = h_i;
            y[index + 0] = -1.0;
            
            x[index + 1] = h_i;
            y[index + 1] = 1.0;
            
            x[index + 2] = -1.0;
            y[index + 2] = h_i;
            
            x[index + 3] = 1.0;
            y[index + 3] = h_i;

            index += 4;
        }
    }  // End of "if (pt_dim == 2)"
    
    if (xpt_dim > pt_dim)
    {
        DTYPE *ext = unit_pp->data + npt * pt_dim;
        memset(ext, 0, sizeof(DTYPE) * npt);
    }

    // Scale proxy points on unit box surface to different size as
    // proxy points on different levels
    DTYPE pow_2_level = 0.5;
    for (int level = 0; level < min_level; level++) pow_2_level *= 2.0;
    for (int level = min_level; level <= max_level; level++)
    {
        pow_2_level *= 2.0;
        H2P_dense_mat_init(&pp[level], xpt_dim, npt);
        DTYPE box_width = max_L / pow_2_level * 0.5;
        DTYPE adm_width = (1.0 + 2.0 * ALPHA_H2) * box_width;
        DTYPE *pp_level = pp[level]->data;
        #pragma omp simd
        for (int i = 0; i < xpt_dim * npt; i++)
            pp_level[i] = adm_width * unit_pp->data[i];
    }
    
    H2P_dense_mat_destroy(unit_pp);
    *pp_ = pp;
}

#define _U_BUILD_KRNL_T_IDX     0
#define _U_BUILD_QR_T_IDX       1
#define _U_BUILD_OTHER_T_IDX    2
#define _U_BUILD_RANDN_T_IDX    3
#define _U_BUILD_GEMM_T_IDX     4

// Build H2 projection matrices using proxy points
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with H2 projection matrices
void H2P_build_H2_UJ_proxy(H2Pack_t h2pack)
{
    int    pt_dim         = h2pack->pt_dim;
    int    xpt_dim        = h2pack->xpt_dim;
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
    ASSERT_PRINTF(h2pack->U       != NULL, "Failed to allocate %d U matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J       != NULL, "Failed to allocate %d J matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J_coord != NULL, "Failed to allocate %d J_coord auxiliary matrices\n", n_node);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]       = NULL;
        h2pack->J[i]       = NULL;
        h2pack->J_coord[i] = NULL;
    }
    H2P_dense_mat_t *U       = h2pack->U;
    H2P_int_vec_t   *J       = h2pack->J;
    H2P_dense_mat_t *J_coord = h2pack->J_coord;
    
    // 2. Construct U for nodes whose level is not smaller than min_adm_level.
    //    min_adm_level is the highest level that still has admissible blocks.
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_t A_block         = thread_buf[tid]->mat0;
        H2P_dense_mat_t node_skel_coord = thread_buf[tid]->mat1;
        H2P_dense_mat_t QR_buff         = thread_buf[tid]->mat2;
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
                H2P_dense_mat_init(&J_coord[node], xpt_dim, node_npt);
                H2P_copy_matrix_block(xpt_dim, node_npt, coord + pt_s, n_point, J_coord[node]->data, node_npt);
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
            H2P_dense_mat_resize(node_skel_coord, xpt_dim, J[node]->length);
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
                    H2P_copy_matrix_block(xpt_dim, src_ld, src_mat, src_ld, dst_mat, dst_ld);
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
            #ifdef H2_UJ_BUILD_RANDOMIZE
            // It seems that this part makes the calculation slower instead of faster
            if (A_blk_ncol > 2 * A_blk_nrow && krnl_dim >= 3)
            {
                H2P_dense_mat_t A_block1 = node_skel_coord;
                H2P_dense_mat_t rand_mat = QR_buff;
                H2P_dense_mat_resize(A_block1, A_blk_nrow, A_blk_nrow);
                H2P_dense_mat_resize(rand_mat, A_blk_ncol, A_blk_nrow);
                H2P_gen_normal_distribution(0.0, 1.0, A_blk_ncol * A_blk_nrow, rand_mat->data);
                CBLAS_GEMM(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, A_blk_nrow, A_blk_nrow, A_blk_ncol, 
                    1.0, A_block->data, A_block->ld, rand_mat->data, rand_mat->ld, 
                    0.0, A_block1->data,  A_block1->ld
                );
                H2P_dense_mat_resize(A_block, A_blk_nrow, A_blk_nrow);
                H2P_copy_matrix_block(A_blk_nrow, A_blk_nrow, A_block1->data, A_block1->ld, A_block->data, A_block->ld);
                H2P_dense_mat_normalize_columns(A_block, A_block1);
            }
            #endif

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
            H2P_dense_mat_init(&J_coord[node], xpt_dim, sub_idx->length);
            H2P_gather_matrix_columns(
                coord, n_point, J_coord[node]->data, J[node]->length, 
                xpt_dim, J[node]->data, J[node]->length
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
            mat_size[_U_SIZE_IDX]     += U[i]->nrow * U[i]->ncol;
            mat_size[_MV_FW_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[_MV_FW_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
            mat_size[_MV_BW_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[_MV_BW_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
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
    int    xpt_dim           = h2pack->xpt_dim;
    int    max_neighbor      = h2pack->max_neighbor;
    int    krnl_dim          = h2pack->krnl_dim;
    int    n_node            = h2pack->n_node;
    int    n_point           = h2pack->n_point;
    int    n_leaf_node       = h2pack->n_leaf_node;
    int    n_thread          = h2pack->n_thread;
    int    max_child         = h2pack->max_child;
    int    max_level         = h2pack->max_level;
    int    min_adm_level     = h2pack->HSS_min_adm_level;
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
    ASSERT_PRINTF(h2pack->U       != NULL, "Failed to allocate %d U matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J       != NULL, "Failed to allocate %d J matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J_coord != NULL, "Failed to allocate %d J_coord auxiliary matrices\n", n_node);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]       = NULL;
        h2pack->J[i]       = NULL;
        h2pack->J_coord[i] = NULL;
    }
    H2P_dense_mat_t *U       = h2pack->U;
    H2P_int_vec_t   *J       = h2pack->J;
    H2P_dense_mat_t *J_coord = h2pack->J_coord;

    double *U_timers = (double*) malloc(sizeof(double) * n_thread * 8);
    
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
        H2P_dense_mat_init(&J_coord[node], xpt_dim, node_npt);
        H2P_copy_matrix_block(xpt_dim, node_npt, coord + pt_s, n_point, J_coord[node]->data, node_npt);
    }

    // 3. Hierarchical construction level by level. min_adm_level is the 
    //    highest level that still has admissible blocks.
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

            double st, et, krnl_t = 0.0, randn_t = 0.0, gemm_t = 0.0, QR_t = 0.0, other_t = 0.0;
            
            thread_buf[tid]->timer -= get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int height = node_height[node];

                // (2) Calculate the number of skeleton points from all inadmissible nodes
                st = get_wtime_sec();
                int n_r_inadm = node_n_r_inadm[node];
                int inadm_skel_npt = 0;
                inadm_skel_idx->length = 0;
                if (n_r_inadm > 0)
                {
                    int *inadm_list = node_inadm_lists + node * max_neighbor;
                    for (int k = 0; k < n_r_inadm; k++)
                    {
                        int inadm_node_k = inadm_list[k];
                        ASSERT_PRINTF(
                            J[inadm_node_k] != NULL,
                            "Node %3d (level %2d): inadmissible node %3d (level %2d) skeleton point not ready!\n",
                            node, level, inadm_node_k, node_level[k]
                        );
                        H2P_int_vec_concatenate(inadm_skel_idx, J[inadm_node_k]);
                    }
                    inadm_skel_npt = inadm_skel_idx->length;
                }
                et = get_wtime_sec();
                other_t += et - st;
                
                // (3) Gather current node's skeleton points (== all children nodes' skeleton points)
                st = get_wtime_sec();
                H2P_dense_mat_resize(node_skel_coord, xpt_dim, J[node]->length);
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
                        H2P_copy_matrix_block(xpt_dim, src_ld, src_mat, src_ld, dst_mat, dst_ld);
                        J_child_size += J[i_child_node]->length;
                    }
                }
                et = get_wtime_sec();
                other_t += et - st;
                
                // (4) Shift points in this node so their center is at the original point
                st = get_wtime_sec();
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
                et = get_wtime_sec();
                other_t += et - st;

                // (5) Gather skeleton points from all inadmissible nodes and shift their centers to 
                //     the original point. We also put proxy points into inadm_skel_coord.
                st = get_wtime_sec();
                H2P_dense_mat_resize(inadm_skel_coord, xpt_dim, inadm_skel_npt + node_pp_npt);
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
                    if (xpt_dim > pt_dim)
                    {
                        for (int k = pt_dim; k < xpt_dim; k++)
                        {
                            DTYPE *coord_k = coord + k * n_point;
                            DTYPE *inadm_skel_coord_k = inadm_skel_coord->data + k * inadm_skel_coord->ld;
                            #pragma omp simd 
                            for (int l = 0; l < inadm_skel_npt; l++)
                                inadm_skel_coord_k[l] = coord_k[inadm_skel_idx->data[l]];
                        }
                    }
                }
                if (node_pp_npt > 0)
                {
                    H2P_copy_matrix_block(
                        xpt_dim, node_pp_npt, pp[level]->data, pp[level]->ld, 
                        inadm_skel_coord->data + inadm_skel_npt, inadm_skel_coord->ld
                    );
                }
                et = get_wtime_sec();
                other_t += et - st;

                // (6) Build the hybrid matrix block
                st = get_wtime_sec();
                int A_blk_nrow = node_skel_npt * krnl_dim;
                int A_blk_ncol = inadm_skel_coord->ncol * krnl_dim;
                H2P_dense_mat_resize(A_block, A_blk_nrow, A_blk_ncol);
                krnl_eval(
                    node_skel_coord->data,  node_skel_coord->ncol,  node_skel_coord->ld,
                    inadm_skel_coord->data, inadm_skel_coord->ncol, inadm_skel_coord->ld, 
                    krnl_param, A_block->data, A_block->ld
                );
                //H2P_dense_mat_normalize_columns(A_block, inadm_skel_coord);
                et = get_wtime_sec();
                krnl_t += et - st;
                if (A_blk_ncol > 2 * A_blk_nrow)
                {
                    st = get_wtime_sec();
                    H2P_dense_mat_t A_block1 = node_skel_coord;
                    H2P_dense_mat_t rand_mat = inadm_skel_coord;
                    H2P_dense_mat_resize(A_block1, A_blk_nrow, A_blk_nrow);
                    H2P_dense_mat_resize(rand_mat, A_blk_ncol, A_blk_nrow);
                    H2P_gen_normal_distribution(0.0, 1.0, A_blk_ncol * A_blk_nrow, rand_mat->data);
                    et = get_wtime_sec();
                    randn_t += et - st;

                    st = get_wtime_sec();
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, A_blk_nrow, A_blk_nrow, A_blk_ncol, 
                        1.0, A_block->data, A_block->ld, rand_mat->data, rand_mat->ld, 
                        0.0, A_block1->data,  A_block1->ld
                    );
                    H2P_dense_mat_resize(A_block, A_blk_nrow, A_blk_nrow);
                    H2P_copy_matrix_block(A_blk_nrow, A_blk_nrow, A_block1->data, A_block1->ld, A_block->data, A_block->ld);
                    et = get_wtime_sec();
                    gemm_t += et - st;
                }

                // (7) ID compress
                // Note: A is transposed in ID compress, be careful when calculating the buffer size
                st = get_wtime_sec();
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
                et = get_wtime_sec();
                QR_t += et - st;
                
                // (8) Choose the skeleton points of this node
                st = get_wtime_sec();
                for (int k = 0; k < sub_idx->length; k++)
                    J[node]->data[k] = J[node]->data[sub_idx->data[k]];
                J[node]->length = sub_idx->length;
                H2P_dense_mat_init(&J_coord[node], xpt_dim, sub_idx->length);
                H2P_gather_matrix_columns(
                    coord, n_point, J_coord[node]->data, J[node]->length, 
                    xpt_dim, J[node]->data, J[node]->length
                );
                et = get_wtime_sec();
                other_t += et - st;
            }  // End of j loop
            thread_buf[tid]->timer += get_wtime_sec();
            double *timers = U_timers + tid * 8;
            timers[_U_BUILD_KRNL_T_IDX]  = krnl_t;
            timers[_U_BUILD_RANDN_T_IDX] = randn_t;
            timers[_U_BUILD_GEMM_T_IDX]  = gemm_t;
            timers[_U_BUILD_QR_T_IDX]    = QR_t;
            timers[_U_BUILD_OTHER_T_IDX] = other_t;
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
        printf("[PROFILING] Build U: level %d, %d/%d threads, %d nodes, ", i, n_thread_i, n_thread, level_n_node[i]);
        printf("min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
        printf("[PROFILING] Build U subroutine time consumption:\n");
        printf("tid, kernel eval, randn gen, gemm, ID compress, misc, total\n");
        for (int tid = 0; tid < n_thread_i; tid++)
        {
            double *timers = U_timers + 8 * tid;
            printf(
                "%3d, %6.3lf, %6.3lf, %6.3lf, %6.3lf, %6.3lf, %6.3lf\n",
                tid, timers[_U_BUILD_KRNL_T_IDX], timers[_U_BUILD_RANDN_T_IDX], timers[_U_BUILD_GEMM_T_IDX], 
                timers[_U_BUILD_QR_T_IDX], timers[_U_BUILD_OTHER_T_IDX], thread_buf[tid]->timer
            );
        }
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
            mat_size[_U_SIZE_IDX]     += U[i]->nrow * U[i]->ncol;
            mat_size[_MV_FW_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[_MV_FW_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
            mat_size[_MV_BW_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[_MV_BW_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
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
    
    free(U_timers);

    for (int i = 0; i < n_thread; i++)
        H2P_thread_buf_reset(thread_buf[i]);
    BLAS_SET_NUM_THREADS(n_thread);
}

// Generate H2 generator matrices metadata
// Input parameter:
//   h2pack : H2Pack structure with H2 projection matrices
// Output parameter:
//   h2pack : H2Pack structure with H2 generator matrices metadata
void H2P_generate_B_metadata(H2Pack_t h2pack)
{
    int    BD_JIT          = h2pack->BD_JIT;
    int    krnl_dim        = h2pack->krnl_dim;
    int    n_node          = h2pack->n_node;
    int    n_thread        = h2pack->n_thread;
    int    krnl_bimv_flops = h2pack->krnl_bimv_flops;
    int    is_RPY_Ewald    = h2pack->is_RPY_Ewald;
    int    *node_level     = h2pack->node_level;
    int    *pt_cluster     = h2pack->pt_cluster;
    size_t *mat_size       = h2pack->mat_size;
    double *JIT_flops      = h2pack->JIT_flops;
    H2P_int_vec_t B_blk    = h2pack->B_blk;

    int n_r_adm_pair, *r_adm_pairs; 
    if (h2pack->is_HSS)
    {
        n_r_adm_pair = h2pack->HSS_n_r_adm_pair;
        r_adm_pairs  = h2pack->HSS_r_adm_pairs;
    } else {
        n_r_adm_pair = h2pack->n_r_adm_pair;
        r_adm_pairs  = h2pack->r_adm_pairs;
    }

    // 1. Allocate arrays for storing sizes and pair information of B matrices
    h2pack->n_B = n_r_adm_pair;
    h2pack->B_nrow = (int*)    malloc(sizeof(int)    * n_r_adm_pair);
    h2pack->B_ncol = (int*)    malloc(sizeof(int)    * n_r_adm_pair);
    h2pack->B_ptr  = (size_t*) malloc(sizeof(size_t) * (n_r_adm_pair + 1));
    int    *B_nrow = h2pack->B_nrow;
    int    *B_ncol = h2pack->B_ncol;
    size_t *B_ptr  = h2pack->B_ptr;
    ASSERT_PRINTF(
        h2pack->B_nrow != NULL && h2pack->B_ncol != NULL && h2pack->B_ptr != NULL,
        "Failed to allocate %d B matrices infomation array\n", n_r_adm_pair
    );

    int B_pair_cnt = 0;
    int *B_pair_i = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    int *B_pair_j = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    int *B_pair_v = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    ASSERT_PRINTF(
        B_pair_i != NULL && B_pair_j != NULL && B_pair_v != NULL,
        "Failed to allocate working buffer for B matrices indexing\n"
    );
    
    // 2. Generate size, pair, and statistic information of each B matrix
    B_ptr[0] = 0;
    size_t B_total_size = 0;
    H2P_int_vec_t *J = h2pack->J;
    h2pack->node_n_r_adm = (int*) malloc(sizeof(int) * n_node);
    ASSERT_PRINTF(
        h2pack->node_n_r_adm != NULL, 
        "Failed to allocate array of size %d for counting node admissible pairs\n", n_node
    );
    int *node_n_r_adm = h2pack->node_n_r_adm;
    memset(node_n_r_adm, 0, sizeof(int) * n_node);
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int node0  = r_adm_pairs[2 * i];
        int node1  = r_adm_pairs[2 * i + 1];
        int level0 = node_level[node0];
        int level1 = node_level[node1];
        node_n_r_adm[node0]++;
        if (is_RPY_Ewald == 0) node_n_r_adm[node1]++;
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
        B_pair_i[B_pair_cnt] = node0;
        B_pair_j[B_pair_cnt] = node1;
        B_pair_v[B_pair_cnt] = i + 1;
        B_pair_cnt++;
        if (is_RPY_Ewald == 0)
        {
            B_pair_i[B_pair_cnt] = node1;
            B_pair_j[B_pair_cnt] = node0;
            B_pair_v[B_pair_cnt] = -(i + 1);
            B_pair_cnt++;
        }
        // Add Statistic info
        mat_size[_MV_MID_SIZE_IDX]  += B_nrow[i] * B_ncol[i];
        mat_size[_MV_MID_SIZE_IDX]  += (B_nrow[i] + B_ncol[i]);
        JIT_flops[_JIT_B_FLOPS_IDX] += (double)(krnl_bimv_flops) * (double)(node0_npt * node1_npt);
        if (is_RPY_Ewald == 0) mat_size[_MV_MID_SIZE_IDX] += (B_nrow[i] + B_ncol[i]);
    }
        
    // 3. Partition B matrices into multiple blocks s.t. each block has approximately
    //    the same workload (total size of B matrices in a block)
    int BD_ntask_thread = (BD_JIT == 1) ? BD_NTASK_THREAD : 1;
    H2P_partition_workload(n_r_adm_pair, B_ptr + 1, B_total_size, n_thread * BD_ntask_thread, B_blk);
    for (int i = 1; i <= n_r_adm_pair; i++) B_ptr[i] += B_ptr[i - 1];
    mat_size[_B_SIZE_IDX] = B_total_size;

    // 4. Store pair-to-index relations in a CSR matrix for matvec, matmul, and SPDHSS construction
    h2pack->B_p2i_rowptr = (int*) malloc(sizeof(int) * (n_node + 1));
    h2pack->B_p2i_colidx = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    h2pack->B_p2i_val    = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    ASSERT_PRINTF(h2pack->B_p2i_rowptr != NULL, "Failed to allocate arrays for B matrices indexing\n");
    ASSERT_PRINTF(h2pack->B_p2i_colidx != NULL, "Failed to allocate arrays for B matrices indexing\n");
    ASSERT_PRINTF(h2pack->B_p2i_val    != NULL, "Failed to allocate arrays for B matrices indexing\n");
    H2P_int_COO_to_CSR(
        n_node, B_pair_cnt, B_pair_i, B_pair_j, B_pair_v, 
        h2pack->B_p2i_rowptr, h2pack->B_p2i_colidx, h2pack->B_p2i_val
    );
    free(B_pair_i);
    free(B_pair_j);
    free(B_pair_v);
}

// Build H2 generator matrices for AOT mode
// Input parameter:
//   h2pack : H2Pack structure with H2 generator matrices metadata
// Output parameter:
//   h2pack : H2Pack structure with H2 generator matrices
void H2P_build_B_AOT(H2Pack_t h2pack)
{
    int    krnl_dim    = h2pack->krnl_dim;
    int    n_point     = h2pack->n_point;
    int    n_thread    = h2pack->n_thread;
    int    *node_level = h2pack->node_level;
    int    *pt_cluster = h2pack->pt_cluster;
    size_t *B_ptr      = h2pack->B_ptr;
    DTYPE  *coord      = h2pack->coord;
    void   *krnl_param = h2pack->krnl_param;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    H2P_int_vec_t    B_blk       = h2pack->B_blk;
    H2P_dense_mat_t  *J_coord    = h2pack->J_coord;
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

    size_t B_total_size = h2pack->mat_size[_B_SIZE_IDX];
    h2pack->B_data = (DTYPE*) malloc_aligned(sizeof(DTYPE) * B_total_size, 64);
    ASSERT_PRINTF(h2pack->B_data != NULL, "Failed to allocate space for storing all %zu B matrices elements\n", B_total_size);
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

// Generate H2 dense blocks metadata
// Input parameter:
//   h2pack : H2Pack structure with H2 projection matrices
// Output parameter:
//   h2pack : H2Pack structure with H2 dense blocks metadata
void H2P_generate_D_metadata(H2Pack_t h2pack)
{
    int    BD_JIT          = h2pack->BD_JIT;
    int    krnl_dim        = h2pack->krnl_dim;
    int    n_node          = h2pack->n_node;
    int    n_thread        = h2pack->n_thread;
    int    n_leaf_node     = h2pack->n_leaf_node;
    int    krnl_bimv_flops = h2pack->krnl_bimv_flops;
    int    is_RPY_Ewald    = h2pack->is_RPY_Ewald;
    int    *leaf_nodes     = h2pack->height_nodes;
    int    *pt_cluster     = h2pack->pt_cluster;
    size_t *mat_size       = h2pack->mat_size;
    double *JIT_flops      = h2pack->JIT_flops;
    H2P_int_vec_t    D_blk0    = h2pack->D_blk0;
    H2P_int_vec_t    D_blk1    = h2pack->D_blk1;
    
    int n_r_inadm_pair, *r_inadm_pairs;
    if (h2pack->is_HSS)
    {
        n_r_inadm_pair = h2pack->HSS_n_r_inadm_pair;
        r_inadm_pairs  = h2pack->HSS_r_inadm_pairs;
    } else {
        n_r_inadm_pair = h2pack->n_r_inadm_pair;
        r_inadm_pairs  = h2pack->r_inadm_pairs;
    }

    // 1. Allocate arrays for storing sizes and pair information of D blocks
    h2pack->n_D = n_leaf_node + n_r_inadm_pair;
    h2pack->D_nrow = (int*)    malloc(sizeof(int)    * h2pack->n_D);
    h2pack->D_ncol = (int*)    malloc(sizeof(int)    * h2pack->n_D);
    h2pack->D_ptr  = (size_t*) malloc(sizeof(size_t) * (h2pack->n_D + 1));
    int    *D_nrow = h2pack->D_nrow;
    int    *D_ncol = h2pack->D_ncol;
    size_t *D_ptr  = h2pack->D_ptr;
    ASSERT_PRINTF(
        h2pack->D_nrow != NULL && h2pack->D_ncol != NULL && h2pack->D_ptr != NULL,
        "Failed to allocate %d D matrices infomation array\n", h2pack->n_D
    );

    int D_pair_cnt = 0;
    int n_Dij_pair = n_leaf_node + 2 * n_r_inadm_pair;
    int *D_pair_i = (int*) malloc(sizeof(int) * n_Dij_pair);
    int *D_pair_j = (int*) malloc(sizeof(int) * n_Dij_pair);
    int *D_pair_v = (int*) malloc(sizeof(int) * n_Dij_pair);
    ASSERT_PRINTF(
        D_pair_i != NULL && D_pair_j != NULL && D_pair_v != NULL,
        "Failed to allocate working buffer for D matrices indexing\n"
    );
    
    // 2. Generate size, pair, and statistic information of each D matrix
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
        D_pair_i[D_pair_cnt] = node;
        D_pair_j[D_pair_cnt] = node;
        D_pair_v[D_pair_cnt] = i + 1;
        D_pair_cnt++;
        // Add statistic info
        mat_size[_MV_DEN_SIZE_IDX]  += D_nrow[i] * D_ncol[i];
        mat_size[_MV_DEN_SIZE_IDX]  += D_nrow[i] + D_ncol[i];
        JIT_flops[_JIT_D_FLOPS_IDX] += (double)(krnl_bimv_flops) * (double)(node_npt * node_npt);
    }
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
        D_pair_i[D_pair_cnt] = node0;
        D_pair_j[D_pair_cnt] = node1;
        D_pair_v[D_pair_cnt] = ii + 1;
        D_pair_cnt++;
        if (is_RPY_Ewald == 0)
        {
            D_pair_i[D_pair_cnt] = node1;
            D_pair_j[D_pair_cnt] = node0;
            D_pair_v[D_pair_cnt] = -(ii + 1);
            D_pair_cnt++;
        }
        // Add statistic info
        mat_size[_MV_DEN_SIZE_IDX]  += D_nrow[ii] * D_ncol[ii];
        mat_size[_MV_DEN_SIZE_IDX]  += (D_nrow[ii] + D_ncol[ii]);
        JIT_flops[_JIT_D_FLOPS_IDX] += (double)(krnl_bimv_flops) * (double)(node0_npt * node1_npt);
        if (is_RPY_Ewald == 0) mat_size[_MV_DEN_SIZE_IDX]  += (D_nrow[ii] + D_ncol[ii]);
    }

    // 3. Partition D blocks into multiple blocks s.t. each block has approximately
    //    the same workload (total size of D blocks in a block)
    int BD_ntask_thread = (BD_JIT == 1) ? BD_NTASK_THREAD : 1;
    H2P_partition_workload(n_leaf_node,    D_ptr + 1,               D0_total_size, n_thread * BD_ntask_thread, D_blk0);
    H2P_partition_workload(n_r_inadm_pair, D_ptr + n_leaf_node + 1, D1_total_size, n_thread * BD_ntask_thread, D_blk1);
    for (int i = 1; i <= n_leaf_node + n_r_inadm_pair; i++) D_ptr[i] += D_ptr[i - 1];
    mat_size[_D_SIZE_IDX] = D0_total_size + D1_total_size;
    
    // 4. Store pair-to-index relations in a CSR matrix for matvec, matmul, and SPDHSS construction
    h2pack->D_p2i_rowptr = (int*) malloc(sizeof(int) * (n_node + 1));
    h2pack->D_p2i_colidx = (int*) malloc(sizeof(int) * n_Dij_pair);
    h2pack->D_p2i_val    = (int*) malloc(sizeof(int) * n_Dij_pair);
    ASSERT_PRINTF(h2pack->D_p2i_rowptr != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(h2pack->D_p2i_colidx != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(h2pack->D_p2i_val    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    H2P_int_COO_to_CSR(
        n_node, D_pair_cnt, D_pair_i, D_pair_j, D_pair_v, 
        h2pack->D_p2i_rowptr, h2pack->D_p2i_colidx, h2pack->D_p2i_val
    );
    free(D_pair_i);
    free(D_pair_j);
    free(D_pair_v);
}

// Build H2 dense blocks for AOT mode
// Input parameter:
//   h2pack : H2Pack structure with H2 dense blocks metadata
// Output parameter:
//   h2pack : H2Pack structure with H2 dense blocks
void H2P_build_D_AOT(H2Pack_t h2pack)
{
    int    krnl_dim    = h2pack->krnl_dim;
    int    n_thread    = h2pack->n_thread;
    int    n_point     = h2pack->n_point;
    int    n_leaf_node = h2pack->n_leaf_node;
    int    *leaf_nodes = h2pack->height_nodes;
    int    *pt_cluster = h2pack->pt_cluster;
    size_t *D_ptr      = h2pack->D_ptr;
    DTYPE  *coord      = h2pack->coord;
    void   *krnl_param = h2pack->krnl_param;
    double *JIT_flops  = h2pack->JIT_flops;
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

    size_t D_total_size = h2pack->mat_size[_D_SIZE_IDX];
    h2pack->D_data = (DTYPE*) malloc_aligned(sizeof(DTYPE) * D_total_size, 64);
    ASSERT_PRINTF(
        h2pack->D_data != NULL, 
        "Failed to allocate space for storing all %zu D matrices elements\n", D_total_size
    );
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
        ERROR_PRINTF("You need to provide a set of proxy points.\n");
        return;
    }
    
    if (krnl_eval == NULL)
    {
        ERROR_PRINTF("You need to provide a valid krnl_eval().\n");
        return;
    }

    h2pack->pp = pp;
    h2pack->BD_JIT = BD_JIT;
    h2pack->krnl_param = krnl_param;
    h2pack->krnl_eval  = krnl_eval;
    h2pack->krnl_bimv  = krnl_bimv;
    h2pack->krnl_bimv_flops = krnl_bimv_flops;
    if (BD_JIT == 1 && krnl_bimv == NULL) 
        WARNING_PRINTF("krnl_eval() will be used in BD_JIT matvec. For better performance, consider using a krnl_bimv().\n");

    // 1. Build projection matrices and skeleton row sets
    st = get_wtime_sec();
    if (h2pack->is_HSS) H2P_build_HSS_UJ_hybrid(h2pack);
    else H2P_build_H2_UJ_proxy(h2pack);
    et = get_wtime_sec();
    h2pack->timers[_U_BUILD_TIMER_IDX] = et - st;

    // 2. Build generator matrices
    st = get_wtime_sec();
    H2P_generate_B_metadata(h2pack);
    if (BD_JIT == 0) H2P_build_B_AOT(h2pack);
    et = get_wtime_sec();
    h2pack->timers[_B_BUILD_TIMER_IDX] = et - st;
    
    // 3. Build dense blocks
    st = get_wtime_sec();
    H2P_generate_D_metadata(h2pack);
    if (BD_JIT == 0) H2P_build_D_AOT(h2pack);
    et = get_wtime_sec();
    h2pack->timers[_D_BUILD_TIMER_IDX] = et - st;
}

