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
#include "H2Pack_gen_proxy_point.h"
#include "H2Pack_utils.h"
#include "utils.h"

#define _GEN_PP_KRNL_T_IDX  0
#define _GEN_PP_SPMM_T_IDX  1
#define _GEN_PP_ID_T_IDX    2
#define _GEN_PP_MISC_T_IDX  3

// Generate proxy points with two domains specified as 
// X = [-L1/2, L1/2]^pt_dim, Y = [-L3/2, L3/2]^pt_dim \ [-L2/2, L2/2]^pt_dim
// generated proxy points are in domain Y
// Input parameters:
//   pt_dim     : Dimension of point coordinate
//   krnl_dim   : Dimension of kernel's return
//   reltol     : Proxy point selection relative error tolerance
//   krnl_param : Pointer to kernel function parameter array
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   X0_size    : Number of candidate points in X
//   Y0_lsize   : Number of candidate points in Y per layer
//   L1, L2, L3 : Box sizes of X and Y
// Output parameters:
//   pp     : Generated proxy points, pp should have been initialized before this function
//   timers : Size 4, timers for different parts 
void H2P_generate_proxy_point_nlayer(
    const int pt_dim, const int krnl_dim, const DTYPE reltol, 
    const void *krnl_param, kernel_eval_fptr krnl_eval, 
    const int X0_size, const int Y0_lsize, const DTYPE L1, const DTYPE L2, const DTYPE L3, 
    H2P_dense_mat_t pp, DTYPE *timers
)
{
    // 1. Initialize working arrays and parameters
    double st, et;
    int n_thread = omp_get_max_threads();
    int n_layer  = DROUND((L3 - L2) / L1);
    int Y0_size  = n_layer * Y0_lsize;
    H2P_dense_mat_t X0_coord, Y0_coord, tmp_coord, Yp_coord;
    H2P_dense_mat_t tmpA, min_dist, QR_buff;
    H2P_int_vec_t   skel_idx, ID_buff;
    st = get_wtime_sec();
    H2P_dense_mat_init(&X0_coord,  pt_dim,  X0_size);
    H2P_dense_mat_init(&Y0_coord,  pt_dim,  Y0_size);
    H2P_dense_mat_init(&tmp_coord, pt_dim,  Y0_size);
    H2P_dense_mat_init(&Yp_coord,  pt_dim,  Y0_size);
    H2P_dense_mat_init(&tmpA,      X0_size * krnl_dim, Y0_size * krnl_dim);
    H2P_dense_mat_init(&min_dist,  X0_size, 1);
    H2P_dense_mat_init(&QR_buff,   2 * Y0_size, 1);
    H2P_int_vec_init(&skel_idx, X0_size);
    H2P_int_vec_init(&ID_buff,  4 * Y0_size);
    et = get_wtime_sec();
    timers[_GEN_PP_MISC_T_IDX] += et - st;
    
    // 2. Generate initial candidate points in X and Y
    //    For Y0, we generate it layer by layer. Each layer has the same number of candidate 
    //    points but different volume. Therefore a inner layer has a higher point density. 
    st = get_wtime_sec();
    H2P_gen_coord_in_ring(X0_size, pt_dim, 0.0, L1, X0_coord->data, X0_coord->ld);
    DTYPE Y0_layer_width = (L3 - L2) / (DTYPE) n_layer;
    for (int i = 0; i < n_layer; i++)
    {
        DTYPE layer_L0 = L2 + Y0_layer_width * (DTYPE) i;
        DTYPE layer_L1 = L2 + Y0_layer_width * (DTYPE) (i + 1);
        H2P_gen_coord_in_ring(Y0_lsize, pt_dim, layer_L0, layer_L1, Y0_coord->data + i * Y0_lsize, Y0_coord->ld);
    }
    et = get_wtime_sec();
    timers[_GEN_PP_MISC_T_IDX] += et - st;

    // 3. Select skeleton points in domain X
    //    Use sparsity + randomize to reduce the ID cost
    // (1) Generate the kernel matrix
    st = get_wtime_sec();
    H2P_eval_kernel_matrix_OMP(krnl_param, krnl_eval, krnl_dim, X0_coord, Y0_coord, tmpA);
    et = get_wtime_sec();
    timers[_GEN_PP_KRNL_T_IDX] += et - st;
    // (2) Generate sparse random matrix and multiply with the kernel matrix to get a reduced matrix
    H2P_int_vec_t   rndmat_idx = ID_buff;
    H2P_dense_mat_t rndmat_val = QR_buff;
    H2P_dense_mat_t tmpA1      = min_dist;
    st = get_wtime_sec();
    int max_nnz_col = 32;
    H2P_gen_rand_sparse_mat_trans(max_nnz_col, tmpA->ncol, tmpA->nrow, rndmat_val, rndmat_idx);
    H2P_dense_mat_resize(tmpA1, tmpA->nrow, tmpA->nrow);
    H2P_calc_sparse_mm_trans(
        tmpA->nrow, tmpA->nrow, tmpA->ncol, rndmat_val, rndmat_idx,
        tmpA->data, tmpA->ld, tmpA1->data, tmpA1->ld
    );
    et = get_wtime_sec();
    timers[_GEN_PP_KRNL_T_IDX] += et - st;
    // (3) Calculate ID approximation on the reduced matrix and select skeleton points in X
    st = get_wtime_sec();
    if (krnl_dim == 1)
    {
        H2P_dense_mat_resize(QR_buff, tmpA1->nrow, 1);
    } else {
        int QR_buff_size = (2 * krnl_dim + 2) * tmpA1->ncol + (krnl_dim + 1) * tmpA1->nrow;
        H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
    }
    H2P_int_vec_set_capacity(ID_buff, 4 * tmpA1->nrow);
    et = get_wtime_sec();
    timers[_GEN_PP_MISC_T_IDX] += et - st;

    st = get_wtime_sec();
    DTYPE reltol_ = reltol * 1e-2;
    H2P_ID_compress(
        tmpA1, QR_REL_NRM, &reltol_, NULL, skel_idx, 
        n_thread, QR_buff->data, ID_buff->data, krnl_dim
    );
    et = get_wtime_sec();
    timers[_GEN_PP_ID_T_IDX]   += et - st;

    st = get_wtime_sec();
    H2P_dense_mat_select_columns(X0_coord, skel_idx);
    H2P_dense_mat_t Xp_coord = X0_coord;
    et = get_wtime_sec();
    timers[_GEN_PP_MISC_T_IDX] += et - st;

    // 4. Select proxy points in Y layer by layer
    H2P_dense_mat_resize(Yp_coord, pt_dim, 0);
    for (int i = 0; i < n_layer; i++)
    {
        // (1) Put selected proxy points and i-th layer candidate points together
        st = get_wtime_sec();
        H2P_dense_mat_resize(tmp_coord, pt_dim, Yp_coord->ncol + Y0_lsize);
        DTYPE *Yp_coord_ptr     = Yp_coord->data;
        DTYPE *Y0_layer_i_ptr   = Y0_coord->data + i * Y0_lsize;
        DTYPE *tmp_coord_Yp_ptr = tmp_coord->data;
        DTYPE *tmp_coord_li_ptr = tmp_coord->data + Yp_coord->ncol;
        H2P_copy_matrix_block(pt_dim, Yp_coord->ncol, Yp_coord_ptr,   Yp_coord->ld, tmp_coord_Yp_ptr, tmp_coord->ld);
        H2P_copy_matrix_block(pt_dim, Y0_lsize,       Y0_layer_i_ptr, Y0_coord->ld, tmp_coord_li_ptr, tmp_coord->ld);
        et = get_wtime_sec();
        timers[_GEN_PP_MISC_T_IDX] += et - st;

        // (2) Generate kernel matrix for this layer 
        st = get_wtime_sec();
        // Be careful, tmp_coord should be placed before Xp_coord
        H2P_eval_kernel_matrix_OMP(krnl_param, krnl_eval, krnl_dim, tmp_coord, Xp_coord, tmpA1);
        et = get_wtime_sec();

        // (3) Calculate ID approximation on the new kernel matrix and select new proxy points in Y
        st = get_wtime_sec();
        if (krnl_dim == 1)
        {
            H2P_dense_mat_resize(QR_buff, tmpA1->nrow, 1);
        } else {
            int QR_buff_size = (2 * krnl_dim + 2) * tmpA1->ncol + (krnl_dim + 1) * tmpA1->nrow;
            H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
        }
        H2P_int_vec_set_capacity(ID_buff, 4 * tmpA1->nrow);
        et = get_wtime_sec();
        timers[_GEN_PP_MISC_T_IDX] += et - st;

        st = get_wtime_sec();
        DTYPE reltol2 = reltol * 1e-2;
        H2P_ID_compress(
            tmpA1, QR_REL_NRM, &reltol2, NULL, skel_idx, 
            n_thread, QR_buff->data, ID_buff->data, krnl_dim
        );
        et = get_wtime_sec();
        timers[_GEN_PP_ID_T_IDX]   += et - st;

        st = get_wtime_sec();
        H2P_dense_mat_select_columns(tmp_coord, skel_idx);
        H2P_dense_mat_resize(Yp_coord, pt_dim, tmp_coord->ncol);
        H2P_copy_matrix_block(pt_dim, tmp_coord->ncol, tmp_coord->data, tmp_coord->ld, Yp_coord->data, Yp_coord->ld);
        et = get_wtime_sec();
        timers[_GEN_PP_MISC_T_IDX] += et - st;
    }  // End of i loop

    // 5. Increase the density of selected proxy points if necessary
    if (reltol >= 1e-12) 
    {
        // No need to increase the density, just copy it
        H2P_dense_mat_resize(pp, pt_dim, Yp_coord->ncol);
        H2P_copy_matrix_block(pt_dim, Yp_coord->ncol, Yp_coord->data, Yp_coord->ld, pp->data, pp->ld);
    } else {
        int Yp_size = Yp_coord->ncol;
        H2P_dense_mat_resize(min_dist, Yp_size, 1);
        DTYPE *coord_i = tmpA->data;
        for (int i = 0; i < Yp_size; i++) min_dist->data[i] = 1e99;
        for (int i = 0; i < Yp_size; i++)
        {
            for (int k = 0; k < pt_dim; k++)
                coord_i[k] = Yp_coord->data[i + k * Yp_coord->ld];
            
            for (int j = 0; j < i; j++)
            {
                DTYPE dist_ij = 0.0;
                for (int k = 0; k < pt_dim; k++)
                {
                    DTYPE diff = coord_i[k] - Yp_coord->data[j + k * Yp_coord->ld];
                    dist_ij += diff * diff;
                }
                dist_ij = DSQRT(dist_ij);
                min_dist->data[i] = MIN(min_dist->data[i], dist_ij);
                min_dist->data[j] = MIN(min_dist->data[j], dist_ij);
            }
        }

        const int Yp_size2 = Yp_size * 2;
        H2P_dense_mat_resize(pp, pt_dim, Yp_size2);
        for (int i = 0; i < Yp_size; i++)
        {
            DTYPE *tmp_coord0 = tmpA->data;
            DTYPE *tmp_coord1 = tmpA->data + pt_dim;
            for (int j = 0; j < pt_dim; j++)
                tmp_coord0[j] = Yp_coord->data[i + j * Yp_coord->ld];
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
                if ((H2P_point_in_box(pt_dim, tmp_coord1, L2) == 0) &&
                    (H2P_point_in_box(pt_dim, tmp_coord1, L3) == 1)) flag = 0;
            }  // End of "while (flag == 1)"

            DTYPE *coord_0 = pp->data + (2 * i);
            DTYPE *coord_1 = pp->data + (2 * i + 1);
            for (int j = 0; j < pt_dim; j++)
            {
                coord_0[j * Yp_size2] = tmp_coord0[j];
                coord_1[j * Yp_size2] = tmp_coord1[j];
            }
        }  // End of i loop
    }  // End of "if (reltol >= 1e-12)"

    // 6. Free working arrays
    H2P_dense_mat_destroy(X0_coord);
    H2P_dense_mat_destroy(Y0_coord);
    H2P_dense_mat_destroy(tmp_coord);
    H2P_dense_mat_destroy(Yp_coord);
    H2P_dense_mat_destroy(tmpA);
    H2P_dense_mat_destroy(min_dist);
    H2P_dense_mat_destroy(QR_buff);
    H2P_int_vec_destroy(skel_idx);
    H2P_int_vec_destroy(ID_buff);
    free(X0_coord);
    free(Y0_coord);
    free(tmp_coord);
    free(Yp_coord);
    free(tmpA);
    free(min_dist);
    free(QR_buff);
    free(skel_idx);
    free(ID_buff);
}

// Generate proxy points for constructing H2 projection and skeleton matrices
// using ID compress for any kernel function
void H2P_generate_proxy_point_ID(
    const int pt_dim, const int krnl_dim, const DTYPE reltol, const int max_level, const int min_level,
    DTYPE max_L, const void *krnl_param, kernel_eval_fptr krnl_eval, H2P_dense_mat_t **pp_
)
{
    // 1. Initialize proxy point arrays and parameters
    int n_level = max_level + 1;
    H2P_dense_mat_t *pp = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_level);
    ASSERT_PRINTF(pp != NULL, "Failed to allocate %d arrays for storing proxy points", n_level);
    for (int i = 0; i <= max_level; i++) 
    {
        H2P_dense_mat_init(&pp[i], pt_dim, 0);
        pp[i]->ncol = 0;
    }
    
    int X0_size = 2000, Y0_lsize = 4000;
    char *X0_size_p  = getenv("H2P_GEN_PP_X0_SIZE");
    char *Y0_lsize_p = getenv("H2P_GEN_PP_Y0_LSIZE");
    if (X0_size_p != NULL)
    {
        int X0_size0 = X0_size;
        X0_size = atoi(X0_size_p);
        if (X0_size < 500 || X0_size > 5000) X0_size = 2000;
        INFO_PRINTF("Overriding parameter %s : %d (default) --> %d (new)\n", "X0_size", X0_size0, X0_size);
    }
    if (Y0_lsize_p != NULL)
    {
        int Y0_lsize0 = Y0_lsize;
        Y0_lsize = atoi(Y0_lsize_p);
        if (Y0_lsize < 1000 || Y0_lsize > 20000) Y0_lsize = 4000;
        INFO_PRINTF("Overriding parameter %s : %d (default) --> %d (new)\n", "Y0_lsize", Y0_lsize0, Y0_lsize);
    }
    double timers[4];

    // 2. Construct proxy points on each level
    DTYPE pow_2_level = 0.5;
    for (int level = 0; level < min_level; level++) pow_2_level *= 2.0;
    for (int level = min_level; level <= max_level; level++)
    {
        // Level 0 and level 1 nodes are not admissible, do not need proxy points
        if (level < 2)
        {
            pow_2_level *= 2.0;
            WARNING_PRINTF("Level %d: no proxy points are generated\n", level);
            continue;
        }

        // Decide box sizes for domains X and Y
        pow_2_level *= 2.0;
        DTYPE L1   = max_L / pow_2_level;
        DTYPE L2   = (1.0 + 2.0 * ALPHA_H2) * L1;
        DTYPE L3_0 = (1.0 + 8.0 * ALPHA_H2) * L1;
        DTYPE L3_1 = 2.0 * max_L - L1;
        DTYPE L3   = MIN(L3_0, L3_1);
        
        // Reset timers
        timers[_GEN_PP_KRNL_T_IDX] = 0.0;
        timers[_GEN_PP_KRNL_T_IDX] = 0.0;
        timers[_GEN_PP_ID_T_IDX]   = 0.0;
        timers[_GEN_PP_MISC_T_IDX] = 0.0;

        // Generate proxy points
        H2P_generate_proxy_point_nlayer(
            pt_dim, krnl_dim, reltol, krnl_param, krnl_eval, 
            X0_size, Y0_lsize, L1, L2, L3, pp[level], &timers[0]
        );
        
        #ifdef PROFILING_OUTPUT
        INFO_PRINTF("Level %d: %d proxy points generated\n", level, pp[level]->ncol);
        INFO_PRINTF(
            "    kernel, SpMM, ID, other time = %.3lf, %.3lf, %.3lf, %.3lf sec\n", 
            timers[_GEN_PP_KRNL_T_IDX], timers[_GEN_PP_KRNL_T_IDX], 
            timers[_GEN_PP_ID_T_IDX],   timers[_GEN_PP_MISC_T_IDX]
        );
        #endif
    }  // End of level loop
    
    *pp_ = pp;
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

