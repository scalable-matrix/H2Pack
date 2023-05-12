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

typedef enum
{
    GEN_PP_KRNL_TIMER_IDX = 0,
    GEN_PP_SPMM_TIMER_IDX,
    GEN_PP_ID_TIMER_IDX,
    GEN_PP_MISC_TIMER_IDX
} gen_pp_timer_idx_t;

struct H2P_gen_pp_param_
{
    int alg;            // Algorithm for selecting Yp proxy points
                        // 0 : Uniform candidate point distribution, one QR
                        // 1 : Nonuniform candidate point distribution, one QR
                        // 2 : Nonuniform candidate point distribution, multiple QR
    int X0_size;        // Number of candidate points in X
    int Y0_lsize;       // Number of candidate points in Y per layer
    int L3_nlayer;      // Y box exterior boundary size factor
    int max_layer;      // Maximum number of layers in domain Y
    int print_timers;   // If we need to print internal timings
};
static struct H2P_gen_pp_param_ gen_pp_param;

// Generate proxy points with two domains specified as 
// X = [-L1/2, L1/2]^pt_dim, Y = [-L3/2, L3/2]^pt_dim \ [-L2/2, L2/2]^pt_dim
// generated proxy points are in domain Y
// Input parameters:
//   pt_dim     : Dimension of point coordinate
//   krnl_dim   : Dimension of kernel's return
//   reltol     : Proxy point selection relative error tolerance
//   krnl_param : Pointer to kernel function parameter array
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   L1, L2, L3 : Box sizes of X and Y
//   alg        : Algorithm for selecting Yp proxy points
//   X0_size    : Number of candidate points in X
//   Y0_lsize   : Number of candidate points in Y per layer
//   max_layer  : Maximum number of layers in domain Y
// Output parameters:
//   pp     : Generated proxy points, pp should have been initialized 
//   timers : Size 4, timers for different parts 
void H2P_generate_proxy_point_nlayer(
    const int pt_dim, const int krnl_dim, const DTYPE reltol, 
    const void *krnl_param, kernel_eval_fptr krnl_eval, 
    const DTYPE L1, const DTYPE L2, const DTYPE L3, 
    const int alg, const int X0_size, const int Y0_lsize, const int max_layer, 
    H2P_dense_mat_p pp, double *timers
)
{
    // 1. Initialize working arrays and parameters
    double st, et;
    int n_thread = omp_get_max_threads();
    int n_layer  = (alg == 0) ? 1 : DROUND((L3 - L2) / L1);
    if (n_layer > max_layer) n_layer = max_layer;
    int Y0_size  = n_layer * Y0_lsize;
    H2P_dense_mat_p X0_coord, Y0_coord, tmp_coord, Yp_coord;
    H2P_dense_mat_p tmpA, min_dist, QR_buff;
    H2P_int_vec_p   skel_idx, ID_buff;
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
    timers[GEN_PP_MISC_TIMER_IDX] += et - st;
    
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
    timers[GEN_PP_MISC_TIMER_IDX] += et - st;

    // 3. Select skeleton points in domain X
    //    Use sparsity + randomize to reduce the ID cost
    // (1) Generate the kernel matrix
    st = get_wtime_sec();
    H2P_eval_kernel_matrix_OMP(krnl_param, krnl_eval, krnl_dim, X0_coord, Y0_coord, tmpA, n_thread);
    et = get_wtime_sec();
    timers[GEN_PP_KRNL_TIMER_IDX] += et - st;
    // (2) Generate sparse random matrix and multiply with the kernel matrix to get a reduced matrix
    H2P_int_vec_p   rndmat_idx = ID_buff;
    H2P_dense_mat_p rndmat_val = QR_buff;
    H2P_dense_mat_p tmpA1      = min_dist;
    st = get_wtime_sec();
    int max_nnz_col = 32;
    H2P_gen_rand_sparse_mat_trans(max_nnz_col, tmpA->ncol, tmpA->nrow, rndmat_val, rndmat_idx);
    H2P_dense_mat_resize(tmpA1, tmpA->nrow, tmpA->nrow);
    H2P_calc_sparse_mm_trans(
        tmpA->nrow, tmpA->nrow, tmpA->ncol, rndmat_val, rndmat_idx,
        tmpA->data, tmpA->ld, tmpA1->data, tmpA1->ld
    );
    et = get_wtime_sec();
    timers[GEN_PP_KRNL_TIMER_IDX] += et - st;
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
    timers[GEN_PP_MISC_TIMER_IDX] += et - st;

    st = get_wtime_sec();
    DTYPE reltol_ = reltol * 1e-2;
    H2P_ID_compress(
        tmpA1, QR_REL_NRM, &reltol_, NULL, skel_idx, 
        n_thread, QR_buff->data, ID_buff->data, krnl_dim
    );
    et = get_wtime_sec();
    timers[GEN_PP_ID_TIMER_IDX]   += et - st;

    st = get_wtime_sec();
    H2P_dense_mat_select_columns(X0_coord, skel_idx);
    H2P_dense_mat_p Xp_coord = X0_coord;
    et = get_wtime_sec();
    timers[GEN_PP_MISC_TIMER_IDX] += et - st;

    if (alg == 0 || alg == 1)
    {
        // 4. Select proxy points in domain Y
        // (1) Generate the kernel matrix
        st = get_wtime_sec();
        // Be careful, Y0_coord should be placed before Xp_coord
        H2P_eval_kernel_matrix_OMP(krnl_param, krnl_eval, krnl_dim, Y0_coord, Xp_coord, tmpA1, n_thread);
        et = get_wtime_sec();
        timers[GEN_PP_KRNL_TIMER_IDX] += et - st;
        // (2) Calculate ID approximation on the kernel matrix and select new proxy points in Y
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
        timers[GEN_PP_MISC_TIMER_IDX] += et - st;

        st = get_wtime_sec();
        DTYPE reltol2 = reltol * 1e-2;
        H2P_ID_compress(
            tmpA1, QR_REL_NRM, &reltol2, NULL, skel_idx, 
            n_thread, QR_buff->data, ID_buff->data, krnl_dim
        );
        et = get_wtime_sec();
        timers[GEN_PP_ID_TIMER_IDX]   += et - st;

        st = get_wtime_sec();
        H2P_dense_mat_select_columns(Y0_coord, skel_idx);
        H2P_dense_mat_resize(Yp_coord, pt_dim, Y0_coord->ncol);
        copy_matrix(sizeof(DTYPE), pt_dim, Y0_coord->ncol, Y0_coord->data, Y0_coord->ld, Yp_coord->data, Yp_coord->ld, 0);
        et = get_wtime_sec();
        timers[GEN_PP_MISC_TIMER_IDX] += et - st;
    }  // End of "if (alg == 1)"

    if (alg == 2)
    {
        // 4. Select proxy points in domain Y layer by layer
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
            copy_matrix(sizeof(DTYPE), pt_dim, Yp_coord->ncol, Yp_coord_ptr,   Yp_coord->ld, tmp_coord_Yp_ptr, tmp_coord->ld, 0);
            copy_matrix(sizeof(DTYPE), pt_dim, Y0_lsize,       Y0_layer_i_ptr, Y0_coord->ld, tmp_coord_li_ptr, tmp_coord->ld, 0);
            et = get_wtime_sec();
            timers[GEN_PP_MISC_TIMER_IDX] += et - st;

            // (2) Generate kernel matrix for this layer 
            st = get_wtime_sec();
            // Be careful, tmp_coord should be placed before Xp_coord
            H2P_eval_kernel_matrix_OMP(krnl_param, krnl_eval, krnl_dim, tmp_coord, Xp_coord, tmpA1, n_thread);
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
            timers[GEN_PP_MISC_TIMER_IDX] += et - st;

            st = get_wtime_sec();
            DTYPE reltol2 = reltol * 1e-2;
            H2P_ID_compress(
                tmpA1, QR_REL_NRM, &reltol2, NULL, skel_idx, 
                n_thread, QR_buff->data, ID_buff->data, krnl_dim
            );
            et = get_wtime_sec();
            timers[GEN_PP_ID_TIMER_IDX]   += et - st;

            st = get_wtime_sec();
            H2P_dense_mat_select_columns(tmp_coord, skel_idx);
            H2P_dense_mat_resize(Yp_coord, pt_dim, tmp_coord->ncol);
            copy_matrix(sizeof(DTYPE), pt_dim, tmp_coord->ncol, tmp_coord->data, tmp_coord->ld, Yp_coord->data, Yp_coord->ld, 0);
            et = get_wtime_sec();
            timers[GEN_PP_MISC_TIMER_IDX] += et - st;
        }  // End of i loop
    }  // End of "if (alg == 2)"

    // 5. Increase the density of selected proxy points if necessary
    if (reltol >= 1e-12) 
    {
        // No need to increase the density, just copy it
        H2P_dense_mat_resize(pp, pt_dim, Yp_coord->ncol);
        copy_matrix(sizeof(DTYPE), pt_dim, Yp_coord->ncol, Yp_coord->data, Yp_coord->ld, pp->data, pp->ld, 0);
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
    H2P_dense_mat_destroy(&X0_coord);
    H2P_dense_mat_destroy(&Y0_coord);
    H2P_dense_mat_destroy(&tmp_coord);
    H2P_dense_mat_destroy(&Yp_coord);
    H2P_dense_mat_destroy(&tmpA);
    H2P_dense_mat_destroy(&min_dist);
    H2P_dense_mat_destroy(&QR_buff);
    H2P_int_vec_destroy(&skel_idx);
    H2P_int_vec_destroy(&ID_buff);
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

// ----- Note: "radius" in this file == 0.5 * length of a cubic box ----- //

// Calculate the enclosing box of a given set of points and adjust it if the proxy point file is provided
void H2P_calc_enclosing_box(const int pt_dim, const int n_point, const DTYPE *coord, const char *fname, DTYPE **enbox_)
{
    // Calculate the center of points in this box
    DTYPE *center = (DTYPE*) malloc(sizeof(DTYPE) * pt_dim);
    memset(center, 0, sizeof(DTYPE) * pt_dim);
    for (int j = 0; j < pt_dim; j++)
    {
        const DTYPE *coord_dim_j = coord + j * n_point;
        for (int i = 0; i < n_point; i++)
            center[j] += coord_dim_j[i];
    }
    for (int j = 0; j < pt_dim; j++) center[j] /= (DTYPE) n_point;

    // Calculate enclosing box radius
    DTYPE radius = 0.0;
    for (int j = 0; j < pt_dim; j++)
    {
        const DTYPE *coord_dim_j = coord + j * n_point;
        DTYPE center_j = center[j];
        for (int i = 0; i < n_point; i++)
        {
            DTYPE tmp = DABS(coord_dim_j[i] - center_j);
            radius = MAX(radius, tmp);
        }
    }

    // Adjust enclosing box radius if proxy point file is available
    FILE *inf = NULL;
    if (fname != NULL) inf = fopen(fname, "r");
    if (inf != NULL)
    {
        int pt_dim_, L3_nlayer, num_pp;
        DTYPE reltol, minL;
        const char *fmt_str = (DTYPE_SIZE == 8) ? "%d %lf %d %lf %d" : "%d %f %d %f %d";
        fscanf(inf, fmt_str, &pt_dim_, &reltol, &L3_nlayer, &minL, &num_pp);
        if (pt_dim == pt_dim_)
        {
            DTYPE k = DCEIL(DLOG2(radius / minL));
            radius = minL * DPOW(2.0, k);
        } else {
            ERROR_PRINTF("File %s point dimension (%d) != current point dimension (%d)\n", fname, pt_dim_, pt_dim);
        }
    }
    if (inf != NULL) fclose(inf);

    // Generate the enclosing box
    DTYPE *enbox = (DTYPE*) malloc(sizeof(DTYPE) * 2 * pt_dim);
    for (int j = 0; j < pt_dim; j++)
    {
        enbox[j] = center[j] - radius;
        enbox[pt_dim + j] = 2 * radius;
    }
    *enbox_ = enbox;
    free(center);
}

// Write a set of proxy points to a text file
void H2P_write_proxy_point_file(
    const char *fname, const int pt_dim, const DTYPE reltol, const int L3_nlayer, 
    const DTYPE minL, const int num_pp, H2P_dense_mat_p *pp
)
{
    FILE *ouf = fopen(fname, "w");

    // Line 1: parameters
    fprintf(ouf, "%d %.3e %d %16.12f %d\n", pt_dim, reltol, L3_nlayer, minL, num_pp);

    // Line 2: number of proxy points in each proxy point set
    for (int i = 0; i < num_pp; i++) fprintf(ouf, "%d ", pp[i]->ncol);
    fprintf(ouf, "\n");

    // Rest part: proxy point coordinates
    for (int i_pp = 0; i_pp < num_pp; i_pp++)
    {
        DTYPE *pp_i_coord = pp[i_pp]->data;
        const int pp_i_npt = pp[i_pp]->ncol; 
        const int pp_i_ld  = pp[i_pp]->ld;
        for (int i = 0; i < pp_i_npt; i++)
        {
            for (int j = 0; j < pt_dim; j++) fprintf(ouf, "% 16.12f  ", pp_i_coord[j * pp_i_ld + i]);
            fprintf(ouf, "\n");
        }
    }

    fclose(ouf);
}


// Generate proxy points for constructing H2 projection and skeleton matrices using 
// ID compress, also try to load proxy points from a file and update this file
void H2P_generate_proxy_point_ID_file(
    H2Pack_p h2pack, const void *krnl_param, kernel_eval_fptr krnl_eval, 
    const char *fname, H2P_dense_mat_p **pp_
)
{
    int   pt_dim   = h2pack->pt_dim;
    int   krnl_dim = h2pack->krnl_dim;
    int   n_level  = h2pack->max_level + 1;
    DTYPE reltol   = h2pack->QR_stop_tol;
    DTYPE pt_maxL  = h2pack->root_enbox[pt_dim] * 0.5;
    DTYPE pt_minL  = pt_maxL * DPOW(0.5, (DTYPE) h2pack->max_level);
    
    // Root box and level 1 box do not have admissible pairs --> don't need proxy points
    pt_maxL *= 0.25;
    
    // These are from proxy point file
    int pt_dim0, L3_nlayer0, num_pp0;
    DTYPE reltol0, minL0, maxL0;

    GET_ENV_INT_VAR(gen_pp_param.alg,          "H2P_GEN_PP_ALG",       "alg",          2,    0,    2);
    GET_ENV_INT_VAR(gen_pp_param.X0_size,      "H2P_GEN_PP_X0_SIZE",   "X0_size",      2000, 500,  5000);
    GET_ENV_INT_VAR(gen_pp_param.Y0_lsize,     "H2P_GEN_PP_Y0_LSIZE",  "Y0_lsize",     4000, 1000, 20000);
    GET_ENV_INT_VAR(gen_pp_param.L3_nlayer,    "H2P_GEN_PP_L3_NLAYER", "L3_nlayer",    8,    8,    32);
    GET_ENV_INT_VAR(gen_pp_param.max_layer,    "H2P_GEN_PP_MAX_LAYER", "max_layer",    8,    4,    32);
    GET_ENV_INT_VAR(gen_pp_param.print_timers, "H2P_PRINT_TIMERS",     "print_timers", 0,    0,    1);

    // Determine min & max box radius in the file & for current points
    FILE *inf = NULL;
    if (fname != NULL) inf = fopen(fname, "r");
    if (inf != NULL)
    {
        const char *fmt_str = (DTYPE_SIZE == 8) ? "%d %lf %d %lf %d" : "%d %f %d %f %d";
        fscanf(inf, fmt_str, &pt_dim0, &reltol0, &L3_nlayer0, &minL0, &num_pp0);
        maxL0 = minL0 * DPOW(2.0, (DTYPE) (num_pp0 - 1));
        DTYPE k  = DLOG2(pt_maxL / minL0);
        DTYPE rk = (DTYPE) DROUND(k);
        int aligned_L = (DABS(rk - k) > 1e-10) ? 0 : 1;
        if (pt_dim0 != pt_dim || reltol0 > reltol || L3_nlayer0 != gen_pp_param.L3_nlayer || aligned_L == 0)
        {
            WARNING_PRINTF("Proxy point file parameters are inconsistent with current point set parameters, calculate all proxy points\n");
            pt_dim0    = pt_dim;
            L3_nlayer0 = gen_pp_param.L3_nlayer;
            num_pp0    = 0;
            reltol0    = reltol;
            minL0      = pt_minL;
            maxL0      = pt_minL * 0.5;  // Make maxL0 invalid
        }
    } else {
        pt_dim0    = pt_dim;
        L3_nlayer0 = gen_pp_param.L3_nlayer;
        num_pp0    = 0;
        reltol0    = reltol;
        minL0      = pt_minL;
        maxL0      = pt_minL * 0.5;  // Make maxL0 invalid
    }  // End of "if (inf != NULL)"
    DTYPE curr_minL   = MIN(pt_minL, minL0);
    DTYPE curr_maxL   = MAX(pt_maxL, maxL0);
    int   curr_num_pp = DROUND(DLOG2(curr_maxL / curr_minL)) + 1;
    int   file_idx_s  = DROUND(DLOG2(minL0     / curr_minL));
    int   file_idx_e  = DROUND(DLOG2(maxL0     / curr_minL));
    int   pt_idx_s    = DROUND(DLOG2(pt_minL   / curr_minL));
    int   pt_idx_e    = DROUND(DLOG2(pt_maxL   / curr_minL));

    if (h2pack->print_dbginfo)
    {
        DEBUG_PRINTF(
            "pt_minL, pt_maxL, minL0, maxL0, curr_minL, curr_maxL = %.3lf  %.3lf  %.3lf  %.3lf  %.3lf  %.3lf\n", 
            pt_minL, pt_maxL, minL0, maxL0, curr_minL, curr_maxL
        );
        DEBUG_PRINTF(
            "curr_num_pp, file_idx_s, file_idx_e, pt_idx_s, pt_idx_e = %d  %d  %d  %d  %d\n", 
            curr_num_pp, file_idx_s, file_idx_e, pt_idx_s, pt_idx_e
        );
    }

    // Note: radius of pp0[i] == 0.5 * radius of pp0[i+1], need to reverse it for pp_
    H2P_dense_mat_p *pp0 = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * curr_num_pp);
    ASSERT_PRINTF(pp0 != NULL, "Failed to allocate %d arrays for storing proxy points", curr_num_pp);
    for (int i = 0; i < curr_num_pp; i++) 
    {
        H2P_dense_mat_init(&pp0[i], pt_dim, 0);
        pp0[i]->ncol = 0;
    }

    // Read proxy points from file
    if (inf != NULL)
    {
        int *pp_sizes = (int*) malloc(sizeof(int) * num_pp0);
        for (int i = 0; i < num_pp0; i++) fscanf(inf, "%d", pp_sizes + i);
        for (int pp_i = file_idx_s; pp_i <= file_idx_e; pp_i++)
        {
            H2P_dense_mat_p pp0_i = pp0[pp_i];
            int pp0_i_npt = pp_sizes[pp_i - file_idx_s];
            H2P_dense_mat_resize(pp0_i, pt_dim, pp0_i_npt);
            DTYPE *pp0_i_coord = pp0_i->data;
            for (int j = 0; j < pp0_i_npt; j++)
            {
                for (int i = 0; i < pt_dim; i++) 
                    fscanf(inf, DTYPE_FMTSTR, &pp0_i_coord[i * pp0_i_npt + j]);
            }
        }
        free(pp_sizes);
        fclose(inf);
    }  // End of if (inf != NULL)

    // Calculate other proxy points
    double timers[4];
    DTYPE L3_nlayer_ = (DTYPE) gen_pp_param.L3_nlayer;
    timers[GEN_PP_KRNL_TIMER_IDX] = 0.0;
    timers[GEN_PP_KRNL_TIMER_IDX] = 0.0;
    timers[GEN_PP_ID_TIMER_IDX]   = 0.0;
    timers[GEN_PP_MISC_TIMER_IDX] = 0.0;
    for (int pp_i = 0; pp_i < curr_num_pp; pp_i++)
    {
        // Note: curr_minL is the radius, L1 is the edge length, need to * 2
        DTYPE L1 = 2.0 * curr_minL * DPOW(2.0, (DTYPE) pp_i);
        DTYPE L2 = (1.0 + 2.0 * ALPHA_H2) * L1;
        DTYPE L3 = (1.0 + L3_nlayer_ * ALPHA_H2) * L1;
        int Y0_lsize_ = gen_pp_param.Y0_lsize;
        if (gen_pp_param.alg == 0)  // Only one ring, multiple Y0_lsize_ by the number of rings
        {
            int n_layer = DROUND((L3 - L2) / L1);
            if (n_layer > gen_pp_param.max_layer) n_layer = gen_pp_param.max_layer;
            Y0_lsize_ *= n_layer;
        }

        if (pp_i >= file_idx_s && pp_i <= file_idx_e) continue;

        H2P_generate_proxy_point_nlayer(
            pt_dim, krnl_dim, reltol, 
            krnl_param, krnl_eval, 
            L1, L2, L3, 
            gen_pp_param.alg, gen_pp_param.X0_size, Y0_lsize_, gen_pp_param.max_layer, 
            pp0[pp_i], &timers[0]
        );
    }  // End of pp_i loop
    if (gen_pp_param.print_timers == 1)
    {
        INFO_PRINTF(
            "Proxy point generation: kernel, SpMM, ID, other time = %.3lf, %.3lf, %.3lf, %.3lf sec\n", 
            timers[GEN_PP_KRNL_TIMER_IDX], timers[GEN_PP_KRNL_TIMER_IDX], 
            timers[GEN_PP_ID_TIMER_IDX],   timers[GEN_PP_MISC_TIMER_IDX]
        );
    }

    // Write current proxy points to file
    if (fname != NULL) H2P_write_proxy_point_file(fname, pt_dim, reltol0, L3_nlayer0, curr_minL, curr_num_pp, pp0);

    // Copy pp0 to output pp_ and free pp0
    H2P_dense_mat_p *pp = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_level);
    ASSERT_PRINTF(pp != NULL, "Failed to allocate %d arrays for storing proxy points", n_level);
    for (int i = 0; i < n_level; i++) 
    {
        H2P_dense_mat_init(&pp[i], pt_dim, 0);
        pp[i]->ncol = 0;
    }
    for (int i = pt_idx_s; i <= pt_idx_e; i++)
    {
        int level = (n_level - 1) - (i - pt_idx_s);
        H2P_dense_mat_resize(pp[level], pp0[i]->nrow, pp0[i]->ncol);
        copy_matrix(sizeof(DTYPE), pp0[i]->nrow, pp0[i]->ncol, pp0[i]->data, pp0[i]->ld, pp[level]->data, pp[level]->ld, 1);
    }
    for (int i = 0; i < curr_num_pp; i++) H2P_dense_mat_destroy(&pp0[i]);
    free(pp0);
    *pp_ = pp;
}

// Generate uniformly distributed proxy points on a box surface for constructing
// H2 projection and skeleton matrices for SOME kernel function
void H2P_generate_proxy_point_surface(
    const int pt_dim, const int xpt_dim, const int min_npt, const int max_level, 
    const int min_level, DTYPE max_L, H2P_dense_mat_p **pp_
)
{
    if (pt_dim < 2 || pt_dim > 3)
    {
        ERROR_PRINTF("Only 2D and 3D systems are supported in this function.\n");
        return;
    }
    
    H2P_dense_mat_p *pp = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * (max_level + 1));
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
    H2P_dense_mat_p unit_pp;
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
    
    H2P_dense_mat_destroy(&unit_pp);
    *pp_ = pp;
}

