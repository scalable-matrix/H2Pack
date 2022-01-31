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
#include "H2Pack_build_with_sample_point.h"
#include "H2Pack_ID_compress.h"
#include "H2Pack_utils.h"
#include "utils.h"

// Calculate the size of the minimal enclosing box for a point cluster
// Input parameters:
//   pt_dim : Point dimensions
//   npt    : Number of points in the cluster
//   coord  : Size pt_dim * npt, each column is a point coordinate
// Output parameter:
//   enbox_size : Size pt_dim, size of the minimal enclosing box for the given point cluster
void H2P_calc_enclosing_box_size(const int pt_dim, const int npt, DTYPE *coord, DTYPE *enbox_size)
{
    for (int i = 0; i < pt_dim; i++)
    {
        DTYPE *coord_i = coord + i * npt;
        DTYPE max_c = coord_i[0], min_c = coord_i[0];
        for (int j = 1; j < npt; j++)
        {
            if (coord_i[j] > max_c) max_c = coord_i[j];
            if (coord_i[j] < min_c) min_c = coord_i[j];
        }
        enbox_size[i] = max_c - min_c;
    }
}

// Decompose an integer into the sum of multiple integers which are approximately proportional
// to another floating point array.
// Input parameters:
//   nelem      : Number of integers after decomposition
//   decomp_sum : The target number to be decomposed
//   prop       : Size 2 * nelem, the first nelem values are the proportions
// Output parameter:
//   decomp : Decomposed values, decomp[i] ~= prop[i] / sum(prop[0:nelem-1]) * decomp_sum
int H2P_proportional_int_decompose(const int nelem, const int decomp_sum, DTYPE *prop, int *decomp)
{
    DTYPE sum_prop = 0.0;
    DTYPE *decomp_prop = prop + nelem;
    for (int i = 0; i < nelem; i++) sum_prop += prop[i];
    int decomp_sum0 = 0;
    for (int i = 0; i < nelem; i++)
    {
        decomp_prop[i] = (DTYPE) decomp_sum * prop[i] / sum_prop;
        decomp[i] = (int) DFLOOR(decomp_prop[i]);
        decomp_sum0 += decomp[i];
    }
    for (int k = decomp_sum0; k < decomp_sum; k++)
    {
        // Add 1 to the position that got hit most by floor
        int min_idx = 0;
        DTYPE max_diff = decomp_prop[0] - (DTYPE) decomp[0];
        for (int i = 1; i < nelem; i++)
        {
            DTYPE diff = decomp_prop[i] - (DTYPE) decomp[i];
            if (diff > max_diff)
            {
                max_diff = diff;
                min_idx = i;
            }
        }
        decomp[min_idx]++;
    }
    int prod1 = 1;
    for (int i = 0; i < nelem; i++) prod1 *= (decomp[i] + 1);
    return prod1;
}

// Compute the squared pairwise distance of two point sets
// Input parameters:
//   coord{0, 1} : Size pt_dim * ld{0, 1}, point set coordinates, each column is a coordinate
//   ld{0, 1}    : Leading dimension of coord{0, 1}
//   n{0, 1}     : Number of points in coord{0, 1}
//   pt_dim      : Point dimension
//   ldd         : Leading dimension of dist2
// Output parameter:
//   dist2 : Size n0 * ldd, squared distance
void H2P_calc_pdist2(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const int pt_dim, DTYPE *dist2, const int ldd
)
{
    for (int i = 0; i < n0; i++)
    {
        const DTYPE *coord0_i = coord0 + i;
        DTYPE *dist2_i = dist2 + i * ldd;
        // This implementation makes GCC auto-vectorization happy
        memset(dist2_i, 0, sizeof(DTYPE) * n1);
        for (int k = 0; k < pt_dim; k++)
        {
            const DTYPE coord0_k_i = coord0_i[k * ld0];
            const DTYPE *coord1_k = coord1 + k * ld1;
            #pragma omp simd
            for (int j = 0; j < n1; j++)
            {
                DTYPE diff = coord0_k_i - coord1_k[j];
                dist2_i[j] += diff * diff;
            }
        }
    }
}

void H2P_select_anchor_grid(
    const int pt_dim, const DTYPE *coord_min, const DTYPE *coord_max, const DTYPE *enbox_size, 
    const int *grid_size, const int grid_algo, H2P_dense_mat_p anchor_coord_
)
{
    int max_grid_size = 0;
    int anchor_npt = 1;
    for (int i = 0; i < pt_dim; i++)
    {
        max_grid_size = (grid_size[i] > max_grid_size) ? grid_size[i] : max_grid_size;
        anchor_npt *= (grid_size[i] + 1);
    }
    max_grid_size++;

    DTYPE *anchor_dim = (DTYPE *) malloc(sizeof(DTYPE) * pt_dim * max_grid_size);
    H2P_dense_mat_resize(anchor_coord_, pt_dim, anchor_npt);
    DTYPE *anchor_coord = anchor_coord_->data;

    // 1. Assign anchor points in each dimension
    for (int i = 0; i < pt_dim; i++)
    {
        DTYPE *anchor_dim_i = anchor_dim + i * max_grid_size;
        if (grid_size[i] == 0) anchor_dim_i[0] = (coord_min[i] + coord_max[i]) * 0.5;
    }
    if (grid_algo == 2)  // Chebyshev anchor points
    {
        for (int i = 0; i < pt_dim; i++)
        {
            if (grid_size[i] == 0) continue;
            DTYPE *anchor_dim_i = anchor_dim + i * max_grid_size;
            DTYPE s0 = 0.5 * (coord_max[i] + coord_min[i]);
            DTYPE s1 = 0.5 * (coord_max[i] - coord_min[i]);
            DTYPE s2 = M_PI / (2.0 * (DTYPE) grid_size[i] + 2);
            for (int j = 0; j <= grid_size[i]; j++)
            {
                DTYPE v0 = 2.0 * (DTYPE) j + 1.0;
                DTYPE v1 = DCOS(v0 * s2);
                anchor_dim_i[j] = s0 + s1 * v1;
            }
        }
    }
    DTYPE c0, c1, c2;
    if (grid_algo == 6)
    {
        c0 = 1.0;
        c1 = 0.5;
        c2 = 0.25;
    }
    if (grid_algo >= 4)
    {
        for (int i = 0; i < pt_dim; i++)
        {
            DTYPE size_i = c0 * enbox_size[i] / ((DTYPE) grid_size[i] + c1);
            DTYPE *anchor_dim_i = anchor_dim + i * max_grid_size;
            for (int j = 0; j <= grid_size[i]; j++)
                anchor_dim_i[j] = coord_min[i] + c2 * size_i + size_i * (DTYPE) j;
        }
    }

    // 2. Do a tensor product to get all anchor points
    int *stride = (int *) malloc(sizeof(int) * (pt_dim + 1));
    stride[0] = 1;
    for (int i = 0; i < pt_dim; i++) stride[i + 1] = stride[i] * (grid_size[i] + 1);
    for (int i = 0; i < anchor_npt; i++)
    {
        for (int j = 0; j < pt_dim; j++)
        {
            int dim_idx = (i / stride[j]) % (grid_size[j] + 1);
            anchor_coord[j * anchor_npt + i] = anchor_dim[j * max_grid_size + dim_idx];
        }
    }

    free(stride);
    free(anchor_dim);
}

// For each point in the generated anchor grid, choose O(1) nearest points
// in the given point cluster as sample points
// Input parameters:
//   pt_dim    : Point dimension
//   coord     : Size pt_dim * npt, each column is a point coordinate
//   grid_size : Size pt_dim, anchor grid size
//   grid_algo : Anchor grid generation algorithm
//   n_thread  : Number of threads to use
// Output parameters:
//   sample_idx : Indices of the chosen sample points
void H2P_select_cluster_sample(
    const int pt_dim, H2P_dense_mat_p coord, const int *grid_size, const int grid_algo, 
    const int n_thread, H2P_int_vec_p sample_idx
)
{
    int npt = coord->ncol;
    H2P_dense_mat_p anchor_coord_;
    H2P_dense_mat_init(&anchor_coord_, pt_dim, 1024);
    DTYPE *coord_min  = (DTYPE *) malloc(sizeof(DTYPE) * pt_dim);
    DTYPE *coord_max  = (DTYPE *) malloc(sizeof(DTYPE) * pt_dim);
    DTYPE *enbox_size = (DTYPE *) malloc(sizeof(DTYPE) * pt_dim);
    for (int i = 0; i < pt_dim; i++)
    {
        DTYPE *coord_i = coord->data + i * npt;
        DTYPE max_c = coord_i[0], min_c = coord_i[0];
        for (int j = 1; j < npt; j++)
        {
            if (coord_i[j] > max_c) max_c = coord_i[j];
            if (coord_i[j] < min_c) min_c = coord_i[j];
        }
        coord_min[i]  = min_c;
        coord_max[i]  = max_c;
        enbox_size[i] = max_c - min_c;
    }
    H2P_select_anchor_grid(
        pt_dim, coord_min, coord_max, enbox_size, 
        grid_size, grid_algo, anchor_coord_
    );
    free(coord_min);
    free(coord_max);
    free(enbox_size);
    DTYPE *anchor_coord = anchor_coord_->data;
    
    // Choose nearest points in the given point cluster
    int anchor_npt = anchor_coord_->ncol;
    DTYPE *dist2_buf = (DTYPE *) malloc(sizeof(DTYPE) * n_thread * npt);
    int *sample_idx_flag = (int *) malloc(sizeof(int) * npt);
    for (int i = 0; i < npt; i++) sample_idx_flag[i] = -1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        DTYPE *thread_dist2 = dist2_buf + tid * npt;
        #pragma omp for
        for (int i = 0; i < anchor_npt; i++)
        {
            H2P_calc_pdist2(
                anchor_coord + i, anchor_npt, 1, 
                coord->data, npt, npt, 
                pt_dim, thread_dist2, npt
            );
            DTYPE min_dist2 = thread_dist2[0];
            int min_idx = 0;
            for (int j = 1; j < npt; j++)
            {
                if (min_dist2 > thread_dist2[j])
                {
                    min_dist2 = thread_dist2[j];
                    min_idx = j;
                }
            }
            sample_idx_flag[min_idx] = 1;
        }
    }
    H2P_int_vec_set_capacity(sample_idx, npt);
    int sample_idx_cnt = 0;
    for (int i = 0; i < npt; i++)
    {
        if (sample_idx_flag[i] != -1)
        {
            sample_idx->data[sample_idx_cnt] = i;
            sample_idx_cnt++;
        }
    }
    sample_idx->length = sample_idx_cnt;
    free(sample_idx_flag);
    free(dist2_buf);
    H2P_dense_mat_destroy(&anchor_coord_);
}

int H2P_select_sample_point_r(
    H2Pack_p h2pack, const void *krnl_param, kernel_eval_fptr krnl_eval, 
    const DTYPE tau, const DTYPE reltol
)
{
    ASSERT_PRINTF(h2pack->pt_dim == h2pack->xpt_dim, "Sample point algorithm does not support RPY with different radii yet\n");

    int k        = h2pack->max_leaf_points * 2;
    int pt_dim   = h2pack->pt_dim;
    int krnl_dim = h2pack->krnl_dim;
    int n_thread = h2pack->n_thread;
    DTYPE *root_enbox = h2pack->root_enbox;
    DTYPE L = root_enbox[pt_dim];
    for (int i = 1; i < pt_dim; i++) 
        L = (L > root_enbox[pt_dim + i]) ? root_enbox[pt_dim + i] : L;
    L /= 6.0;  // Don't know why we need this, just copy from the MATLAB code

    DTYPE *c1_enbox_size = (DTYPE *) malloc(sizeof(DTYPE) * pt_dim * 2);
    int *r_list = (int *) malloc(sizeof(int) * pt_dim);
    H2P_dense_mat_p coord0, coord1, coord1s, A0, A1, U, UA, QR_buff;
    H2P_int_vec_p sample_idx, sub_idx, ID_buff;
    H2P_dense_mat_init(&coord0,  pt_dim, k);
    H2P_dense_mat_init(&coord1,  pt_dim, k);
    H2P_dense_mat_init(&coord1s, pt_dim, k);
    H2P_dense_mat_init(&A0, k * krnl_dim, k * krnl_dim);
    H2P_dense_mat_init(&A1, k * krnl_dim, k * krnl_dim);
    H2P_dense_mat_init(&U,  k * krnl_dim, k * krnl_dim);
    H2P_dense_mat_init(&UA, k * krnl_dim, k * krnl_dim);
    H2P_dense_mat_init(&QR_buff, 1, 8192);
    H2P_int_vec_init(&sample_idx, 1024);
    H2P_int_vec_init(&ID_buff, 1024);
    sub_idx = sample_idx;

    int stop_type = QR_REL_NRM;
    DTYPE stop_tol = reltol * 1e-4;
    if (stop_tol < 1e-15) stop_tol = 1e-15;
    void *stop_param = (void *) &stop_tol;

    // Create two sets of points in unit boxes
    DTYPE coord1_shift = L / tau;
    for (int i = 0; i < k * pt_dim; i++)
    {
        coord0->data[i] = L * (DTYPE) drand48();
        coord1->data[i] = L * (DTYPE) drand48() + coord1_shift;
    }

    // Find an r value by checking approximation error to A
    // The r initial guess formula is provided by Difeng
    int r1 = floor(-1.25 * log10(stop_tol) - 0.25);
    if (r1 < 1) r1 = 1;
    int r2 = ceil(sqrt(floor(2.0 * log10(stop_tol) / log10(0.7))));
    int r = (r1 < r2) ? r1 : r2;
    int flag = 0;
    DTYPE A0_fnorm = 0.0;
    krnl_eval(
        coord0->data, k, k,
        coord1->data, k, k, 
        krnl_param, A0->data, A0->ld
    );
    #pragma omp parallel for reduction(+: A0_fnorm)
    for (int i = 0; i < A0->nrow * A0->ncol; i++)
        A0_fnorm += A0->data[i] * A0->data[i];
    A0_fnorm = DSQRT(A0_fnorm);
    while (flag == 0)
    {
        H2P_calc_enclosing_box_size(pt_dim, k, coord1->data, c1_enbox_size);
        H2P_proportional_int_decompose(pt_dim, r, c1_enbox_size, r_list);

        // sample_idx = H2_select_cluster_sample(pt_dim, coord2, r_list, 6);
        // coord2_samples = coord2(sample_idx, :);
        H2P_select_cluster_sample(pt_dim, coord1, r_list, 6, n_thread, sample_idx);
        H2P_dense_mat_copy(coord1, coord1s);
        H2P_dense_mat_select_columns(coord1s, sample_idx);

        // A1 = kernel({coord1, coord2_samples});
        int n_sample = sample_idx->length;
        H2P_dense_mat_resize(A1, k * krnl_dim, n_sample * krnl_dim);
        krnl_eval(
            coord0->data, k, k,
            coord1s->data, n_sample, n_sample, 
            krnl_param, A1->data, A1->ld
        );

        // ID compress of A1
        if (krnl_dim == 1)
        {
            H2P_dense_mat_resize(QR_buff, A1->nrow, 1);
        } else {
            int QR_buff_size = (2 * krnl_dim + 2) * A1->ncol + (krnl_dim + 1) * A1->nrow;
            H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
        }
        H2P_int_vec_set_capacity(ID_buff, 4 * A1->nrow);
        H2P_ID_compress(
            A1, stop_type, stop_param, &U, sub_idx, 
            n_thread, QR_buff->data, ID_buff->data, krnl_dim
        );

        // Copy A0(sub_idx, :) to A1
        H2P_dense_mat_resize(A1, sub_idx->length * krnl_dim, k * krnl_dim);
        #pragma omp parallel for
        for (int i = 0; i < sub_idx->length; i++)
        {
            for (int j = 0; j < krnl_dim; j++)
            {
                int A1_irow = i * krnl_dim + j;
                int A0_irow = sub_idx->data[i] * krnl_dim + j;
                memcpy(A1->data + A1_irow * A1->ld, A0->data + A0_irow * A0->ld, sizeof(DTYPE) * k * krnl_dim);
            }
        }

        // UA = U * A(subidx, :);
        H2P_dense_mat_resize(UA, U->nrow, A0->ncol);
        CBLAS_GEMM(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, A0->nrow, A0->ncol, U->ncol,
            1.0, U->data, U->ld, A1->data, A1->ld, 0, UA->data, UA->ld
        );

        // err = norm(UA - A, 'fro') / norm(A, 'fro');
        DTYPE err_fnorm = 0.0, relerr;
        #pragma omp parallel for reduction(+: err_fnorm)
        for (int i = 0; i < A0->nrow * A0->ncol; i++)
        {
            DTYPE diff = DABS(A0->data[i] - UA->data[i]);
            err_fnorm += diff * diff;
        }
        err_fnorm = DSQRT(err_fnorm);
        relerr = err_fnorm / A0_fnorm;
        if ( (relerr < reltol * 0.1) || ((k - n_sample) < (k / 10)) ) flag = 1;
        else r++;
    }  // End "while (flag == 0)"

    free(c1_enbox_size);
    free(r_list);
    H2P_dense_mat_destroy(&coord0);
    H2P_dense_mat_destroy(&coord1);
    H2P_dense_mat_destroy(&coord1s);
    H2P_dense_mat_destroy(&A0);
    H2P_dense_mat_destroy(&A1);
    H2P_dense_mat_destroy(&U);
    H2P_dense_mat_destroy(&UA);
    H2P_dense_mat_destroy(&QR_buff);
    H2P_int_vec_destroy(&sample_idx);
    H2P_int_vec_destroy(&ID_buff);
    return r;
}

// Select sample points for constructing H2 projection and skeleton matrices 
void H2P_select_sample_point_ref(
    H2Pack_p h2pack, const void *krnl_param, kernel_eval_fptr krnl_eval, 
    const DTYPE tau, H2P_dense_mat_p **sample_points_
)
{
    int   pt_dim        = h2pack->pt_dim;
    int   xpt_dim       = h2pack->xpt_dim;
    int   min_adm_level = h2pack->is_HSS ? h2pack->HSS_min_adm_level : h2pack->min_adm_level;
    int   max_level     = h2pack->max_level;
    int   max_child     = h2pack->max_child;
    int   n_point       = h2pack->n_point;
    int   n_node        = h2pack->n_node;
    int   n_leaf_node   = h2pack->n_leaf_node;
    int   n_r_adm_pair  = h2pack->is_HSS ? h2pack->HSS_n_r_adm_pair : h2pack->n_r_adm_pair;
    int   *level_n_node = h2pack->level_n_node;
    int   *level_nodes  = h2pack->level_nodes;
    int   *n_child      = h2pack->n_child;
    int   *children     = h2pack->children;
    int   *pt_cluster   = h2pack->pt_cluster;
    int   *r_adm_pairs  = h2pack->is_HSS ? h2pack->HSS_r_adm_pairs : h2pack->r_adm_pairs;
    DTYPE *coord        = h2pack->coord;

    // 1. Allocate temporary arrays and output arrays
    H2P_int_vec_p   *adm_list   = (H2P_int_vec_p *)   malloc(sizeof(H2P_int_vec_p)   * n_node);
    H2P_dense_mat_p *clu_refine = (H2P_dense_mat_p *) malloc(sizeof(H2P_dense_mat_p) * n_node);
    H2P_dense_mat_p *sample_pt  = (H2P_dense_mat_p *) malloc(sizeof(H2P_dense_mat_p) * n_node);
    for (int i = 0; i < n_node; i++)
    {
        H2P_int_vec_init(adm_list + i, n_node);
        H2P_dense_mat_init(clu_refine + i, 300, xpt_dim);
        H2P_dense_mat_init(sample_pt + i, 300, xpt_dim);
    }

    // 2. Convert the reduced admissible pairs to reduced admissible list of each node
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int c0 = r_adm_pairs[2 * i + 0];
        int c1 = r_adm_pairs[2 * i + 1];
        H2P_int_vec_push_back(adm_list[c0], c1);
        H2P_int_vec_push_back(adm_list[c1], c0);
    }

    int ri = H2P_select_sample_point_r(h2pack, krnl_param, krnl_eval, tau, h2pack->QR_stop_tol);

    // 3. Bottom-up sweep
    int r_up = ri;
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];

        // (1) Update refined points associated with clusters at level i
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            int n_child_node = n_child[node];
            if (n_child_node == 0)
            {
                int pt_s = pt_cluster[node * 2];
                int pt_e = pt_cluster[node * 2 + 1];
                int node_npt = pt_e - pt_s + 1;
                H2P_dense_mat_resize(clu_refine[node], xpt_dim, node_npt);
                copy_matrix_block(sizeof(DTYPE), xpt_dim, node_npt, coord + pt_s, n_point, clu_refine[node]->data, node_npt);
            } else {
                int *child_nodes = children + node * max_child;
                int child_clu_sum = 0;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    int i_child_npt  = clu_refine[i_child_node]->ncol;
                    child_clu_sum += i_child_npt;
                }
                H2P_dense_mat_resize(clu_refine[node], xpt_dim, child_clu_sum);
                child_clu_sum = 0;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    int i_child_npt  = clu_refine[i_child_node]->ncol;
                    int lds = i_child_npt;
                    int ldd = clu_refine[node]->ncol;
                    DTYPE *src = clu_refine[i_child_node]->data;
                    DTYPE *dst = clu_refine[node]->data + child_clu_sum;
                    copy_matrix_block(sizeof(DTYPE), xpt_dim, i_child_npt, src, lds, dst, ldd);
                    child_clu_sum += i_child_npt;
                }
            }  // End of "if (n_child_node == 0)"
        }  // End of j loop

        // (2) Select refined points for all nodes at i-th level
        #pragma omp parallel
        {
            int *ri_list = (int *) malloc(sizeof(int) * pt_dim);
            H2P_int_vec_p sample_idx;
            H2P_int_vec_init(&sample_idx, 1024);
            DTYPE *enbox_size = (DTYPE *) malloc(sizeof(DTYPE) * 2 * pt_dim);
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int node_npt_refine = clu_refine[node]->ncol;
                if (node_npt_refine <= 5) continue;
                H2P_calc_enclosing_box_size(pt_dim, clu_refine[node]->ncol, clu_refine[node]->data, enbox_size);
                int ri_npt = H2P_proportional_int_decompose(pt_dim, r_up, enbox_size, ri_list);
                if (ri_npt < node_npt_refine)
                {
                    H2P_select_cluster_sample(pt_dim, clu_refine[node], ri_list, 6,  1, sample_idx);
                    H2P_dense_mat_select_columns(clu_refine[node], sample_idx);
                }
            }  // End of j loop
            H2P_int_vec_destroy(&sample_idx);
            free(ri_list);
            free(enbox_size);
        }
    }  // End of i loop
    
    // 4. Top-down sweep
    int r_down;
    if (ri > 3) r_down = (ri + 3 > 10) ? (ri + 3) : 10;
    else r_down = ri + 3;
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];

        #pragma omp parallel
        {
            int *ri_list = (int *) malloc(sizeof(int) * pt_dim);
            H2P_int_vec_p sample_idx;
            H2P_int_vec_init(&sample_idx, 1024);
            DTYPE *enbox_size = (DTYPE *) malloc(sizeof(DTYPE) * 2 * pt_dim);
            H2P_dense_mat_p workbuf_d;
            H2P_dense_mat_init(&workbuf_d, 1, 1024);
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                if (adm_list[node]->length == 0) continue;

                // (1) Gather all far-field refined points as initial sample points
                int sample_npt0 = sample_pt[node]->ncol;
                H2P_dense_mat_resize(workbuf_d, xpt_dim, sample_npt0);
                copy_matrix_block(sizeof(DTYPE), xpt_dim, sample_npt0, sample_pt[node]->data, sample_npt0, workbuf_d->data, sample_npt0);
                int init_sample_npt = sample_npt0;
                for (int k = 0; k < adm_list[node]->length; k++)
                {
                    int pair_node = adm_list[node]->data[k];
                    int pair_node_npt = clu_refine[pair_node]->ncol;
                    init_sample_npt += pair_node_npt;
                }
                H2P_dense_mat_resize(sample_pt[node], xpt_dim, init_sample_npt);
                copy_matrix_block(sizeof(DTYPE), xpt_dim, sample_npt0, workbuf_d->data, sample_npt0, sample_pt[node]->data, init_sample_npt);
                init_sample_npt = sample_npt0;
                for (int k = 0; k < adm_list[node]->length; k++)
                {
                    int pair_node = adm_list[node]->data[k];
                    int pair_node_npt = clu_refine[pair_node]->ncol;
                    int lds = pair_node_npt;
                    int ldd = sample_pt[node]->ncol;
                    DTYPE *src = clu_refine[pair_node]->data;
                    DTYPE *dst = sample_pt[node]->data + init_sample_npt;
                    copy_matrix_block(sizeof(DTYPE), xpt_dim, pair_node_npt, src, lds, dst, ldd);
                    init_sample_npt += pair_node_npt;
                }

                // (2) Refine initial sample points
                H2P_calc_enclosing_box_size(pt_dim, sample_pt[node]->ncol, sample_pt[node]->data, enbox_size);
                int ri_npt = H2P_proportional_int_decompose(pt_dim, r_down, enbox_size, ri_list);
                if (ri_npt < init_sample_npt)
                {
                    H2P_select_cluster_sample(pt_dim, sample_pt[node], ri_list, 2, 1, sample_idx);
                    H2P_dense_mat_select_columns(sample_pt[node], sample_idx);
                }

                // (3) Pass refined sample points to children
                int sample_npt   = sample_idx->length;
                int n_child_node = n_child[node];
                int *child_nodes = children + node * max_child;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    H2P_dense_mat_resize(sample_pt[i_child_node], xpt_dim, sample_pt[node]->ncol);
                    copy_matrix_block(sizeof(DTYPE), xpt_dim, sample_npt, sample_pt[node]->data, sample_npt, sample_pt[i_child_node]->data, sample_npt);
                }
            }  // End of j loop
            free(enbox_size);
            free(ri_list);
            H2P_int_vec_destroy(&sample_idx);
            H2P_dense_mat_destroy(&workbuf_d);
        }
    }  // End of i loop

    // Free temporary arrays and return sample points
    for (int i = 0; i < n_node; i++)
    {
        H2P_int_vec_destroy(adm_list + i);
        H2P_dense_mat_destroy(clu_refine + i);
    }
    free(adm_list);
    free(clu_refine);
    *sample_points_ = sample_pt;
}

void H2P_select_sample_point_vol(
    H2Pack_p h2pack, const void *krnl_param, kernel_eval_fptr krnl_eval, 
    const DTYPE tau, H2P_dense_mat_p **sample_points_
)
{
    int   pt_dim        = h2pack->pt_dim;
    int   xpt_dim       = h2pack->xpt_dim;
    int   min_adm_level = h2pack->is_HSS ? h2pack->HSS_min_adm_level : h2pack->min_adm_level;
    int   max_level     = h2pack->max_level;
    int   max_child     = h2pack->max_child;
    int   n_point       = h2pack->n_point;
    int   n_node        = h2pack->n_node;
    int   n_leaf_node   = h2pack->n_leaf_node;
    int   n_r_adm_pair  = h2pack->is_HSS ? h2pack->HSS_n_r_adm_pair : h2pack->n_r_adm_pair;
    int   *level_n_node = h2pack->level_n_node;
    int   *level_nodes  = h2pack->level_nodes;
    int   *n_child      = h2pack->n_child;
    int   *children     = h2pack->children;
    int   *pt_cluster   = h2pack->pt_cluster;
    int   *r_adm_pairs  = h2pack->is_HSS ? h2pack->HSS_r_adm_pairs : h2pack->r_adm_pairs;
    DTYPE *coord        = h2pack->coord;

    // 1. Allocate temporary arrays and output arrays
    H2P_int_vec_p   *adm_list   = (H2P_int_vec_p *)   malloc(sizeof(H2P_int_vec_p)   * n_node);
    H2P_dense_mat_p *clu_refine = (H2P_dense_mat_p *) malloc(sizeof(H2P_dense_mat_p) * n_node);
    H2P_dense_mat_p *sample_pt  = (H2P_dense_mat_p *) malloc(sizeof(H2P_dense_mat_p) * n_node);
    for (int i = 0; i < n_node; i++)
    {
        H2P_int_vec_init(adm_list + i, n_node);
        H2P_dense_mat_init(clu_refine + i, 300, xpt_dim);
        H2P_dense_mat_init(sample_pt + i, 300, xpt_dim);
    }

    // 2. Convert the reduced admissible pairs to reduced admissible list of each node
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int c0 = r_adm_pairs[2 * i + 0];
        int c1 = r_adm_pairs[2 * i + 1];
        H2P_int_vec_push_back(adm_list[c0], c1);
        H2P_int_vec_push_back(adm_list[c1], c0);
    }

    int ri = H2P_select_sample_point_r(h2pack, krnl_param, krnl_eval, tau, h2pack->QR_stop_tol);

    // 3. Bottom-up sweep
    int r_up = ri;
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];

        // (1) Update refined points associated with clusters at level i
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            int n_child_node = n_child[node];
            if (n_child_node == 0)
            {
                int pt_s = pt_cluster[node * 2];
                int pt_e = pt_cluster[node * 2 + 1];
                int node_npt = pt_e - pt_s + 1;
                H2P_dense_mat_resize(clu_refine[node], xpt_dim, node_npt);
                copy_matrix_block(sizeof(DTYPE), xpt_dim, node_npt, coord + pt_s, n_point, clu_refine[node]->data, node_npt);
            } else {
                int *child_nodes = children + node * max_child;
                int child_clu_sum = 0;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    int i_child_npt  = clu_refine[i_child_node]->ncol;
                    child_clu_sum += i_child_npt;
                }
                H2P_dense_mat_resize(clu_refine[node], xpt_dim, child_clu_sum);
                child_clu_sum = 0;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    int i_child_npt  = clu_refine[i_child_node]->ncol;
                    int lds = i_child_npt;
                    int ldd = clu_refine[node]->ncol;
                    DTYPE *src = clu_refine[i_child_node]->data;
                    DTYPE *dst = clu_refine[node]->data + child_clu_sum;
                    copy_matrix_block(sizeof(DTYPE), xpt_dim, i_child_npt, src, lds, dst, ldd);
                    child_clu_sum += i_child_npt;
                }
            }  // End of "if (n_child_node == 0)"
        }  // End of j loop

        // (2) Select refined points for all nodes at i-th level
        #pragma omp parallel
        {
            int *ri_list = (int *) malloc(sizeof(int) * pt_dim);
            H2P_int_vec_p sample_idx;
            H2P_int_vec_init(&sample_idx, 1024);
            DTYPE *enbox_size = (DTYPE *) malloc(sizeof(DTYPE) * 2 * pt_dim);
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int node_npt_refine = clu_refine[node]->ncol;
                if (node_npt_refine <= 5) continue;
                H2P_calc_enclosing_box_size(pt_dim, clu_refine[node]->ncol, clu_refine[node]->data, enbox_size);
                int ri_npt = H2P_proportional_int_decompose(pt_dim, r_up, enbox_size, ri_list);
                if (ri_npt < node_npt_refine)
                {
                    H2P_select_cluster_sample(pt_dim, clu_refine[node], ri_list, 6,  1, sample_idx);
                    H2P_dense_mat_select_columns(clu_refine[node], sample_idx);
                }
            }  // End of j loop
            H2P_int_vec_destroy(&sample_idx);
            free(ri_list);
            free(enbox_size);
        }
    }  // End of i loop
    
    // 4. Top-down sweep
    int r_down;
    if (ri > 3) r_down = (ri + 3 > 10) ? (ri + 3) : 10;
    else r_down = ri + 3;
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];

        #pragma omp parallel
        {
            int *ri_list = (int *) malloc(sizeof(int) * pt_dim);
            DTYPE *enbox_size = (DTYPE *) malloc(sizeof(DTYPE) * 2 * pt_dim);
            DTYPE *coord_max  = (DTYPE *) malloc(sizeof(DTYPE) * pt_dim);
            DTYPE *coord_min  = (DTYPE *) malloc(sizeof(DTYPE) * pt_dim);
            H2P_int_vec_p sample_idx;
            H2P_dense_mat_p workbuf_d, yFi;
            H2P_int_vec_init(&sample_idx, 1024);
            H2P_dense_mat_init(&workbuf_d, 1, 1024);
            H2P_dense_mat_init(&yFi, 1, 1024);

            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                if (adm_list[node]->length == 0) continue;

                // Stage 1 in Difeng's code
                // (1) Calculate enclosing box of all far-field refined points
                for (int k = 0; k < pt_dim; k++)
                {
                    coord_max[k] = -1e100;
                    coord_min[k] =  1e100;
                }
                int nFp = 0;
                for (int k = 0; k < adm_list[node]->length; k++)
                {
                    int pair_node = adm_list[node]->data[k];
                    int pair_node_npt = clu_refine[pair_node]->ncol;
                    nFp += pair_node_npt;
                    for (int ii = 0; ii < pt_dim; ii++)
                    {
                        DTYPE *pair_coord_ii = clu_refine[pair_node]->data + ii * clu_refine[pair_node]->ld;
                        for (int jj = 0; jj < pair_node_npt; jj++)
                        {
                            coord_min[ii] = (coord_min[ii] < pair_coord_ii[jj]) ? coord_min[ii] : pair_coord_ii[jj];
                            coord_max[ii] = (coord_max[ii] > pair_coord_ii[jj]) ? coord_max[ii] : pair_coord_ii[jj];
                        }
                        enbox_size[ii] = coord_max[ii] - coord_min[ii];
                    }
                }

                // Here sample_pt[node] are passed from parent node
                int yFi_size = 0;
                if (sample_pt[node]->ncol > nFp)
                {
                    // Put all refined points in far-field nodes into yFi
                    H2P_dense_mat_resize(yFi, xpt_dim, nFp);
                    for (int k = 0; k < adm_list[node]->length; k++)
                    {
                        int pair_node = adm_list[node]->data[k];
                        int pair_node_npt = clu_refine[pair_node]->ncol;
                        int lds = pair_node_npt;
                        int ldd = yFi->ncol;
                        DTYPE *src = clu_refine[pair_node]->data;
                        DTYPE *dst = yFi->data + yFi_size;
                        copy_matrix_block(sizeof(DTYPE), xpt_dim, pair_node_npt, src, lds, dst, ldd);
                        yFi_size += pair_node_npt;
                    }
                } else {
                    int ri_npt0 = H2P_proportional_int_decompose(pt_dim, r_down, enbox_size, ri_list);
                    int num_far = adm_list[node]->length;

                    // (1) Find the center of each far-field node's refined points
                    DTYPE *centers = (DTYPE *) malloc(sizeof(DTYPE) * pt_dim * num_far);
                    for (int k = 0; k < num_far; k++)
                    {
                        int pair_node = adm_list[node]->data[k];
                        int pair_node_npt = clu_refine[pair_node]->ncol;
                        for (int ii = 0; ii < pt_dim; ii++)
                        {
                            DTYPE *pair_coord_ii = clu_refine[pair_node]->data + ii * clu_refine[pair_node]->ld;
                            DTYPE sum = 0.0;
                            for (int jj = 0; jj < pair_node_npt; jj++)
                                sum += pair_coord_ii[jj];
                            centers[ii * num_far + k] = sum / (DTYPE) pair_node_npt;
                        }
                    }

                    // (2) For each anchor point, assign it to the closest center
                    H2P_dense_mat_p anchor_coord_;
                    H2P_dense_mat_init(&anchor_coord_, 1, 1024);
                    H2P_select_anchor_grid(
                        pt_dim, coord_min, coord_max, enbox_size,
                        ri_list, 2, anchor_coord_
                    );
                    int num_anchor = ri_npt0;
                    DTYPE *anchor_coord = anchor_coord_->data;
                    
                    int *iG = (int *) malloc(sizeof(int) * num_anchor);
                    int *center_group_cnt = (int *) malloc(sizeof(int) * num_far);
                    DTYPE *tmp_dist2 = (DTYPE *) malloc(sizeof(DTYPE) * num_anchor * num_far);
                    H2P_calc_pdist2(
                        anchor_coord_->data, anchor_coord_->ld, num_anchor,
                        centers, num_far, num_far,
                        pt_dim, tmp_dist2, num_far
                    );
                    memset(center_group_cnt, 0, sizeof(int) * num_far);
                    for (int k = 0; k < num_anchor; k++)
                    {
                        iG[k] = 0;
                        DTYPE *dist2_k = tmp_dist2 + k * num_far;
                        DTYPE min_dist = dist2_k[0];
                        for (int ii = 1; ii < num_far; ii++)
                        {
                            if (dist2_k[ii] < min_dist)
                            {
                                min_dist = dist2_k[ii];
                                iG[k] = ii;
                            }
                        }
                        center_group_cnt[iG[k]]++;
                    }

                    // (3) Group the anchor points by their assigned center
                    int *group_displs = (int *) malloc(sizeof(int) * num_far);
                    DTYPE *anchor_group = (DTYPE *) malloc(sizeof(DTYPE) * pt_dim * num_anchor);
                    group_displs[0] = 0;
                    for (int k = 1; k < num_far; k++)
                        group_displs[k] = group_displs[k - 1] + center_group_cnt[k - 1];
                    for (int k = 0; k < num_anchor; k++)
                    {
                        int group_idx = iG[k];
                        int displs = group_displs[group_idx];
                        group_displs[group_idx]++;
                        assert(displs < num_anchor);
                        for (int ii = 0; ii < pt_dim; ii++)
                            anchor_group[ii * num_anchor + displs] = anchor_coord[ii * num_anchor + k];
                    }
                    group_displs[0] = 0;
                    for (int k = 1; k < num_far; k++) 
                        group_displs[k] = group_displs[k - 1] + center_group_cnt[k - 1];

                    // (4) Select refined points from each group
                    int max_yFi_size = num_anchor + num_far;
                    H2P_dense_mat_resize(yFi, xpt_dim, max_yFi_size);
                    yFi_size = 0;
                    for (int k = 0; k < num_far; k++)
                    {
                        int pair_node = adm_list[node]->data[k];
                        int pair_node_npt = clu_refine[pair_node]->ncol;
                        int group_k_cnt = center_group_cnt[k];
                        if (group_k_cnt > 0)
                        {
                            size_t tmp_dist2_bytes = sizeof(DTYPE) * pair_node_npt * group_k_cnt;
                            int *select_flag = (int *) malloc(sizeof(int) * pair_node_npt);
                            memset(select_flag, 0, sizeof(int) * pair_node_npt);
                            DTYPE *tmp_dist2 = (DTYPE *) malloc(tmp_dist2_bytes);

                            H2P_calc_pdist2(
                                anchor_group + group_displs[k], num_anchor, group_k_cnt,
                                clu_refine[pair_node]->data, pair_node_npt, pair_node_npt,
                                pt_dim, tmp_dist2, pair_node_npt
                            );
                            
                            for (int ii = 0; ii < group_k_cnt; ii++)
                            {
                                DTYPE *dist2_ii = tmp_dist2 + ii * pair_node_npt;
                                DTYPE min_dist = dist2_ii[0];
                                int min_idx = 0;
                                for (int jj = 1; jj < pair_node_npt; jj++)
                                {
                                    if (dist2_ii[jj] < min_dist)
                                    {
                                        min_dist = dist2_ii[jj];
                                        min_idx = jj;
                                    }
                                }
                                select_flag[min_idx] = 1;
                            }

                            for (int ii = 0; ii < pair_node_npt; ii++)
                            {
                                if (select_flag[ii] == 0) continue;
                                for (int jj = 0; jj < pt_dim; jj++)
                                    yFi->data[jj * yFi->ld + yFi_size] = clu_refine[pair_node]->data[jj * pair_node_npt + ii];
                                yFi_size++;
                                ASSERT_PRINTF(yFi_size <= max_yFi_size, "yFi_size %d > max_yFi_size %d\n", yFi_size, max_yFi_size);
                            }

                            free(tmp_dist2);
                            free(select_flag);
                        } else {
                            DTYPE *tmp_dist2 = (DTYPE *) malloc(sizeof(DTYPE) * pair_node_npt);
                            H2P_calc_pdist2(
                                centers + k, num_far, 1,
                                clu_refine[pair_node]->data, pair_node_npt, pair_node_npt,
                                pt_dim, tmp_dist2, pair_node_npt
                            );
                            DTYPE min_dist = tmp_dist2[0];
                            int min_idx = 0;
                            for (int jj = 1; jj < pair_node_npt; jj++)
                            {
                                if (tmp_dist2[jj] < min_dist)
                                {
                                    min_dist = tmp_dist2[jj];
                                    min_idx = jj;
                                }
                            }
                            for (int jj = 0; jj < pt_dim; jj++)
                                yFi->data[jj * yFi->ld + yFi_size] = clu_refine[pair_node]->data[jj * pair_node_npt + min_idx];
                            yFi_size++;
                            ASSERT_PRINTF(yFi_size <= max_yFi_size, "yFi_size %d > max_yFi_size %d\n", yFi_size, max_yFi_size);
                            free(tmp_dist2);
                        }
                    }  // End of k loop
                    
                    free(centers);
                    free(iG);
                    free(center_group_cnt);
                    free(tmp_dist2);
                    free(group_displs);
                    free(anchor_group);
                    H2P_dense_mat_destroy(&anchor_coord_);
                }  // End of "if (sample_pt[node]->ncol > nFp)"

                // Stage 2 in Difeng's code

                // (1) Gather selected far-field refined points as initial sample points
                int sample_npt0 = sample_pt[node]->ncol;
                int init_sample_npt = sample_npt0 + yFi_size;
                H2P_dense_mat_resize(workbuf_d, xpt_dim, sample_npt0);
                copy_matrix_block(sizeof(DTYPE), xpt_dim, sample_npt0, sample_pt[node]->data, sample_npt0, workbuf_d->data, sample_npt0);
                H2P_dense_mat_resize(sample_pt[node], xpt_dim, init_sample_npt);
                copy_matrix_block(sizeof(DTYPE), xpt_dim, sample_npt0, workbuf_d->data, sample_npt0, sample_pt[node]->data, init_sample_npt);
                copy_matrix_block(sizeof(DTYPE), xpt_dim, yFi_size, yFi->data, yFi->ld, sample_pt[node]->data + sample_npt0, init_sample_npt);

                // (2) Refine initial sample points
                H2P_calc_enclosing_box_size(pt_dim, sample_pt[node]->ncol, sample_pt[node]->data, enbox_size);
                int ri_npt = H2P_proportional_int_decompose(pt_dim, r_down, enbox_size, ri_list);
                if (ri_npt < init_sample_npt)
                {
                    H2P_select_cluster_sample(pt_dim, sample_pt[node], ri_list, 2, 1, sample_idx);
                    H2P_dense_mat_select_columns(sample_pt[node], sample_idx);
                }

                // (3) Pass refined sample points to children
                int sample_npt   = sample_idx->length;
                int n_child_node = n_child[node];
                int *child_nodes = children + node * max_child;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    H2P_dense_mat_resize(sample_pt[i_child_node], xpt_dim, sample_pt[node]->ncol);
                    copy_matrix_block(sizeof(DTYPE), xpt_dim, sample_npt, sample_pt[node]->data, sample_npt, sample_pt[i_child_node]->data, sample_npt);
                }
            }  // End of j loop
            H2P_dense_mat_destroy(&workbuf_d);
            H2P_dense_mat_destroy(&yFi);
            H2P_int_vec_destroy(&sample_idx);
            free(ri_list);
            free(enbox_size);
        }
    }  // End of i loop

    // Free temporary arrays and return sample points
    for (int i = 0; i < n_node; i++)
    {
        H2P_int_vec_destroy(adm_list + i);
        H2P_dense_mat_destroy(clu_refine + i);
    }
    free(adm_list);
    free(clu_refine);
    *sample_points_ = sample_pt;
}

void H2P_select_sample_point(
    H2Pack_p h2pack, const void *krnl_param, kernel_eval_fptr krnl_eval, 
    const DTYPE tau, H2P_dense_mat_p **sample_points_
)
{
    int algo = 0;
    GET_ENV_INT_VAR(algo, "H2P_SELECT_SP_ALG", "select_sp_alg", 1, 0, 1);
    if (algo == 0) H2P_select_sample_point_ref(h2pack, krnl_param, krnl_eval, tau, sample_points_);
    if (algo == 1) H2P_select_sample_point_vol(h2pack, krnl_param, krnl_eval, tau, sample_points_);
}

typedef enum
{
    U_BUILD_KRNL_TIMER_IDX = 0,
    U_BUILD_QR_TIMER_IDX,
    U_BUILD_OTHER_TIMER_IDX,
    U_BUILD_RANDN_TIMER_IDX,
    U_BUILD_GEMM_TIMER_IDX
} u2_build_timer_idx_t;

// Build H2 projection matrices using sample points
// Input parameter:
//   h2pack    : H2Pack structure with point partitioning info
//   sample_pt : Sample points (as far-field points) for each node
// Output parameter:
//   h2pack : H2Pack structure with H2 projection matrices
void H2P_build_H2_UJ_sample(H2Pack_p h2pack, H2P_dense_mat_p *sample_pt)
{
    int    xpt_dim        = h2pack->xpt_dim;
    int    krnl_dim       = h2pack->krnl_dim;
    int    n_node         = h2pack->n_node;
    int    n_point        = h2pack->n_point;
    int    n_thread       = h2pack->n_thread;
    int    max_child      = h2pack->max_child;
    int    stop_type      = h2pack->QR_stop_type;
    int    *children      = h2pack->children;
    int    *n_child       = h2pack->n_child;
    int    *node_height   = h2pack->node_height;
    int    *pt_cluster    = h2pack->pt_cluster;
    DTYPE  *coord         = h2pack->coord;
    size_t *mat_size      = h2pack->mat_size;
    void   *krnl_param    = h2pack->krnl_param;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    DAG_task_queue_p upward_tq   = h2pack->upward_tq;

    // 1e-4 is suggested by Difeng
    DTYPE QR_stop_tol = h2pack->QR_stop_tol * 1e-4;  
    
    void *stop_param = NULL;
    if (stop_type == QR_RANK) 
        stop_param = &h2pack->QR_stop_rank;
    if ((stop_type == QR_REL_NRM) || (stop_type == QR_ABS_NRM))
        stop_param = &QR_stop_tol;
    
    // 1. Allocate U and J
    h2pack->n_UJ = n_node;
    h2pack->U       = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    h2pack->J       = (H2P_int_vec_p*)   malloc(sizeof(H2P_int_vec_p)   * n_node);
    h2pack->J_coord = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    ASSERT_PRINTF(h2pack->U       != NULL, "Failed to allocate %d U matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J       != NULL, "Failed to allocate %d J matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J_coord != NULL, "Failed to allocate %d J_coord auxiliary matrices\n", n_node);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]       = NULL;
        h2pack->J[i]       = NULL;
        h2pack->J_coord[i] = NULL;
    }
    H2P_dense_mat_p *U       = h2pack->U;
    H2P_int_vec_p   *J       = h2pack->J;
    H2P_dense_mat_p *J_coord = h2pack->J_coord;
    
    // 2. Construct U for nodes whose level is not smaller than min_adm_level.
    //    min_adm_level is the highest level that still has admissible blocks.
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p A_block         = thread_buf[tid]->mat0;
        H2P_dense_mat_p node_skel_coord = thread_buf[tid]->mat1;
        H2P_dense_mat_p QR_buff         = thread_buf[tid]->mat2;
        H2P_int_vec_p   sub_idx         = thread_buf[tid]->idx0;
        H2P_int_vec_p   ID_buff         = thread_buf[tid]->idx1;

        thread_buf[tid]->timer = -get_wtime_sec();
        int node = DAG_task_queue_get_task(upward_tq);
        while (node != -1)
        {
            // (1) Update row indices associated with clusters for current node
            int height = node_height[node];
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
                copy_matrix_block(sizeof(DTYPE), xpt_dim, node_npt, coord + pt_s, n_point, J_coord[node]->data, node_npt);
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
                    copy_matrix_block(sizeof(DTYPE), xpt_dim, src_ld, src_mat, src_ld, dst_mat, dst_ld);
                    J_child_size += J[i_child_node]->length;
                }
            }  // End of "if (level == 0)"

            // (3) Build the kernel matrix block
            int node_skel_npt = J[node]->length;
            int node_sp_npt   = sample_pt[node]->ncol;
            int A_blk_nrow = node_skel_npt * krnl_dim;
            int A_blk_ncol = node_sp_npt   * krnl_dim;
            H2P_dense_mat_resize(A_block, A_blk_nrow, A_blk_ncol);
            krnl_eval(
                node_skel_coord->data, node_skel_npt, node_skel_npt,
                sample_pt[node]->data, node_sp_npt,   node_sp_npt, 
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
    
    if (h2pack->print_timers == 1)
    {
        double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
        for (int i = 0; i < n_thread; i++)
        {
            double thread_i_timer = thread_buf[i]->timer;
            avg_t += thread_i_timer;
            max_t = MAX(max_t, thread_i_timer);
            min_t = MIN(min_t, thread_i_timer);
        }
        avg_t /= (double) n_thread;
        INFO_PRINTF("Build U: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    }

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
            mat_size[U_SIZE_IDX]      += U[i]->nrow * U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
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

// These two functions are in H2Pack_build.c but not in H2Pack_utils.h
void H2P_build_B_AOT(H2Pack_p h2pack);
void H2P_build_D_AOT(H2Pack_p h2pack);

// Build H2 representation with a kernel function and sample points
void H2P_build_with_sample_point(
    H2Pack_p h2pack, H2P_dense_mat_p *sample_pt, const int BD_JIT, void *krnl_param, 
    kernel_eval_fptr krnl_eval, kernel_bimv_fptr krnl_bimv, const int krnl_bimv_flops
)
{
    double st, et;
    double *timers = h2pack->timers;

    if (sample_pt == NULL)
    {
        ERROR_PRINTF("You need to provide a set of sample points.\n");
        return;
    }
    
    if (krnl_eval == NULL)
    {
        ERROR_PRINTF("You need to provide a valid krnl_eval().\n");
        return;
    }

    h2pack->BD_JIT = BD_JIT;
    h2pack->krnl_param = krnl_param;
    h2pack->krnl_eval  = krnl_eval;
    h2pack->krnl_bimv  = krnl_bimv;
    h2pack->krnl_bimv_flops = krnl_bimv_flops;
    if (BD_JIT == 1 && krnl_bimv == NULL) 
        WARNING_PRINTF("krnl_eval() will be used in BD_JIT matvec. For better performance, consider using a krnl_bimv().\n");

    // 1. Build projection matrices and skeleton row sets
    st = get_wtime_sec();
    //if (h2pack->is_HSS) H2P_build_HSS_UJ_hybrid(h2pack);
    //else H2P_build_H2_UJ_proxy(h2pack);
    H2P_build_H2_UJ_sample(h2pack, sample_pt);
    et = get_wtime_sec();
    timers[U_BUILD_TIMER_IDX] = et - st;

    // 2. Build generator matrices
    st = get_wtime_sec();
    H2P_generate_B_metadata(h2pack);
    if (BD_JIT == 0) H2P_build_B_AOT(h2pack);
    et = get_wtime_sec();
    timers[B_BUILD_TIMER_IDX] = et - st;
    
    // 3. Build dense blocks
    st = get_wtime_sec();
    H2P_generate_D_metadata(h2pack);
    if (BD_JIT == 0) H2P_build_D_AOT(h2pack);
    et = get_wtime_sec();
    timers[D_BUILD_TIMER_IDX] = et - st;

    // 4. Set up forward and backward permutation indices
    int n_point    = h2pack->n_point;
    int krnl_dim   = h2pack->krnl_dim;
    int *coord_idx = h2pack->coord_idx;
    int *fwd_pmt_idx = (int*) malloc(sizeof(int) * n_point * krnl_dim);
    int *bwd_pmt_idx = (int*) malloc(sizeof(int) * n_point * krnl_dim);
    for (int i = 0; i < n_point; i++)
    {
        for (int j = 0; j < krnl_dim; j++)
        {
            fwd_pmt_idx[i * krnl_dim + j] = coord_idx[i] * krnl_dim + j;
            bwd_pmt_idx[coord_idx[i] * krnl_dim + j] = i * krnl_dim + j;
        }
    }
    h2pack->fwd_pmt_idx = fwd_pmt_idx;
    h2pack->bwd_pmt_idx = bwd_pmt_idx;
}
