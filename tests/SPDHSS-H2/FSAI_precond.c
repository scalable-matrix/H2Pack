#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack.h"
#include "H2Pack_utils.h"
#include "CSRPlus.h"
#include "FSAI_precond.h"

// Calculate the square of distance of each pair of 2D points for KNN search
static void calc_2D_dist2(
    const DTYPE *coord0, const int ld0, const int n0, 
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *dist, const int ld 
)
{
    const DTYPE *x0 = coord0 + ld0 * 0;
    const DTYPE *y0 = coord0 + ld0 * 1;
    const DTYPE *x1 = coord1 + ld1 * 0;
    const DTYPE *y1 = coord1 + ld1 * 1;
    #pragma omp parallel for 
    for (int i = 0; i < n0; i++)
    {
        DTYPE x0_i = x0[i];
        DTYPE y0_i = y0[i];
        DTYPE *dist_i = dist + i * ld;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            dist_i[j] = dx * dx + dy * dy;
        }
    }
}

// Calculate the square of distance of each pair of 3D points for KNN search
static void calc_3D_dist2(
    const DTYPE *coord0, const int ld0, const int n0, 
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *dist, const int ld
)
{
    const DTYPE *x0 = coord0 + ld0 * 0;
    const DTYPE *y0 = coord0 + ld0 * 1;
    const DTYPE *z0 = coord0 + ld0 * 1;
    const DTYPE *x1 = coord1 + ld1 * 0;
    const DTYPE *y1 = coord1 + ld1 * 1;
    const DTYPE *z1 = coord1 + ld1 * 1;
    #pragma omp parallel for 
    for (int i = 0; i < n0; i++)
    {
        DTYPE x0_i = x0[i];
        DTYPE y0_i = y0[i];
        DTYPE z0_i = z0[i];
        DTYPE *dist_i = dist + i * ld;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            dist_i[j] = dx * dx + dy * dy + dz * dz;
        }
    }
}

// Quick-sort for (key, val) pairs
static void qsort_DTYPE_int_pair(DTYPE *key, int *val, int l, int r)
{
    int i = l, j = r, tmp_val;
    DTYPE mid_key = key[(l + r) / 2], tmp_key;
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            tmp_key = key[i]; key[i] = key[j]; key[j] = tmp_key;
            tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) qsort_DTYPE_int_pair(key, val, i, r);
    if (j > l) qsort_DTYPE_int_pair(key, val, l, j);
}

// Search k nearest neighbors for each point using the hierarchical partition in H2Pack
static void H2P_search_knn(H2Pack_p h2pack, const int k, int *knn)
{
    int pt_dim       = h2pack->pt_dim;
    int n_point      = h2pack->n_point;
    int n_leaf_node  = h2pack->n_leaf_node;
    int max_neighbor = h2pack->max_neighbor;
    int max_leaf_pts = h2pack->max_leaf_points;
    int *leaf_nodes  = h2pack->height_nodes;
    int *pt_cluster  = h2pack->pt_cluster;
    DTYPE *enbox = h2pack->enbox;
    DTYPE *coord = h2pack->coord;

    H2P_dense_mat_p dist;
    H2P_dense_mat_p neighbor_pt_coord;
    H2P_int_vec_p   pt_idx;
    int max_candidate = max_leaf_pts * max_neighbor;
    H2P_dense_mat_init(&dist, max_leaf_pts, max_candidate);
    H2P_dense_mat_init(&neighbor_pt_coord, pt_dim, max_candidate);
    H2P_int_vec_init(&pt_idx, max_leaf_pts * max_candidate);

    for (int i = 0; i < n_leaf_node; i++)
    {
        int node      = leaf_nodes[i];
        int node_pt_s = pt_cluster[2 * node];
        int node_pt_e = pt_cluster[2 * node + 1];
        int node_npt  = node_pt_e - node_pt_s + 1;
        DTYPE *node_enbox = enbox + node * 2 * pt_dim;

        // Gather all points in inadmissible leaf nodes
        int neighbor_pt_cnt = 0;
        for (int j = 0; j < n_leaf_node; j++)
        {
            int node_j = leaf_nodes[j];
            DTYPE *node_j_enbox = enbox + node_j * 2 * pt_dim;
            if (H2P_check_box_admissible(node_enbox, node_j_enbox, pt_dim, ALPHA_H2) == 1) continue;
            int node_j_pt_s = pt_cluster[2 * node_j];
            int node_j_pt_e = pt_cluster[2 * node_j + 1];
            int node_j_npt  = node_j_pt_e - node_j_pt_s + 1;
            copy_matrix_block(
                sizeof(DTYPE), pt_dim, node_j_npt, coord + node_j_pt_s, n_point, 
                neighbor_pt_coord->data + neighbor_pt_cnt, max_candidate
            );
            for (int k = 0; k < node_j_npt; k++)
                pt_idx->data[neighbor_pt_cnt + k] = node_j_pt_s + k;
            neighbor_pt_cnt += node_j_npt;
        }  // End of j loop
        ASSERT_PRINTF(
            neighbor_pt_cnt <= max_candidate, 
            "Node %d: inadm nodes + self only has %d points, > %d estimated maximum\n",
            node, neighbor_pt_cnt, max_candidate
        );
        if (neighbor_pt_cnt < k)
        {
            WARNING_PRINTF(
                "Node %d has only %d (< %d) nearest neighbors from inadmissible nodes\n",
                node, neighbor_pt_cnt, k
            );
        }
        for (int j = 1; j < node_npt; j++)
            memcpy(pt_idx->data + j * neighbor_pt_cnt, pt_idx->data, sizeof(int) * neighbor_pt_cnt);

        // Calculate pairwise distance
        H2P_dense_mat_resize(dist, node_npt, neighbor_pt_cnt);
        if (pt_dim == 2)
        {
            calc_2D_dist2(
                coord + node_pt_s,       n_point,       node_npt,
                neighbor_pt_coord->data, max_candidate, neighbor_pt_cnt,
                dist->data, dist->ld
            );
        }
        if (pt_dim == 3)
        {
            calc_3D_dist2(
                coord + node_pt_s,       n_point,       node_npt,
                neighbor_pt_coord->data, max_candidate, neighbor_pt_cnt,
                dist->data, dist->ld
            );
        }

        // Sort pairwise distance and get the nearest neighbors
        if (neighbor_pt_cnt > k)
        {
            #pragma omp parallel for
            for (int j = 0; j < node_npt; j++)
            {
                DTYPE *dist_j   =   dist->data + j * neighbor_pt_cnt;
                int   *pt_idx_j = pt_idx->data + j * neighbor_pt_cnt;
                qsort_DTYPE_int_pair(dist_j, pt_idx_j, 0, neighbor_pt_cnt - 1);
            }  // End of j loop
            copy_matrix_block(sizeof(int), node_npt, k, pt_idx->data, neighbor_pt_cnt, knn + node_pt_s * k, k);
        } else {
            copy_matrix_block(sizeof(int), node_npt, neighbor_pt_cnt, pt_idx->data, neighbor_pt_cnt, knn + node_pt_s * k, k);
            // Not enough neighbor points, set the rest as self
            #pragma omp parallel for
            for (int j = node_pt_s; j <= node_pt_e; j++)
            {
                int *knn_j = knn + j * k;
                for (int l = neighbor_pt_cnt; l < k; l++) knn_j[l] = j;
            }
        }
    }  // End of i loop

    H2P_dense_mat_destroy(&dist);
    H2P_dense_mat_destroy(&neighbor_pt_coord);
    H2P_int_vec_destroy(&pt_idx);
    free(dist);
    free(pt_idx);
}

// Construct a FSAI_precond from a H2Pack structure
void H2P_build_FSAI_precond(H2Pack_p h2pack, const int rank, const DTYPE shift, FSAI_precond_p *precond_)
{
    FSAI_precond_p precond = (FSAI_precond_p) malloc(sizeof(FSAI_precond_s));
    assert(precond != NULL);

    if (h2pack->pt_dim != 2 && h2pack->pt_dim != 3)
    {
        ERROR_PRINTF("FSAI preconditioner construction only support 2D or 3D points\n");
        return;
    }

    double st = get_wtime_sec();

    int mat_size    = h2pack->krnl_mat_size;
    int n_point     = h2pack->n_point;
    int n_thread    = h2pack->n_thread;
    int pt_dim      = h2pack->pt_dim;
    int xpt_dim     = h2pack->xpt_dim;
    int krnl_dim    = h2pack->krnl_dim;
    int n_neighbor  = rank / krnl_dim;
    int max_nnz     = krnl_dim * krnl_dim * n_neighbor * n_point;
    DTYPE *coord = h2pack->coord;
    
    int   *knn = (int*)   malloc(sizeof(int)   * n_point * n_neighbor);
    int   *row = (int*)   malloc(sizeof(int)   * max_nnz);
    int   *col = (int*)   malloc(sizeof(int)   * max_nnz);
    DTYPE *val = (DTYPE*) malloc(sizeof(DTYPE) * max_nnz);
    ASSERT_PRINTF(
        knn != NULL && row != NULL & col != NULL && val != NULL,
        "Failed to allocate working arrays for FSAI preconditioner construction\n"
    );

    H2P_search_knn(h2pack, n_neighbor, knn);

    //FILE *ouf = fopen("C_knn.bin", "wb");
    //fwrite(knn, sizeof(int), n_neighbor * n_point, ouf);
    //fclose(ouf);

    int nnz = 0;
    int *row_ptr = (int*) malloc(sizeof(int) * (n_point + 1));
    row_ptr[0] = 0;
    for (int i = 0; i < n_point; i++)
    {
        int num_i = 0;
        int *nn_i = knn + i * n_neighbor;
        for (int j = 0; j < n_neighbor; j++)
            if (nn_i[j] < i) num_i++;
        num_i++;  // For self
        num_i *= krnl_dim * krnl_dim;
        row_ptr[i + 1] = num_i + row_ptr[i]; 
    }
    nnz = row_ptr[n_point];
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_thread_buf_p tb = h2pack->tb[tid];
        H2P_int_vec_p   nn_idx   = tb->idx0;
        H2P_int_vec_p   col_idx  = tb->idx1;
        H2P_dense_mat_p nn_coord = tb->mat0;
        H2P_dense_mat_p tmpAU    = tb->mat1;
        H2P_dense_mat_p tmpYDL   = tb->mat2;

        #pragma omp for schedule(guided)
        for (int i = 0; i < n_point; i++)
        {
            int row_i_spos = row_ptr[i];
            int row_i_epos = row_ptr[i + 1];
            int row_nnz    = row_i_epos - row_i_spos;
            H2P_int_vec_set_capacity(nn_idx, row_nnz);
            // nn_i  = [neighbor(i, neighbor(i, :) < i), i];
            // num_i = length(nn_i);
            int num_i = 0;
            int *nn_i = knn + i * n_neighbor;
            for (int j = 0; j < n_neighbor; j++)
            {
                if (nn_i[j] >= i) continue;
                nn_idx->data[num_i] = nn_i[j];
                num_i++;
            }
            nn_idx->data[num_i] = i;
            num_i++;
            nn_idx->length = num_i;
            // tmpA = kernel({coord(nn_i, :), coord(nn_i, :)}) + shift * eye(krnl_dim * num_i);
            H2P_dense_mat_resize(nn_coord, xpt_dim, num_i);
            for (int k = 0; k < xpt_dim; k++)
            {
                DTYPE *nn_coord_k = nn_coord->data + k * num_i;
                DTYPE *coord_k = coord + k * n_point;
                for (int l = 0; l < num_i; l++)
                    nn_coord_k[l] = coord_k[nn_idx->data[l]];
            }
            int A_size = num_i * krnl_dim;
            H2P_dense_mat_resize(tmpAU, A_size + krnl_dim, A_size);
            DTYPE *tmpA = tmpAU->data;
            DTYPE *tmpU = tmpAU->data + A_size * A_size;
            h2pack->krnl_eval(
                nn_coord->data, num_i, num_i, 
                nn_coord->data, num_i, num_i, 
                h2pack->krnl_param, tmpA, A_size
            );
            for (int j = 0; j < A_size; j++)
                tmpA[j * A_size + j] += shift;
            // tmpU = [zeros(krnl_dim * (num_i - 1), krnl_dim); eye(krnl_dim)];
            memset(tmpU, 0, sizeof(DTYPE) * A_size * krnl_dim);
            int offset = krnl_dim * (num_i - 1);
            for (int j = 0; j < krnl_dim; j++)
                tmpU[(offset + j) * krnl_dim + j] = 1.0;
            // tmpY = (tmpA \ tmpU)';
            if (A_size == 1)
            {
                tmpU[0] /= tmpA[0];
            } else {
                H2P_int_vec_set_capacity(col_idx, A_size);
                int *ipiv = col_idx->data;
                LAPACK_GETRF(LAPACK_ROW_MAJOR, A_size, A_size, tmpA, A_size, ipiv);
                LAPACK_GETRS(LAPACK_ROW_MAJOR, 'N', A_size, krnl_dim, tmpA, A_size, ipiv, tmpU, krnl_dim);
            }
            H2P_dense_mat_resize(tmpYDL, krnl_dim, A_size + 2 * krnl_dim);
            DTYPE *tmpY = tmpYDL->data;
            DTYPE *tmpD = tmpY + krnl_dim * A_size;
            DTYPE *tmpL = tmpD + krnl_dim * krnl_dim;
            if (krnl_dim == 1)
            {
                DTYPE coef = 1.0 / sqrt(tmpU[A_size - 1]);
                for (int j = 0; j < A_size; j++) tmpY[j] = tmpU[j] * coef;
            } else {
                H2P_transpose_dmat(1, A_size, krnl_dim, tmpU, krnl_dim, tmpY, A_size);
                // tmpD = tmpY(:, end-krnl_dim+1:end); 
                DTYPE *tmpY_src = tmpY + (A_size - krnl_dim);
                copy_matrix_block(sizeof(DTYPE), krnl_dim, krnl_dim, tmpY_src, A_size, tmpD, krnl_dim);
                // tmpL = 0.5 * (tmpL + tmpL');
                for (int j = 0; j < krnl_dim; j++)
                {
                    for (int k = 0; k < krnl_dim; k++)
                    {
                        int idx_jk = j * krnl_dim + k;
                        int idx_kj = k * krnl_dim + j;
                        tmpL[idx_jk] = 0.5 * (tmpD[idx_jk] + tmpD[idx_kj]);
                    }
                }
                // tmpL = chol(tmpL, 'lower');
                int info;
                info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', krnl_dim, tmpL, krnl_dim);
                ASSERT_PRINTF(info == 0, "Point %d Cholesky factorization for tmpL returned %d\n", i, info);
                // tmpY = linsolve(tmpL, tmpY, struct('LT', true));
                CBLAS_TRSM(
                    CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 
                    krnl_dim, A_size, 1.0, tmpL, krnl_dim, tmpY, A_size
                );
            }  // End of "if (krnl_dim == 1)"
            H2P_int_vec_set_capacity(col_idx, num_i * krnl_dim);
            for (int j = 0; j < num_i; j++)
            {
                int *col_idx_j = col_idx->data + j * krnl_dim;
                for (int k = 0; k < krnl_dim; k++)
                    col_idx_j[k] = krnl_dim * nn_idx->data[j] + k;
            }
            int cnt = 0;
            for (int j = 0; j < krnl_dim; j++)
            {
                for (int k = 0; k < A_size; k++)
                {
                    row[row_i_spos + cnt] = krnl_dim * i + j;
                    col[row_i_spos + cnt] = col_idx->data[k];
                    val[row_i_spos + cnt] = tmpY[j * A_size + k];
                    cnt++;
                }
            }  // End of j loop
        }  // End of i loop
    }  // End of "#pragma omp parallel"

    CSRP_mat_p G = NULL, Gt = NULL;
    CSRP_init_with_COO_mat(mat_size, mat_size, nnz, row, col, val, &G);
    CSRP_init_with_COO_mat(mat_size, mat_size, nnz, col, row, val, &Gt);
    CSRP_partition_multithread(G,  n_thread, n_thread);
    CSRP_partition_multithread(Gt, n_thread, n_thread);
    CSRP_optimize_NUMA(G);
    CSRP_optimize_NUMA(Gt);

    free(knn);
    free(row);
    free(col);
    free(val);
    free(row_ptr);

    double et = get_wtime_sec();

    DTYPE *x0 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    ASSERT_PRINTF(x0 != NULL, "Failed to allocate working array of size %d for FSAI preconditioner\n", mat_size);
    precond->mat_size = mat_size;
    precond->x0       = x0;
    precond->G        = G;
    precond->Gt       = Gt;
    precond->t_build  = et - st;
    precond->t_apply  = 0.0;
    precond->n_apply  = 0;
    precond->mem_MB   = 2.0 * ((sizeof(DTYPE) + sizeof(int)) * (nnz + mat_size)) / 1048576.0;
    *precond_ = precond;
}

// Apply FSAI preconditioner, x := M_{FSAI}^{-1} * b
void FSAI_precond_apply(FSAI_precond_p precond, const DTYPE *b, DTYPE *x)
{
    if (precond == NULL) return;
    double st = get_wtime_sec();
    CSRP_SpMV(precond->G,  b, precond->x0);
    CSRP_SpMV(precond->Gt, precond->x0, x);
    double et = get_wtime_sec();
    precond->t_apply += et - st;
    precond->n_apply++;
}

// Destroy a FSAI_precond structure
// Input parameter:
//   precond : A FSAI_precond structure to be destroyed
void FSAI_precond_destroy(FSAI_precond_p *precond_)
{
    FSAI_precond_p precond = *precond_;
    if (precond == NULL) return;
    CSRP_destroy(&precond->G);
    CSRP_destroy(&precond->Gt);
    free(precond->G);
    free(precond->Gt);
    free(precond);
    *precond_ = NULL;
}

// Print statistic info of a FSAI_precond structure
void FSAI_precond_print_stat(FSAI_precond_p precond)
{
    if (precond == NULL) return;
    printf(
        "FSAI precond used memory = %.2lf MB, build time = %.3lf sec, apply avg time = %.3lf sec\n", 
        precond->mem_MB, precond->t_build, precond->t_apply / (double) precond->n_apply
    );
}
