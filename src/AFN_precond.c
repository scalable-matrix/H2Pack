#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <float.h>

#include "omp.h"
#include "linalg_lib_wrapper.h"
#include "H2Pack.h"
#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "AFN_precond.h"

#if DTYPE_SIZE == DOUBLE_SIZE
#define NEXTAFTER nextafter
#endif
#if DTYPE_SIZE == FLOAT_SIZE
#define NEXTAFTER nextafterf
#endif

// =============== Nystrom Preconditioner =============== //

// Build a Nystrom preconditioner with diagonal shift
// Nystrom approximation: K1 = [K11, K12], K ~= K1^T * inv(K11) * K1
// Input parameters:
//   mu  : Diagonal shift
//   n1  : Size of K11 block
//   n2  : Number of columns in K12 block
//   K11 : Size n1 * n1, row major
//   K12 : Size n1 * n2, row major
// Output parameters:
//   *nys_M_ : Size n1
//   *nys_U_ : Size n1 * n, row major
void Nys_precond_build_(
    const DTYPE mu, const int n1, const int n2, DTYPE *K11, 
    DTYPE *K12, DTYPE **nys_M_, DTYPE **nys_U_
)
{
    // K1 = [K11, K12];
    int n = n1 + n2;
    DTYPE *K1 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n);
    ASSERT_PRINTF(K1 != NULL, "Failed to allocate K1 of size %d x %d\n", n1, n);
    copy_matrix(sizeof(DTYPE), n1, n1, K11, n1, K1,      n, 1);
    copy_matrix(sizeof(DTYPE), n1, n2, K12, n2, K1 + n1, n, 1);

    // Slightly shift the diagonal to make Nystrom stable
    // TODO: sqrt(n) or sqrt(n1)?
    DTYPE nu = 0, K1_fnorm = 0;
    #pragma omp parallel for schedule(static) reduction(+:K1_fnorm)
    for (int i = 0; i < n * n1; i++) K1_fnorm += K1[i] * K1[i];
    nu = DSQRT((DTYPE) n) * (NEXTAFTER(K1_fnorm, K1_fnorm + 1.0) - K1_fnorm);

    // K11 = K11 + nu * eye(n1);
    for (int i = 0; i < n1; i++) K11[i * n1 + i] += nu;
    // invL = inv(chol(K11, 'lower'));
    DTYPE *invL = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    ASSERT_PRINTF(invL != NULL, "Failed to allocate invL of size %d x %d\n", n1, n1);
    int info = 0;
    info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', n1, K11, n1);
    ASSERT_PRINTF(info == 0, "LAPACK_POTRF failed, info = %d\n", info);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n1 * n1; i++) invL[i] = 0;
    for (int i = 0; i < n1; i++) invL[i * n1 + i] = 1.0;
    CBLAS_TRSM(
        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, 
        CblasNonUnit, n1, n1, 1.0, K11, n1, invL, n1
    );
    // M = K1' * invL';
    DTYPE *M = (DTYPE *) malloc(sizeof(DTYPE) * n * n1);
    ASSERT_PRINTF(M != NULL, "Failed to allocate M of size %d x %d\n", n, n1);
    CBLAS_GEMM(
        CblasRowMajor, CblasTrans, CblasTrans, n, n1, n1, 
        1.0, K1, n, invL, n1, 0.0, M, n1
    );
    free(K1);
    free(invL);

    // [U, S, ~] = svd(M, 0);
    DTYPE *S = (DTYPE *) malloc(sizeof(DTYPE) * n1);
    ASSERT_PRINTF(S != NULL, "Failed to allocate S of size %d\n", n1);
    //#define NYSTROM_SVD_DIRECT
    #ifdef NYSTROM_SVD_DIRECT
    DTYPE *superb = (DTYPE *) malloc(sizeof(DTYPE) * n1);
    info = LAPACK_GESVD(
        LAPACK_ROW_MAJOR, 'O', 'N', n, n1, M, n1, 
        S, NULL, n1, NULL, n1, superb
    );
    ASSERT_PRINTF(info == 0, "LAPACK_GESVD failed, info = %d\n", info);
    free(superb);
    *nys_U_ = U;
    int min_eigval_idx = n1 - 1;
    #else
    // Use EVD is usually faster but may be less accurate
    // MKL with ICC 19.1.3 has a bug in LAPACK_GESVD so we have to use EVD instead
    // MTM = M' * M;
    DTYPE *MTM = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    ASSERT_PRINTF(MTM != NULL, "Failed to allocate MTM of size %d x %d\n", n1, n1);
    // GEMM might make MTM not SPD, use SYRK seems to be better
    CBLAS_SYRK(CblasRowMajor, CblasUpper, CblasTrans, n1, n, 1.0, M, n1, 0.0, MTM, n1);
    // [V, S] = eig(MTM);
    // S = sqrt(S);
    // V = V * inv(S);
    info = LAPACK_SYEVD(LAPACK_ROW_MAJOR, 'V', 'U', n1, MTM, n1, S);
    DTYPE max_S = S[n1 - 1];
    DTYPE S_min_tol = DSQRT(S[n1 - 1]) * D_EPS * 10;
    DTYPE *invS = (DTYPE *) malloc(sizeof(DTYPE) * n1);
    for (int i = 0; i < n1; i++)
    {
        if (S[i] < 0) S[i] = max_S * D_EPS * D_EPS;  // Safeguard
        S[i] = DSQRT(S[i]);
        invS[i] = 1.0 / S[i];
        // Truncate extremely small eigenvalues?
        if (S[i] < S_min_tol) invS[i] = 0.0;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n1; i++)
        for (int j = 0; j < n1; j++) MTM[i * n1 + j] *= invS[j];
    // U = M * V;
    DTYPE *U = (DTYPE *) malloc(sizeof(DTYPE) * n * n1);
    ASSERT_PRINTF(U != NULL, "Failed to allocate U of size %d x %d\n", n, n1);
    CBLAS_GEMM(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n1, n1, 
        1.0, M, n1, MTM, n1, 0.0, U, n1
    );
    *nys_U_ = U;
    free(M);
    free(MTM);
    free(invS);
    int min_eigval_idx = 0;
    #endif

    // S = max(diag(S).^2 - nu, 0);
    // eta = S(n1) + mu;
    // nys_M = eta ./ (S + mu);
    DTYPE *nys_M = (DTYPE *) malloc(sizeof(DTYPE) * n1);
    ASSERT_PRINTF(nys_M != NULL, "Failed to allocate nys_M of size %d\n", n1);
    for (int i = 0; i < n1; i++)
    {
        S[i] = S[i] * S[i] - nu;
        if (S[i] < 0) S[i] = 0;
    }
    DTYPE eta = S[min_eigval_idx] + mu;
    for (int i = 0; i < n1; i++) nys_M[i] = eta / (S[i] + mu);
    free(S);
    *nys_M_ = nys_M;
}

// Apply a Nystrom preconditioner
// Input parameters:
//   n1      : Number of rows in nys_M and nys_U
//   n       : Number of columns in nys_U, and size of x
//   nys_M   : Size n1, output of Nys_precond_build_()
//   nys_U   : Size n1 * n, row major, output of Nys_precond_build_()
//   x       : Size n, input vector
//   t       : Size n, work buffer
// Output parameter:
//   y : Size n, output vector
void Nys_precond_apply_(
    const int n1, const int n, const DTYPE *nys_M, const DTYPE *nys_U, 
    const DTYPE *x, DTYPE *y, DTYPE *t
)
{
    // t = nys_U' * x;
    CBLAS_GEMV(CblasRowMajor, CblasTrans, n, n1, 1.0, nys_U, n1, x, 1, 0.0, t, 1);
    // y = x - nys_U * t;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) y[i] = x[i];
    CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n, n1, -1.0, nys_U, n1, t, 1, 1.0, y, 1);
    // t = nys_M .* t;
    for (int i = 0; i < n1; i++) t[i] *= nys_M[i];
    // y = y + nys_U * t;
    CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n, n1, 1.0, nys_U, n1, t, 1, 1.0, y, 1);
}

// ============= Nystrom Preconditioner End ============= //


// ================= FSAI Preconditioner ================ //

static inline void swap_int_DTYPE_pair(int *int_arr, DTYPE *d_arr, int i, int j)
{
    int tmp_i = int_arr[i]; int_arr[i] = int_arr[j]; int_arr[j] = tmp_i;
    DTYPE tmp_d = d_arr[i]; d_arr[i] = d_arr[j]; d_arr[j] = tmp_d;
}

// Quick sort for (int, DTYPE) key-value pairs
static void Qsort_int_DTYPE_key_val(int *key, DTYPE *val, const int l, const int r)
{
    int i = l, j = r;
    int mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            swap_int_DTYPE_pair(key, val, i, j);
            i++;  j--;
        }
    }
    if (i < r) Qsort_int_DTYPE_key_val(key, val, i, r);
    if (j > l) Qsort_int_DTYPE_key_val(key, val, l, j);
}

// Quick partitioning for (DTYPE, int) key-value pairs and get the first k smallest elements
static void Qpart_DTYPE_int_key_val(DTYPE *key, int *val, const int l, const int r, const int k)
{
    int i = l, j = r;
    DTYPE mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            swap_int_DTYPE_pair(val, key, i, j);
            i++;  j--;
        }
    }
    if (j > l) Qpart_DTYPE_int_key_val(key, val, l, j, k);
    if ((i < r) && (i < k)) Qpart_DTYPE_int_key_val(key, val, i, r, k);
}

// Convert a COO matrix to a sorted CSR matrix
static void COO2CSR(
    const int nrow, const int ncol, const int nnz,
    const int *row, const int *col, const DTYPE *val, 
    int **row_ptr_, int **col_idx_, DTYPE **csr_val_
)
{
    int *row_ptr = (int *) malloc(sizeof(int) * (nrow + 1));
    int *col_idx = (int *) malloc(sizeof(int) * nnz);
    DTYPE *csr_val = (DTYPE *) malloc(sizeof(DTYPE) * nnz);
    ASSERT_PRINTF(
        row_ptr != NULL && col_idx != NULL && csr_val != NULL, 
        "Failed to allocate CSR arrays for matrix of size %d * %d, %d nnz\n",
        nrow, ncol, nnz
    );

    // Get the number of non-zeros in each row
    memset(row_ptr, 0, sizeof(int) * (nrow + 1));
    for (int i = 0; i < nnz; i++) row_ptr[row[i] + 1]++;

    // Calculate the displacement of 1st non-zero in each row
    for (int i = 2; i <= nrow; i++) row_ptr[i] += row_ptr[i - 1];

    // Use row_ptr to bucket sort col[] and val[]
    for (int i = 0; i < nnz; i++)
    {
        int idx = row_ptr[row[i]];
        col_idx[idx] = col[i];
        csr_val[idx] = val[i];
        row_ptr[row[i]]++;
    }

    // Reset row_ptr
    for (int i = nrow; i >= 1; i--) row_ptr[i] = row_ptr[i - 1];
    row_ptr[0] = 0;

    // Sort the non-zeros in each row according to column indices
    #pragma omp parallel for
    for (int i = 0; i < nrow; i++)
        Qsort_int_DTYPE_key_val(col_idx, csr_val, row_ptr[i], row_ptr[i + 1] - 1);

    *row_ptr_ = row_ptr;
    *col_idx_ = col_idx;
    *csr_val_ = csr_val;
}

// Select k exact nearest neighbors for each point s.t. the indices of 
// neighbors are smaller than the index of the point
// Input parameters:
//   fsai_npt  : Maximum number of nonzeros in each row of the FSAI matrix (number of nearest neighbors)
//   n, pt_dim : Number of points and point dimension
//   coord     : Size pt_dim * ldc, row major, each column is a point coordinate
//   ldc       : Leading dimension of coord, >= n
// Output parameters:
//   nn_idx : Size n * fsai_npt, row major, indices of the nearest neighbors
//   nn_cnt : Size n, number of selected nearest neighbors for each point
static void FSAI_exact_KNN(
    const int fsai_npt, const int n, const int pt_dim, 
    const DTYPE *coord, const int ldc, int *nn_idx, int *nn_cnt
)
{
    int n_thread = omp_get_max_threads();
    memset(nn_cnt, 0, sizeof(int) * n);
    int *idx_buf = (int *) malloc(sizeof(int) * n_thread * n);
    DTYPE *dist2_buf = (DTYPE *) malloc(sizeof(DTYPE) * n_thread * n);
    ASSERT_PRINTF(idx_buf != NULL && dist2_buf != NULL, "Failed to allocate work arrays for FSAI_exact_KNN()\n");
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *idx = idx_buf + tid * n;
        DTYPE *dist2 = dist2_buf + tid * n;
        // Do the large chunk of work first
        #pragma omp for schedule(dynamic, 256)
        for (int i = n - 1; i >= 0; i--)
        {
            int nn_cnt_i = 0;
            int *nn_idx_i = nn_idx + i * fsai_npt;
            if (i < fsai_npt)
            {
                nn_cnt_i = i + 1;
                for (int j = 0; j < nn_cnt_i; j++) nn_idx_i[j] = j;
            } else {
                nn_cnt_i = fsai_npt;
                for (int j = 0; j < i; j++) idx[j] = j;
                H2P_calc_pdist2_OMP(coord + i, ldc, 1, coord, ldc, i, pt_dim, dist2, n, 1);
                Qpart_DTYPE_int_key_val(dist2, idx, 0, i - 1, fsai_npt);
                memcpy(nn_idx_i, idx, sizeof(int) * fsai_npt);
                nn_idx_i[nn_cnt_i - 1] = i;
            }
            nn_cnt[i] = nn_cnt_i;
        }
    }
    free(idx_buf);
    free(dist2_buf);
}

// Check if segments [s1, s1+l1] and [s2, s2+l2] are neighbors
static inline int is_neighbor_segments(const DTYPE s1, const DTYPE l1, const DTYPE s2, const DTYPE l2)
{
    int is_neighbor = 0;
    DTYPE diff_s = DABS(s2 - s1);
    if (s1 <= s2) is_neighbor = (diff_s / l1 < 1.00001);
    else is_neighbor = (diff_s / l2 < 1.00001);
    return is_neighbor;
}

// Select k approximate nearest neighbors for each point using the tree structure 
// in H2Pack s.t. the indices of neighbors are smaller than the index of the point 
// Input parameters:
//   fsai_npt   : Maximum number of nonzeros in each row of the FSAI matrix (number of nearest neighbors)
//   n, pt_dim  : Number of points and point dimension
//   coord      : Size pt_dim * ldc, row major, each column is a point coordinate
//   ldc        : Leading dimension of coord, >= n
//   coord0_idx : Size n, for each point in coord, its index in the original input point set
//   h2mat      : H2Pack structure of the point set
// Output parameters:
//   nn_idx : Size n * fsai_npt, row major, indices of the nearest neighbors
//   nn_cnt : Size n, number of selected nearest neighbors for each point
static void FSAI_approx_KNN_with_H2P(
    const int fsai_npt, const int n, const int pt_dim, 
    const DTYPE *coord, const int ldc, const int *coord0_idx,
    H2Pack_p h2mat, int *nn_idx, int *nn_cnt
)
{
    int n_thread = omp_get_max_threads();
    memset(nn_cnt, 0, sizeof(int) * n);

    // 1. Find the highest level of H2 tree leaf nodes, set it as search level,
    //    and find the maximum number of points in a node at search level
    int search_lvl = h2mat->max_level;
    int *n_child = h2mat->n_child;
    for (int i = 0; i < h2mat->n_node; i++)
    {
        if (n_child[i] > 0) continue;
        int node_lvl = h2mat->node_level[i];
        if (node_lvl < search_lvl) search_lvl = node_lvl;
    }
    int max_sl_node_npt = 0;
    int search_lvl_nnode = h2mat->level_n_node[search_lvl];
    int *search_lvl_nodes = h2mat->level_nodes + search_lvl * h2mat->n_leaf_node;
    int *pt_cluster = h2mat->pt_cluster;  // Notice: each pair in pt_cluster is [start, end] not [start, end)
    for (int i = 0; i < search_lvl_nnode; i++)
    {
        int node = search_lvl_nodes[i];
        int node_npt = pt_cluster[2 * node + 1] - pt_cluster[2 * node] + 1;
        if (node_npt > max_sl_node_npt) max_sl_node_npt = node_npt;
    }

    // 2. Create a mapping for points in the H2 tree to the coord array
    //    OIPS in the comments == "original input point set"
    int n0 = h2mat->n_point;  // Number of points in the OIPS
    int *h2_coord_idx = h2mat->coord_idx;                   // For each point in the H2 tree, its index in the OIPS
    int *idx0_to_idx  = (int *) malloc(sizeof(int) * n0);   // For each point in the OIPS, its index in coord
    int *h2_to_idx    = (int *) malloc(sizeof(int) * n0);   // For each point in the H2 tree, its index in coord
    ASSERT_PRINTF(
        idx0_to_idx != NULL && h2_to_idx != NULL, 
        "Failed to allocate work arrays for FSAI_approx_KNN_with_H2P()"
    );
    for (int i = 0; i < n0; i++) idx0_to_idx[i] = -1;
    for (int i = 0; i < n;  i++) idx0_to_idx[coord0_idx[i]] = i;
    for (int i = 0; i < n0; i++) h2_to_idx[i] = idx0_to_idx[h2_coord_idx[i]];

    // 3. Find all neighbor nodes (including self) for each node at search level.
    //    Each node has at most 3^pt_dim neighbors.
    int max_neighbor = h2mat->max_neighbor;
    int *sl_node_neighbors = (int *) malloc(sizeof(int) * search_lvl_nnode * max_neighbor);
    int *sl_node_neighbor_cnt = (int *) malloc(sizeof(int) * search_lvl_nnode);
    ASSERT_PRINTF(
        sl_node_neighbors != NULL && sl_node_neighbor_cnt != NULL, 
        "Failed to allocate work arrays for FSAI_approx_KNN_with_H2P()"
    );
    DTYPE *enbox = h2mat->enbox;
    memset(sl_node_neighbor_cnt, 0, sizeof(int) * search_lvl_nnode);
    for (int i = 0; i < search_lvl_nnode; i++)
    {
        int node_i = search_lvl_nodes[i];
        sl_node_neighbors[i * max_neighbor] = node_i;
        sl_node_neighbor_cnt[i] = 1;
    }
    for (int i = 0; i < search_lvl_nnode; i++)
    {
        int node_i = search_lvl_nodes[i];
        DTYPE *enbox_i = enbox + (2 * pt_dim) * node_i;
        for (int j = i + 1; j < search_lvl_nnode; j++)
        {
            int node_j = search_lvl_nodes[j];
            DTYPE *enbox_j = enbox + (2 * pt_dim) * node_j;
            int is_neighbor = 1;
            for (int d = 0; d < pt_dim; d++)
            {
                DTYPE sid = enbox_i[d];
                DTYPE lid = enbox_i[d + pt_dim];
                DTYPE sjd = enbox_j[d];
                DTYPE ljd = enbox_j[d + pt_dim];
                int is_neighbor_d = is_neighbor_segments(sid, lid, sjd, ljd);
                is_neighbor = is_neighbor && is_neighbor_d;
            }
            if (is_neighbor)
            {
                int cnt_i = sl_node_neighbor_cnt[i];
                int cnt_j = sl_node_neighbor_cnt[j];
                ASSERT_PRINTF(cnt_i < max_neighbor, "Node %d has more than %d neighbors\n", node_i, max_neighbor);
                ASSERT_PRINTF(cnt_j < max_neighbor, "Node %d has more than %d neighbors\n", node_j, max_neighbor);
                sl_node_neighbors[i * max_neighbor + cnt_i] = node_j;
                sl_node_neighbors[j * max_neighbor + cnt_j] = node_i;
                sl_node_neighbor_cnt[i]++;
                sl_node_neighbor_cnt[j]++;
            }
        }  // End of j loop
    }  // End of i loop

    // 4. Find the nearest neighbors from sl_node_neighbors for each point.
    //    The nearest neighbor candidates of each point should <= max_neighbor * max_sl_node_npt
    int max_nn_points = max_neighbor * max_sl_node_npt;
    int *idx_buf        = (int *)   malloc(sizeof(int)   * n_thread * max_nn_points * 2);
    char *flag_buf      = (char *)  malloc(sizeof(char)  * n_thread * n);
    DTYPE *dist2_buf    = (DTYPE *) malloc(sizeof(DTYPE) * n_thread * max_nn_points);
    DTYPE *nn_coord_buf = (DTYPE *) malloc(sizeof(DTYPE) * n_thread * max_nn_points * pt_dim);
    ASSERT_PRINTF(
        idx_buf != NULL && dist2_buf != NULL && nn_coord_buf != NULL,
        "Failed to allocate work arrays for FSAI_approx_KNN_with_H2P()"
    );
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        DTYPE *dist2 = dist2_buf + tid * max_nn_points;
        DTYPE *nn_coord = nn_coord_buf + tid * pt_dim * max_nn_points;
        int *nn_idx0 = idx_buf + tid * max_nn_points * 2;
        int *nn_idx1 = nn_idx0 + max_nn_points;
        char *flag = flag_buf + tid * n;
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < search_lvl_nnode; i++)
        {
            int node_i = search_lvl_nodes[i];
            int nn_cnt0 = 0;
            // (1) Put all points in node_s's neighbor nodes into a large candidate set
            for (int j = 0; j < sl_node_neighbor_cnt[i]; j++)
            {
                int node_j = sl_node_neighbors[i * max_neighbor + j];
                int node_j_pt_s = pt_cluster[2 * node_j];
                int node_j_pt_e = pt_cluster[2 * node_j + 1];
                for (int k = node_j_pt_s; k <= node_j_pt_e; k++)
                {
                    int idx_k = h2_to_idx[k];   // The k-th point in H2 tree, its index in coord
                    if (idx_k == -1) continue;
                    nn_idx0[nn_cnt0++] = idx_k;
                }
            }
            // (2) For each point in this node, find its KNN in the refined candidate set
            int node_i_pt_s = pt_cluster[2 * node_i];
            int node_i_pt_e = pt_cluster[2 * node_i + 1];
            for (int pt_j = node_i_pt_s; pt_j <= node_i_pt_e; pt_j++)
            {
                int idx_j = h2_to_idx[pt_j];   // The pt_j-th point in H2 tree, its index in coord
                if (idx_j == -1) continue;
                // All point indices in nn_idx0 are in coord, no need to translate index again
                // FSAI requires the NN points have indices < current point
                int nn_cnt1 = 0;
                for (int k = 0; k < nn_cnt0; k++)
                    if (nn_idx0[k] < idx_j) nn_idx1[nn_cnt1++] = nn_idx0[k];
                // If the number of NN candidates < fsai_npt, expand the candidate set with the first
                // (max_nn_points - nn_cnt1) points (this is the simplest way, but may miss some exact NNs)
                if ((nn_cnt1 < fsai_npt) && (idx_j >= fsai_npt))
                {
                    memset(flag, 0, sizeof(char) * n);
                    for (int k = 0; k < nn_cnt1; k++) flag[nn_idx1[k]] = 1;
                    for (int k = 0; k < idx_j; k++)
                    {
                        if (flag[k] == 1) continue;
                        nn_idx1[nn_cnt1++] = k;
                        if (nn_cnt1 == max_nn_points - 1) break;
                    }
                }
                int *nn_idx_j = nn_idx + idx_j * fsai_npt;
                if (idx_j < fsai_npt)
                {
                    nn_cnt[idx_j] = idx_j + 1;
                    for (int j = 0; j <= idx_j; j++) nn_idx_j[j] = j;
                } else {
                    int nn_cnt_j = (fsai_npt < nn_cnt1) ? fsai_npt : nn_cnt1;
                    nn_cnt[idx_j] = nn_cnt_j;
                    H2P_gather_matrix_columns(coord, ldc, nn_coord, nn_cnt1, pt_dim, nn_idx1, nn_cnt1);
                    H2P_calc_pdist2_OMP(coord + idx_j, ldc, 1, nn_coord, nn_cnt1, nn_cnt1, pt_dim, dist2, nn_cnt1, 1);
                    Qpart_DTYPE_int_key_val(dist2, nn_idx1, 0, nn_cnt1 - 1, nn_cnt_j);
                    memcpy(nn_idx_j, nn_idx1, sizeof(int) * nn_cnt_j);
                    nn_idx_j[nn_cnt_j - 1] = idx_j;
                }
            }  // End of pt_j loop
        }  // End of i loop
    }  // End of "#pragma omp parallel"

    // Sanity check
    int invalid_cnt = 0;
    for (int i = 0; i < n; i++)
        if (nn_cnt[i] < 1 || nn_cnt[i] > fsai_npt) invalid_cnt++;
    ASSERT_PRINTF(invalid_cnt == 0, "%d points have invalid NN count\n", invalid_cnt);

    free(sl_node_neighbors);
    free(sl_node_neighbor_cnt);
    free(idx0_to_idx);
    free(h2_to_idx);
    free(dist2_buf);
    free(flag_buf);
    free(nn_coord_buf);
    free(idx_buf);
}

// Build a Factoized Sparse Approximate Inverse (FSAI) preconditioner 
// for a kernel matrix K(X, X) + mu * I - P * P^T, where P is a low rank matrix
// Input parameters:
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   krnl_param : Pointer to kernel function parameter array
//   fsai_npt   : Maximum number of nonzeros in each row of the FSAI matrix
//   n, pt_dim  : Number of points and point dimension
//   coord      : Size pt_dim * ldc, row major, each column is a point coordinate
//   ldc        : Leading dimension of coord, >= n
//   coord0_idx : Size n, for each point in coord, its index in the original input point set
//   n1         : Number of columns in P
//   P          : Size n * n1, row major, each column is a low rank basis
//   mu         : Diagonal shift
//   h2mat      : Optional, pointer to an initialized H2Pack struct, used for FSAI KNN search
// Output parameters:
//   *{G, GT}_rowptr_ : CSR row pointer arrays of G and G^T
//   *{G, GT}_colidx_ : CSR column index arrays of G and G^T
//   *{G, GT}_val_    : CSR value arrays of G and G^T
//   *t_knn_          : Time spent on KNN search, if not NULL
//   *t_fsai_         : Time spent on FSAI matrix construction, if not NULL
//   *t_csr_          : Time spent on CSR matrix construction, if not NULL
void FSAI_precond_build_(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int fsai_npt,
    const int n, const int pt_dim, const DTYPE *coord, const int ldc, 
    const int *coord0_idx, const int n1, const DTYPE *P, const DTYPE mu, void *h2mat, 
    int **G_rowptr_,  int **G_colidx_,  DTYPE **G_val_, 
    int **GT_rowptr_, int **GT_colidx_, DTYPE **GT_val_,
    double *t_knn_, double *t_fsai_, double *t_csr_
)
{
    double st, et;

    int n_thread = omp_get_max_threads();
    int thread_mat_bufsize = fsai_npt * (pt_dim + fsai_npt + n1);
    int   *nn_idx = (int *)   malloc(sizeof(int)   * n * fsai_npt);
    int   *row    = (int *)   malloc(sizeof(int)   * n * fsai_npt);
    int   *col    = (int *)   malloc(sizeof(int)   * n * fsai_npt);
    int   *displs = (int *)   malloc(sizeof(int)   * (n + 1));
    DTYPE *val    = (DTYPE *) malloc(sizeof(DTYPE) * n * fsai_npt);
    DTYPE *matbuf = (DTYPE *) malloc(sizeof(DTYPE) * n_thread * thread_mat_bufsize);
    ASSERT_PRINTF(
        nn_idx != NULL && matbuf != NULL && row != NULL && 
        col != NULL && val != NULL && displs != NULL,
        "Failed to allocate work buffers for FSAI_precond_build_()\n"
    );

    // 1. Find fsai_npt nearest neighbors of each point
    st = get_wtime_sec();
    displs[0] = 0;
    if (h2mat != NULL)
    {
        H2Pack_p h2mat_ = (H2Pack_p) h2mat;
        FSAI_approx_KNN_with_H2P(fsai_npt, n, pt_dim, coord, ldc, coord0_idx, h2mat_, nn_idx, displs + 1);
    } else {
        FSAI_exact_KNN(fsai_npt, n, pt_dim, coord, ldc, nn_idx, displs + 1);
    }
    et = get_wtime_sec();
    if (t_knn_ != NULL) *t_knn_ = et - st;

    // 2. Build the FSAI COO matrix
    st = get_wtime_sec();
    BLAS_SET_NUM_THREADS(1);
    #pragma omp parallel if (n_thread > 1) num_threads(n_thread)
    {
        // In the first fsai_npt rows, each row has less than fsai_npt nonzeros;
        // after that, each row has exactly fsai_npt nonzeros. Using a static
        // partitioning scheme should be good enough.
        int tid = omp_get_thread_num();
        DTYPE *thread_matbuf = matbuf + tid * thread_mat_bufsize;
        DTYPE *nn_coord = thread_matbuf;
        DTYPE *tmpK     = nn_coord + fsai_npt * pt_dim;
        DTYPE *Pnn      = tmpK     + fsai_npt * fsai_npt;
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
        {
            int nn_cnt_i  = displs[i + 1];
            int j_start   = i * fsai_npt;
            int *nn_idx_i = nn_idx + j_start;
            // row(idx + (1 : nn_cnt_i)) = i;
            // col(idx + (1 : nn_cnt_i)) = nn;
            // Xnn = X(nn, :);
            for (int j = j_start; j < j_start + nn_cnt_i; j++)
            {
                row[j] = i;
                col[j] = nn_idx_i[j - j_start];
            }
            H2P_gather_matrix_columns(coord, ldc, nn_coord, nn_cnt_i, pt_dim, nn_idx_i, nn_cnt_i);
            // tmpK = kernel(Xnn, Xnn) + mu * eye(nn_cnt_i);
            // tmpK = tmpK - P(nn, :) * P(nn, :)';
            krnl_eval(nn_coord, nn_cnt_i, nn_cnt_i, nn_coord, nn_cnt_i, nn_cnt_i, krnl_param, tmpK, nn_cnt_i);
            for (int j = 0; j < nn_cnt_i; j++) tmpK[j * nn_cnt_i + j] += mu;
            if (n1 > 0)
            {
                for (int j = j_start; j < j_start + nn_cnt_i; j++)
                {
                    int j0 = j - j_start;
                    memcpy(Pnn + j0 * n1, P + nn_idx_i[j0] * n1, sizeof(DTYPE) * n1);
                }
                CBLAS_SYRK(CblasRowMajor, CblasLower, CblasNoTrans, nn_cnt_i, n1, -1.0, Pnn, n1, 1.0, tmpK, nn_cnt_i);
            }
            // tmpU = [zeros(nn_cnt_i-1, 1); 1];
            // tmpY = tmpK \ tmpU;
            DTYPE *tmpU = val + j_start;
            memset(tmpU, 0, sizeof(DTYPE) * nn_cnt_i);
            tmpU[nn_cnt_i - 1] = 1.0;
            // The standard way is using LAPACK_ROW_MAJOR here, but since tmpK is 
            // symmetric, we use LAPACK_COL_MAJOR to avoid internal transpose
            #define FSAI_USE_CHOL
            #ifdef  FSAI_USE_CHOL
            // CblasRowMajor + CblasLower in SYRK, == LAPACK_COL_MAJOR + 'U'
            int info = LAPACK_POTRF(LAPACK_COL_MAJOR, 'U', nn_cnt_i, tmpK, nn_cnt_i);
            ASSERT_PRINTF(info == 0, "LAPACK_POTRF return %d\n", info);
            info = LAPACK_POTRS(LAPACK_COL_MAJOR, 'U', nn_cnt_i, 1, tmpK, nn_cnt_i, tmpU, nn_cnt_i);
            ASSERT_PRINTF(info == 0, "LAPACK_POTRS return %d\n", info);
            #else
            for (int ii = 0; ii < nn_cnt_i; ii++)
                for (int jj = ii + 1; jj < nn_cnt_i; jj++)
                    tmpK[ii * nn_cnt_i + jj] = tmpK[jj * nn_cnt_i + ii];
            int info = LAPACK_GETRF(LAPACK_COL_MAJOR, nn_cnt_i, nn_cnt_i, tmpK, nn_cnt_i, nn_idx_i);
            ASSERT_PRINTF(info == 0, "LAPACK_GETRF return %d\n", info);
            info = LAPACK_GETRS(LAPACK_COL_MAJOR, 'N', nn_cnt_i, 1, tmpK, nn_cnt_i, nn_idx_i, tmpU, nn_cnt_i);
            ASSERT_PRINTF(info == 0, "LAPACK_GETRS return %d\n", info);
            #endif
            // val(idx + (1 : nn_cnt_i)) = tmpY / sqrt(tmpY(nn_cnt_i));
            DTYPE scale_factor = 1.0 / DSQRT(tmpU[nn_cnt_i - 1]);
            for (int j = 0; j < nn_cnt_i; j++) tmpU[j] *= scale_factor;
        }  // End of i loop
    }  // End of "#pragma omp parallel"
    BLAS_SET_NUM_THREADS(n_thread);
    free(nn_idx);
    free(matbuf);
    et = get_wtime_sec();
    if (t_fsai_ != NULL) *t_fsai_ = et - st;

    // 3. Convert the FSAI COO matrix to CSR format
    st = get_wtime_sec();
    for (int i = 1; i <= n; i++) displs[i] += displs[i - 1];
    int nnz = displs[n];
    int   *row1 = (int *)   malloc(sizeof(int)   * nnz);
    int   *col1 = (int *)   malloc(sizeof(int)   * nnz);
    DTYPE *val1 = (DTYPE *) malloc(sizeof(DTYPE) * nnz);
    ASSERT_PRINTF(row1 != NULL && col1 != NULL && val1 != NULL, "Failed to allocate CSR matrix build buffer\n");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        int nnz_i = displs[i + 1] - displs[i];
        memcpy(row1 + displs[i], row + i * fsai_npt, sizeof(int) * nnz_i);
        memcpy(col1 + displs[i], col + i * fsai_npt, sizeof(int) * nnz_i);
        memcpy(val1 + displs[i], val + i * fsai_npt, sizeof(DTYPE) * nnz_i);
    }
    free(row);
    free(col);
    free(val);
    free(displs);
    int *G_rowptr = NULL, *G_colidx = NULL, *GT_rowptr = NULL, *GT_colidx = NULL;
    DTYPE *G_val = NULL, *GT_val = NULL;
    COO2CSR(n, n, nnz, row1, col1, val1, &G_rowptr,  &G_colidx,  &G_val);
    COO2CSR(n, n, nnz, col1, row1, val1, &GT_rowptr, &GT_colidx, &GT_val);
    *G_rowptr_  = G_rowptr;   *G_colidx_  = G_colidx;   *G_val_ = G_val;
    *GT_rowptr_ = GT_rowptr;  *GT_colidx_ = GT_colidx;  *GT_val_ = GT_val;
    free(row1);
    free(col1);
    free(val1);
    et = get_wtime_sec();
    if (t_csr_ != NULL) *t_csr_ = et - st;
}

// CSR SpMV for the G matrix in FSAI
static void FSAI_CSR_SpMV(
    const int nrow, const int *row_ptr, const int *col_idx, 
    const DTYPE *val, const DTYPE *x, DTYPE *y
)
{
    // In the first fsai_npt rows, row i has i nonzeros. After that, 
    // each row has exactly fsai_npt nonzeros. Using a static 
    // partitioning scheme should be good enough.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nrow; i++)
    {
        DTYPE res = 0;
        #pragma omp simd
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            res += val[j] * x[col_idx[j]];
        y[i] = res;
    }
}

// Apply the FSAI preconditioner
// Input parameters:
//   {G, GT}_rowptr : CSR row pointer arrays of G and G^T
//   {G, GT}_colidx : CSR column index arrays of G and G^T
//   {G, GT}_val    : CSR value arrays of G and G^T
//   n, x, t        : Size of the square G matrix and the 
//                    input vector x and work buffer t
// Output parameter:
//   y : Size n, output vector
void FSAI_precond_apply_(
    const int *G_rowptr, const int *G_colidx, const DTYPE *G_val,
    const int *GT_rowptr, const int *GT_colidx, const DTYPE *GT_val,
    const int n, const DTYPE *x, DTYPE *y, DTYPE *t
)
{
    FSAI_CSR_SpMV(n, G_rowptr,  G_colidx,  G_val,  x, t);
    FSAI_CSR_SpMV(n, GT_rowptr, GT_colidx, GT_val, t, y);
}

// ============== FSAI Preconditioner End =============== //

// Select k points from X using Farthest Point Sampling (FPS)
// Input parameters:
//   npt    : Number of points in coord
//   pt_dim : Dimension of each point
//   coord  : Matrix, size pt_dim-by-npt, coordinates of points
//   k      : Number of points to select, <= npt
// Output parameter:
//   idx : Vector, size min(k, npt), indices of selected points
void AFNi_FPS(const int npt, const int pt_dim, const DTYPE *coord, const int k, int *idx)
{
    DTYPE *workbuf = (DTYPE *) malloc(sizeof(DTYPE) * (2 * npt + pt_dim));
    DTYPE *center = workbuf;
    DTYPE *tmp_d  = center + pt_dim;
    DTYPE *min_d  = tmp_d  + npt;
    ASSERT_PRINTF(workbuf != NULL, "Failed to allocate work arrays for AFNi_FPS()\n");

    memset(center, 0, sizeof(DTYPE) * pt_dim);
    for (int i = 0; i < pt_dim; i++)
    {
        const DTYPE *coord_i = coord + i * npt;
        #pragma omp simd
        for (int j = 0; j < npt; j++) center[i] += coord_i[j];
        center[i] /= (DTYPE) npt;
    }

    int n_thread = omp_get_max_threads();
    volatile int   *min_d_idx = (volatile int *)   malloc(sizeof(int)   * n_thread);
    volatile DTYPE *min_d_val = (volatile DTYPE *) malloc(sizeof(DTYPE) * n_thread);
    int min_d_idx0;
    #pragma omp parallel num_threads(n_thread) shared(min_d_idx0)
    {
        int tid = omp_get_thread_num();
        int spos, len, epos;
        calc_block_spos_len(npt, n_thread, tid, &spos, &len);
        epos = spos + len;

        // (1) Calculate the distance of all points to the center
        for (int j = spos; j < epos; j++)
        {
            tmp_d[j] = 0.0;
            min_d[j] = DBL_MAX;
        }
        for (int d = 0; d < pt_dim; d++)
        {
            DTYPE center_d = center[d];
            const DTYPE *coord_d = coord + d * npt;
            #pragma omp simd
            for (int j = spos; j < epos; j++)
            {
                DTYPE diff = coord_d[j] - center_d;
                tmp_d[j] += diff * diff;
            }
        }
        // (2) Each thread find its local minimum
        int tmp_idx = spos;
        DTYPE tmp_val = tmp_d[spos];
        for (int j = spos + 1; j < epos; j++)
            if (tmp_d[j] < tmp_val) { tmp_val = tmp_d[j]; tmp_idx = j; }
        min_d_idx[tid] = tmp_idx;
        min_d_val[tid] = tmp_val;
        // (3) Find the global minimum distance to center
        #pragma omp barrier
        #pragma omp master
        {
            int tmp_idx2 = 0;
            DTYPE tmp_val2 = min_d_val[0];
            for (int t = 1; t < n_thread; t++)
                if (min_d_val[t] < tmp_val2) { tmp_val2 = min_d_val[t]; tmp_idx2 = t; }
            idx[0] = min_d_idx[tmp_idx2];
            min_d_idx0 = idx[0];
        }
        #pragma omp flush(min_d_idx0)

        // Find the rest k - 1 points
        for (int i = 1; i < k; i++)
        {
            #pragma omp barrier
            // (1) Calculate the distance of all points to the last selected point and update min_d
            memset(tmp_d + spos, 0, sizeof(DTYPE) * len);
            for (int d = 0; d < pt_dim; d++)
            {
                DTYPE last_d = coord[d * npt + min_d_idx0];
                const DTYPE *coord_d = coord + d * npt;
                #pragma omp simd
                for (int j = spos; j < epos; j++)
                {
                    DTYPE diff = coord_d[j] - last_d;
                    tmp_d[j] += diff * diff;
                }
            }
            for (int j = spos; j < epos; j++) min_d[j] = (tmp_d[j] < min_d[j]) ? tmp_d[j] : min_d[j];
            // (2) Each thread find its local maximum in min_d
            tmp_idx = spos;
            tmp_val = min_d[spos];
            for (int j = spos + 1; j < epos; j++)
                if (min_d[j] > tmp_val) { tmp_val = min_d[j]; tmp_idx = j; }
            min_d_idx[tid] = tmp_idx;
            min_d_val[tid] = tmp_val;
            // (3) Find the global maximum distance to the last selected point
            #pragma omp barrier
            #pragma omp master
            {
                int tmp_idx2 = 0;
                DTYPE tmp_val2 = min_d_val[0];
                for (int t = 1; t < n_thread; t++)
                    if (min_d_val[t] > tmp_val2) { tmp_val2 = min_d_val[t]; tmp_idx2 = t; }
                idx[i] = min_d_idx[tmp_idx2];
                min_d_idx0 = idx[i];
            }
            #pragma omp flush(min_d_idx0)
        }
    }

    free((void *) min_d_idx);
    free((void *) min_d_val);
    free(workbuf);
}

// Sample and scale ss_npt points, then use them to estimate the rank
// Input and output parameters are the same as AFNi_rank_est()
static int AFNi_rank_est_scaled(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const int ss_npt, const int n_rep
)
{
    int sample_r = 0, r = 0;
    DTYPE scale_factor = DPOW((DTYPE) ss_npt / (DTYPE) npt, 1.0 / (DTYPE) pt_dim);
    DTYPE *workbuf = (DTYPE *) malloc(sizeof(DTYPE) * (ss_npt * (pt_dim + 4 * ss_npt)));
    int *FPS_perm = (int *) malloc(sizeof(int) * ss_npt);
    int *sample_idx = (int *) malloc(sizeof(int) * ss_npt);
    ASSERT_PRINTF(
        workbuf != NULL && FPS_perm != NULL && sample_idx != NULL,
        "Failed to allocate work arrays for AFNi_rank_est_scaled()\n"
    );
    DTYPE *Xs   = workbuf;
    DTYPE *Ks   = Xs  + ss_npt * pt_dim;
    DTYPE *K11  = Ks  + ss_npt * ss_npt;
    DTYPE *K1   = K11 + ss_npt * ss_npt;
    DTYPE *Knys = K1  + ss_npt * ss_npt;
    for (int i_rep = 0; i_rep < n_rep; i_rep++)
    {
        // Randomly sample and scale ss_npt points
        H2P_rand_sample(npt, ss_npt, sample_idx, NULL);
        for (int i = 0; i < pt_dim; i++)
        {
            DTYPE *Xs_i = Xs + i * ss_npt;
            const DTYPE *coord_i = coord + i * npt;
            for (int j = 0; j < ss_npt; j++)
                Xs_i[j] = coord_i[sample_idx[j]] * scale_factor;
        }
        // Reorder points using FPS for Nystrom (use Ks as a temporary buffer)
        AFNi_FPS(ss_npt, pt_dim, Xs, ss_npt, FPS_perm);
        H2P_gather_matrix_columns(Xs, ss_npt, Ks, ss_npt, pt_dim, FPS_perm, ss_npt);
        memcpy(Xs, Ks, sizeof(DTYPE) * ss_npt * pt_dim);
        // Shift the Ks matrix to make Nystrom stable
        krnl_eval(Xs, ss_npt, ss_npt, Xs, ss_npt, ss_npt, krnl_param, Ks, ss_npt);
        DTYPE Ks_fnorm = calc_2norm(ss_npt * ss_npt, Ks);
        DTYPE nu = DSQRT((DTYPE) ss_npt) * (NEXTAFTER(Ks_fnorm, Ks_fnorm + 1.0) - Ks_fnorm);
        for (int i = 0; i < ss_npt; i++) Ks[i * ss_npt + i] += nu;
        // Binary search to find the minimal rank
        int rs = 1, re = ss_npt, rc;
        while (rs < re)
        {
            rc = (rs + re) / 2;
            // Since Ks is reordered by FPS, use the first rc  
            // rows & columns as Nystrom basis is the best
            // K11 = Ks(1 : rc, 1 : rc);
            // K1  = Ks(1 : rc, :);
            copy_matrix(sizeof(DTYPE), rc, rc,     Ks, ss_npt, K11, rc,     0);
            copy_matrix(sizeof(DTYPE), rc, ss_npt, Ks, ss_npt, K1,  ss_npt, 0);
            // Knys = K1' * (K11 \ K1);  % K1' * inv(K11) * K1
            int *ipiv = sample_idx;  // sample_idx is no longer needed and its size >= rc
            LAPACK_GETRF(LAPACK_ROW_MAJOR, rc, rc, K11, rc, ipiv);
            LAPACK_GETRS(LAPACK_ROW_MAJOR, 'N', rc, ss_npt, K11, rc, ipiv, K1, ss_npt);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, ss_npt, ss_npt, rc,
                1.0, Ks, ss_npt, K1, ss_npt, 0.0, Knys, ss_npt
            );
            DTYPE err_fnorm, relerr;
            calc_err_2norm(ss_npt * ss_npt, Ks, Knys, &Ks_fnorm, &err_fnorm);
            relerr = err_fnorm / Ks_fnorm;
            if (relerr < 0.1) re = rc; 
            else rs = rc + 1;
        }  // End of while (rs < re)
        sample_r = (rs > sample_r) ? rs : sample_r;
    }  // End of i_rep loop
    r = DCEIL((DTYPE) sample_r * (DTYPE) npt / (DTYPE) ss_npt);
    free(workbuf);
    free(FPS_perm);
    return r;
}

// Use the original points to estimate the rank for Nystrom
// Input and output parameters are the same as AFNi_rank_est()
static int AFNi_Nys_rank_est(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const DTYPE mu, const int max_k, const int ss_npt, const int n_rep
)
{
    int sample_r = 0, r = 0, max_k_ = max_k;
    if (max_k_ < ss_npt) max_k_ = ss_npt;
    DTYPE *workbuf = (DTYPE *) malloc(sizeof(DTYPE) * (max_k_ * pt_dim + max_k_ * max_k_ + max_k));
    int *sample_idx = (int *) malloc(sizeof(int) * max_k_);
    ASSERT_PRINTF(workbuf != NULL && sample_idx != NULL, "Failed to allocate work arrays in AFNi_Nys_rank_est()\n");
    DTYPE *Xs = workbuf;
    DTYPE *Ks = Xs + max_k_ * pt_dim;
    DTYPE *ev = Ks + max_k_ * max_k_;
    for (int i_rep = 0; i_rep < n_rep; i_rep++)
    {
        // Randomly sample and scale min(ss_npt, max_k_) points
        DTYPE scale_factor = 1.0;
        int ss_npt_ = max_k_;
        if (max_k_ > ss_npt)
        {
            H2P_rand_sample(npt, ss_npt, sample_idx, NULL);
            scale_factor = DPOW((DTYPE) ss_npt / (DTYPE) max_k_, 1.0 / (DTYPE) pt_dim);
            ss_npt_ = ss_npt;
        } else {
            H2P_rand_sample(npt, max_k_, sample_idx, NULL);
        }
        for (int i = 0; i < pt_dim; i++)
        {
            DTYPE *Xs_i = Xs + i * ss_npt_;
            const DTYPE *coord_i = coord + i * npt;
            for (int j = 0; j < ss_npt_; j++)
                Xs_i[j] = coord_i[sample_idx[j]] * scale_factor;
        }
        // Ks = kernel(Xs, Xs) + mu * eye(size(Xs, 1));
        // ev = eig(Ks);
        krnl_eval(Xs, ss_npt_, ss_npt_, Xs, ss_npt_, ss_npt_, krnl_param, Ks, ss_npt_);
        for (int i = 0; i < ss_npt; i++) Ks[i * ss_npt + i] += mu;
        LAPACK_SYEVD(LAPACK_ROW_MAJOR, 'N', 'U', ss_npt_, Ks, ss_npt_, ev);
        // sample_r = max(sample_r, sum(ev > 1.1 * mu));
        int rc = 0;
        DTYPE threshold = 1.1 * mu;
        for (int i = 0; i < ss_npt_; i++) if (ev[i] > threshold) rc++;
        sample_r = (rc > sample_r) ? rc : sample_r;
    }  // End of i_rep loop
    r = DCEIL((DTYPE) sample_r * (DTYPE) max_k_ / (DTYPE) ss_npt);
    free(sample_idx);
    free(workbuf);
    return r;
}

// Estimate the rank for AFN
// Input parameters:
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   krnl_param : Pointer to kernel function parameter array
//   npt        : Number of points in coord
//   pt_dim     : Dimension of each point
//   coord      : Matrix, size pt_dim-by-npt, coordinates of points
//   mu         : Scalar, diagonal shift of the kernel matrix
//   max_k      : Maximum global low-rank approximation rank (K11's size)
//   ss_npt     : Number of points in the sampling set
//   n_rep      : Number of repetitions for rank estimation
// Output parameter:
//   <ret> : Return the estimated rank
static int AFNi_rank_est(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const DTYPE mu, const int max_k, const int ss_npt, const int n_rep
)
{
    int r_scaled = AFNi_rank_est_scaled(krnl_eval, krnl_param, npt, pt_dim, coord, ss_npt, n_rep);
    if (r_scaled > max_k)
    {
        // Estimated rank > max K11 size, will use AFN, return immediately
        return r_scaled;
    } else {
        // Estimated rank is small, will use Nystrom instead of AFN, 
        // use the original points to better estimate the rank
        int r_unscaled = AFNi_Nys_rank_est(krnl_eval, krnl_param, npt, pt_dim, coord, mu, max_k, ss_npt, n_rep);
        return r_unscaled;
    }
}

// Build the AFN preconditioner
// See AFN_precond_build() for input parameters
static void AFNi_AFN_precond_build(
    AFN_precond_p AFN_precond, const DTYPE mu, DTYPE *K11, DTYPE *K12, 
    const DTYPE *coord2, const int ld2, const int pt_dim, 
    kernel_eval_fptr krnl_eval, void *krnl_param, const int fsai_npt, void *h2mat
)
{
    AFN_precond->is_nys = 0;
    AFN_precond->is_afn = 1;
    int n1 = AFN_precond->n1;
    int n2 = AFN_precond->n2;
    double st, et;
    double st0 = get_wtime_sec();

    st = get_wtime_sec();
    // K11 = K11 + mu * eye(n1);
    for (int i = 0; i < n1; i++) K11[i * n1 + i] += mu;
    // invL = inv(chol(K11, 'lower'));
    DTYPE *invL = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    ASSERT_PRINTF(invL != NULL, "Failed to allocate invL of size %d x %d\n", n1, n1);
    int info = 0;
    info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', n1, K11, n1);
    ASSERT_PRINTF(info == 0, "LAPACK_POTRF failed, info = %d\n", info);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n1 * n1; i++) invL[i] = 0;
    for (int i = 0; i < n1; i++) invL[i * n1 + i] = 1.0;
    CBLAS_TRSM(
        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, 
        CblasNonUnit, n1, n1, 1.0, K11, n1, invL, n1
    );
    // Build V21 s.t. V21 * V21' == K12' * (K11 \ K12)
    // V21 = K12' * invL';
    AFN_precond->afn_K12 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n2);
    ASSERT_PRINTF(AFN_precond->afn_K12 != NULL, "Failed to allocate afn_K12 of size %d x %d\n", n1, n2);
    #pragma omp parallel for schedule(static) 
    for (int i = 0; i < n1 * n2; i++) AFN_precond->afn_K12[i] = K12[i];
    DTYPE *V21 = (DTYPE *) malloc(sizeof(DTYPE) * n2 * n1);
    ASSERT_PRINTF(V21 != NULL, "Failed to allocate V21 of size %d x %d\n", n2, n1);
    CBLAS_GEMM(
        CblasRowMajor, CblasTrans, CblasTrans, n2, n1, n1, 
        1.0, K12, n2, invL, n1, 0.0, V21, n1
    );
    free(invL);

    // Copy K12 and invert K11, currently K11 is overwritten by its Cholesky factor
    DTYPE *invK11 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    DTYPE *K12_   = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n2);
    ASSERT_PRINTF(invK11 != NULL, "Failed to allocate invK11 of size %d x %d\n", n1, n1);
    ASSERT_PRINTF(K12_   != NULL, "Failed to allocate K12_ of size %d x %d\n", n1, n2);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n1; i++)
    {
        DTYPE *K11_i    = K11    + i * n1;
        DTYPE *invK11_i = invK11 + i * n1;
        DTYPE *K12_i    = K12    + i * n2;
        DTYPE *K12_i_   = K12_   + i * n2;
        memcpy(invK11_i, K11_i, sizeof(DTYPE) * n1);
        memcpy(K12_i_,   K12_i, sizeof(DTYPE) * n2);
    }
    info = LAPACK_POTRI(LAPACK_ROW_MAJOR, 'L', n1, invK11, n1);
    ASSERT_PRINTF(info == 0, "LAPACK_POTRI failed, info = %d\n", info);
    // POTRI only returns the lower/upper triangular part, remember to symmetrize
    for (int i = 0; i < n1 - 1; i++)
        for (int j = i + 1; j < n1; j++)
            invK11[i * n1 + j] = invK11[j * n1 + i];
    AFN_precond->afn_invK11 = invK11;
    AFN_precond->afn_K12    = K12_;
    et = get_wtime_sec();
    AFN_precond->t_afn_mat = et - st;

    // FSAI for S = K22 - V21 * V21'
    int *G_rowptr = NULL, *G_colidx = NULL, *GT_rowptr = NULL, *GT_colidx = NULL;
    DTYPE *G_val = NULL, *GT_val = NULL;
    int *coord0_idx = AFN_precond->perm + n1;
    FSAI_precond_build_(
        krnl_eval, krnl_param, fsai_npt, 
        n2, pt_dim, coord2, n1 + n2, 
        coord0_idx, n1, V21, mu, h2mat, 
        &G_rowptr, &G_colidx, &G_val, 
        &GT_rowptr, &GT_colidx, &GT_val, 
        &AFN_precond->t_afn_knn, &AFN_precond->t_afn_fsai, &AFN_precond->t_afn_csr
    );
    AFN_precond->afn_G_rowptr  = G_rowptr;
    AFN_precond->afn_G_colidx  = G_colidx;
    AFN_precond->afn_G_val     = G_val;
    AFN_precond->afn_GT_rowptr = GT_rowptr;
    AFN_precond->afn_GT_colidx = GT_colidx;
    AFN_precond->afn_GT_val    = GT_val;
    free(V21);

    double et0 = get_wtime_sec();
    AFN_precond->t_afn = et0 - st0;
}

// Construct an AFN preconditioner for a kernel matrix
void AFN_precond_build(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const DTYPE mu, const int max_k, const int ss_npt,
    const int fsai_npt, void *h2mat, AFN_precond_p *AFN_precond_
)
{
    AFN_precond_p AFN_precond = (AFN_precond_p) malloc(sizeof(AFN_precond_s));
    memset(AFN_precond, 0, sizeof(AFN_precond_s));
    double st, et, st0, et0;
    st0 = get_wtime_sec();

    // 1. Estimate the numerical low rank of the kernel matrix + diagonal shift
    st = get_wtime_sec();
    int n_rep = 3;
    int est_rank = AFNi_rank_est(krnl_eval, krnl_param, npt, pt_dim, coord, mu, max_k, ss_npt, n_rep);
    int n  = npt;
    int n1 = (est_rank < max_k) ? est_rank : max_k;
    int n2 = n - n1;
    AFN_precond->n  = n;
    AFN_precond->n1 = n1;
    AFN_precond->n2 = n2;
    AFN_precond->px = (DTYPE *) malloc(sizeof(DTYPE) * n);
    AFN_precond->py = (DTYPE *) malloc(sizeof(DTYPE) * n);
    AFN_precond->t1 = (DTYPE *) malloc(sizeof(DTYPE) * n);
    AFN_precond->t2 = (DTYPE *) malloc(sizeof(DTYPE) * n);
    ASSERT_PRINTF(
        AFN_precond->px != NULL && AFN_precond->py != NULL && 
        AFN_precond->t1 != NULL && AFN_precond->t2 != NULL,
        "Failed to allocate AFN preconditioner matvec buffers\n"
    );
    et = get_wtime_sec();
    AFN_precond->t_rankest = et - st;
    AFN_precond->est_rank = est_rank;

    // 2. Use FPS to select n1 points, swap them to the front
    st = get_wtime_sec();
    AFN_precond->perm = (int *) malloc(sizeof(int) * n);
    uint8_t *flag = (uint8_t *) malloc(sizeof(uint8_t) * n);
    DTYPE *coord_perm = (DTYPE *) malloc(sizeof(DTYPE) * npt * pt_dim);
    ASSERT_PRINTF(
        AFN_precond->perm != NULL && flag != NULL && coord_perm != NULL,
        "Failed to allocate AFN preconditioner FPS buffers\n"
    );
    int *perm = AFN_precond->perm;
    AFNi_FPS(npt, pt_dim, coord, n1, perm);
    memset(flag, 0, sizeof(uint8_t) * n);
    for (int i = 0; i < n1; i++) flag[perm[i]] = 1;
    int idx = n1;
    for (int i = 0; i < n; i++)
        if (flag[i] == 0) perm[idx++] = i;
    H2P_gather_matrix_columns(coord, npt, coord_perm, npt, pt_dim, perm, npt);
    et = get_wtime_sec();
    AFN_precond->t_fps= et - st;

    // 3. Calculate K11 and K12 used for both Nystrom and AFN
    st = get_wtime_sec();
    DTYPE *coord_n1 = coord_perm;
    DTYPE *coord_n2 = coord_perm + n1;
    DTYPE *K11 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    DTYPE *K12 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n2);
    int n_thread = omp_get_max_threads();
    ASSERT_PRINTF(
        K11 != NULL && K12 != NULL,
        "Failed to allocate AFN preconditioner K11/K12 buffers\n"
    );
    H2P_eval_kernel_matrix_OMP(
        krnl_eval, krnl_param, 
        coord_n1, n, n1, coord_n1, n, n1, 
        K11, n1, n_thread
    );
    H2P_eval_kernel_matrix_OMP(
        krnl_eval, krnl_param, 
        coord_n1, n, n1, coord_n2, n, n2, 
        K12, n2, n_thread
    );
    et = get_wtime_sec();
    AFN_precond->t_K11K12 = et - st;

    // 4. Build the Nystrom or AFN preconditioner
    if (est_rank < max_k)
    {
        st = get_wtime_sec();
        AFN_precond->is_afn = 0;
        AFN_precond->is_nys = 1;
        Nys_precond_build_(mu, n1, n2, K11, K12, &AFN_precond->nys_M, &AFN_precond->nys_U);
        et = get_wtime_sec();
        AFN_precond->t_nys = et - st;
    } else {
        AFNi_AFN_precond_build(
            AFN_precond, mu, K11, K12, coord_n2, n, pt_dim, 
            krnl_eval, krnl_param, fsai_npt, h2mat
        );
    }

    free(flag);
    free(coord_perm);
    free(K11);
    free(K12);

    et0 = get_wtime_sec();
    AFN_precond->t_build = et0 - st0;
    *AFN_precond_ = AFN_precond;
}

// Destroy an initialized AFN_precond struct
void AFN_precond_destroy(AFN_precond_p *AFN_precond_)
{
    AFN_precond_p AFN_precond = *AFN_precond_;
    if (AFN_precond == NULL) return;
    free(AFN_precond->perm);
    free(AFN_precond->px);
    free(AFN_precond->py);
    free(AFN_precond->t1);
    free(AFN_precond->t2);
    free(AFN_precond->nys_U);
    free(AFN_precond->nys_M);
    free(AFN_precond->afn_G_rowptr);
    free(AFN_precond->afn_GT_rowptr);
    free(AFN_precond->afn_G_colidx);
    free(AFN_precond->afn_GT_colidx);
    free(AFN_precond->afn_G_val);
    free(AFN_precond->afn_GT_val);
    free(AFN_precond->afn_invK11);
    free(AFN_precond->afn_K12);
    free(AFN_precond);
    *AFN_precond_ = NULL;
}

// Apply an AFN preconditioner to a vector
void AFN_precond_apply(AFN_precond_p AFN_precond, const DTYPE *x, DTYPE *y)
{
    if (AFN_precond == NULL) return;
    int   n     = AFN_precond->n;
    int   n1    = AFN_precond->n1;
    int   n2    = AFN_precond->n2;
    int   *perm = AFN_precond->perm;
    DTYPE *px   = AFN_precond->px;
    DTYPE *py   = AFN_precond->py;
    DTYPE *t1   = AFN_precond->t1;
    DTYPE *t2   = AFN_precond->t2;
    double st = get_wtime_sec();

    // px = x(perm);
    for (int i = 0; i < n; i++) px[i] = x[perm[i]];
    if (AFN_precond->is_nys)
        Nys_precond_apply_(n1, n, AFN_precond->nys_M, AFN_precond->nys_U, px, py, t1);
    if (AFN_precond->is_afn)
    {
        DTYPE *afn_invK11 = AFN_precond->afn_invK11;
        DTYPE *afn_K12    = AFN_precond->afn_K12;
        // x1 = px(1 : n1, :);
        // x2 = px(n1+1 : n, :);
        // t11 = iK11 * x1;  % Size n1
        DTYPE *x1 = px, *x2 = px + n1;
        DTYPE *t11 = t1, *t12 = t1 + n1;
        CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n1, n1, 1.0, afn_invK11, n1, x1, 1, 0.0, t11, 1);
        // t12 = x2 - K12' * t11;  % Size n2
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n2; i++) t12[i] = x2[i];
        CBLAS_GEMV(CblasRowMajor, CblasTrans, n1, n2, -1.0, afn_K12, n2, t11, 1, 1.0, t12, 1);
        // t22 = G  * t12;  % Size n2
        // y2  = G' * t22;
        DTYPE *y1 = py, *y2 = py + n1;
        DTYPE *t21 = t2, *t22 = t2 + n1;
        FSAI_precond_apply_(
            AFN_precond->afn_G_rowptr,  AFN_precond->afn_G_colidx,  AFN_precond->afn_G_val,
            AFN_precond->afn_GT_rowptr, AFN_precond->afn_GT_colidx, AFN_precond->afn_GT_val,
            n2, t12, y2, t22
        );
        // t21 = x1 - K12 * y2;  % Size n1
        for (int i = 0; i < n1; i++) t21[i] = x1[i];
        CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n1, n2, -1.0, afn_K12, n2, y2, 1, 1.0, t21, 1);
        // y1 = iK11 * t21;
        CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n1, n1, 1.0, afn_invK11, n1, t21, 1, 0.0, y1, 1);
        // py = [y1; y2];
    }
    // y(perm) = py;
    for (int i = 0; i < n; i++) y[perm[i]] = py[i];

    double et = get_wtime_sec();
    AFN_precond->t_apply += et - st;
    AFN_precond->n_apply++;
}

void AFN_precond_print_stat(AFN_precond_p AFN_precond)
{
    if (AFN_precond == NULL) return;
    printf("AFN preconditioner build time   = %.3f s\n", AFN_precond->t_build);
    printf("  * Rank estimation             = %.3f s\n", AFN_precond->t_rankest);
    printf("  * FPS select & permute        = %.3f s\n", AFN_precond->t_fps);
    printf("  * Build K11 and K12 blocks    = %.3f s\n", AFN_precond->t_K11K12);
    printf("  * Build Nystrom precond       = %.3f s\n", AFN_precond->t_nys);
    printf("  * Build AFN precond           = %.3f s\n", AFN_precond->t_afn);
    printf("    * Matrix operations         = %.3f s\n", AFN_precond->t_afn_mat);
    printf("    * FSAI KNN search           = %.3f s\n", AFN_precond->t_afn_knn);
    printf("    * FSAI COO matrix build     = %.3f s\n", AFN_precond->t_afn_fsai);
    printf("    * FASI COO matrix to CSR    = %.3f s\n", AFN_precond->t_afn_csr);
    if (AFN_precond->n_apply > 0)
    {
        double t_apply_avg = AFN_precond->t_apply / (double) AFN_precond->n_apply;
        printf("Apply preconditioner: %d times, per apply: %.3f s\n", AFN_precond->n_apply, t_apply_avg);
    }
}