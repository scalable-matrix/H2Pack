#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_utils.h"
#include "utils.h"

// Check if two boxes are admissible 
int H2P_check_box_admissible(const DTYPE *box0, const DTYPE *box1, const int pt_dim, const DTYPE alpha)
{
    for (int i = 0; i < pt_dim; i++)
    {
        // Radius of each box's i-th dimension
        DTYPE r0 = box0[pt_dim + i];
        DTYPE r1 = box1[pt_dim + i];
        // Center of each box's i-th dimension
        DTYPE c0 = box0[i] + 0.5 * r0;
        DTYPE c1 = box1[i] + 0.5 * r1;
        DTYPE min_r = MIN(r0, r1);
        DTYPE dist  = DABS(c0 - c1);
        if (dist >= alpha * min_r + 0.5 * (r0 + r1)) return 1;
    }
    return 0;
}

// Gather some columns from a matrix to another matrix
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

// Evaluate a kernel matrix with OpenMP parallelization
void H2P_eval_kernel_matrix_OMP(
    const void *krnl_param, kernel_eval_fptr krnl_eval, const int krnl_dim, 
    H2P_dense_mat_p x_coord, H2P_dense_mat_p y_coord, H2P_dense_mat_p kernel_mat
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
int H2P_point_in_box(const int pt_dim, DTYPE *coord, DTYPE L)
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

// Generate npt uniformly distributed random points in a ring 
// [-L1/2, L1/2]^pt_dim excluding [-L0/2, L0/2]^pt_dim 
void H2P_gen_coord_in_ring(const int npt, const int pt_dim, const DTYPE L0, const DTYPE L1, DTYPE *coord, const int ldc)
{
    const DTYPE semi_L1 = 0.5 * L1;
    DTYPE coord_i[8];
    ASSERT_PRINTF(pt_dim <= 8, "Temporary array too small (8 < %d)\n", pt_dim);
    for (int i = 0; i < npt; i++)
    {
        int flag = 0;
        while (flag == 0)
        {
            for (int j = 0; j < pt_dim; j++) coord_i[j] = (DTYPE) drand48() * L1 - semi_L1;
            if ((H2P_point_in_box(pt_dim, coord_i, L1) == 1) && (H2P_point_in_box(pt_dim, coord_i, L0) == 0))
            {
                flag = 1;
                for (int j = 0; j < pt_dim; j++) coord[j * ldc + i] = coord_i[j];
            }
        }
    }
}

// Generate a random sparse matrix A for calculating y^T := A^T * x^T
void H2P_gen_rand_sparse_mat_trans(
    const int max_nnz_col, const int k, const int n, 
    H2P_dense_mat_p A_valbuf, H2P_int_vec_p A_idxbuf
)
{
    // Note: we calculate y^T := A^T * x^T. Since x/y is row-major, 
    // each of its row is a column of x^T/y^T. We can just use SpMV
    // to calculate y^T(:, i) := A^T * x^T(:, i). 

    int rand_nnz_col = (max_nnz_col <= k) ? max_nnz_col : k;
    int nnz = n * rand_nnz_col;
    H2P_dense_mat_resize(A_valbuf, 1, nnz);
    H2P_int_vec_set_capacity(A_idxbuf, (n + 1) + nnz + k);
    DTYPE *val = A_valbuf->data;
    int *row_ptr = A_idxbuf->data;
    int *col_idx = row_ptr + (n + 1);
    int *flag = col_idx + nnz; 
    memset(flag, 0, sizeof(int) * k);
    for (int i = 0; i < nnz; i++) 
        val[i] = (DTYPE) (2.0 * (rand() & 1) - 1.0);
    for (int i = 0; i <= n; i++) 
        row_ptr[i] = i * rand_nnz_col;
    for (int i = 0; i < n; i++)
    {
        int cnt = 0;
        int *row_i_cols = col_idx + i * rand_nnz_col;
        while (cnt < rand_nnz_col)
        {
            int col = rand() % k;
            if (flag[col] == 0) 
            {
                flag[col] = 1;
                row_i_cols[cnt] = col;
                cnt++;
            }
        }
        for (int j = 0; j < rand_nnz_col; j++)
            flag[row_i_cols[j]] = 0;
    }
    A_idxbuf->length = (n + 1) + nnz;
}

// Calculate y^T := A^T * x^T, where A is a sparse matrix, x and y are row-major matrices
void H2P_calc_sparse_mm_trans(
    const int m, const int n, const int k,
    H2P_dense_mat_p A_valbuf, H2P_int_vec_p A_idxbuf,
    DTYPE *x, const int ldx, DTYPE *y, const int ldy
)
{
    const DTYPE *val = A_valbuf->data;
    const int *row_ptr = A_idxbuf->data;
    const int *col_idx = row_ptr + (n + 1);  // A is k-by-n
    // Doing a naive OpenMP CSR SpMM here is good enough, using MKL SpBLAS is actually
    // slower, probably due to the cost of optimizing the storage of sparse matrix
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
    {
        DTYPE *x_i = x + i * ldx;
        DTYPE *y_i = y + i * ldy;
        
        for (int j = 0; j < n; j++)
        {
            DTYPE res = 0.0;
            #pragma omp simd
            for (int l = row_ptr[j]; l < row_ptr[j+1]; l++)
                res += val[l] * x_i[col_idx[l]];
            y_i[j] = res;
        }
    }
}

// Generate normal distribution random number, Marsaglia polar method
void H2P_gen_normal_distribution(const DTYPE mu, const DTYPE sigma, const size_t nelem, DTYPE *x)
{
    DTYPE u1, u2, w, mult, x1, x2;
    for (size_t i = 0; i < nelem - 1; i += 2)
    {
        do 
        {
            u1 = (DTYPE) (drand48() * 2.0 - 1.0);
            u2 = (DTYPE) (drand48() * 2.0 - 1.0);
            w  = u1 * u1 + u2 * u2;
        } while (w >= 1.0 || w <= 1e-15);
        mult = DSQRT((-2.0 * DLOG(w)) / w);
        x1 = u1 * mult;
        x2 = u2 * mult;
        x[i]   = mu + sigma * x1;
        x[i+1] = mu + sigma * x2;
    }
    if (nelem % 2)
    {
        do 
        {
            u1 = (DTYPE) (drand48() * 2.0 - 1.0);
            u2 = (DTYPE) (drand48() * 2.0 - 1.0);
            w  = u1 * u1 + u2 * u2;
        } while (w >= 1.0 || w <= 1e-15);
        mult = DSQRT((-2.0 * DLOG(w)) / w);
        x1 = u1 * mult;
        x[nelem - 1] = mu + sigma * x1;
    }
}

// Quick sorting an integer key-value pair array by key
void H2P_qsort_int_key_val(int *key, int *val, int l, int r)
{
    int i = l, j = r, tmp_key, tmp_val;
    int mid_key = key[(l + r) / 2];
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
    if (i < r) H2P_qsort_int_key_val(key, val, i, r);
    if (j > l) H2P_qsort_int_key_val(key, val, l, j);
}

// Convert a integer COO matrix to a CSR matrix 
void H2P_int_COO_to_CSR(
    const int nrow, const int nnz, const int *row, const int *col, 
    const int *val, int *row_ptr, int *col_idx, int *val_
)
{
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
        val_[idx] = val[i];
        row_ptr[row[i]]++;
    }
    // Reset row_ptr
    for (int i = nrow; i >= 1; i--) row_ptr[i] = row_ptr[i - 1];
    row_ptr[0] = 0;
    // Sort the non-zeros in each row according to column indices
    #pragma omp parallel for
    for (int i = 0; i < nrow; i++)
        H2P_qsort_int_key_val(col_idx, val_, row_ptr[i], row_ptr[i + 1] - 1);
}

// Get the value of integer CSR matrix element A(row, col)
int H2P_get_int_CSR_elem(
    const int *row_ptr, const int *col_idx, const int *val,
    const int row, const int col
)
{
    int res = 0;
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++)
    {
        if (col_idx[i] == col) 
        {
            res = val[i];
            break;
        }
    }
    return res;
}

// Set the value of integer CSR matrix element A(row, col) to new_val
void H2P_set_int_CSR_elem(
    const int *row_ptr, const int *col_idx, int *val,
    const int row, const int col, const int new_val
)
{
    int has_element = 0;
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++)
    {
        if (col_idx[i] == col) 
        {
            val[i] = new_val;
            has_element = 1;
            break;
        }
    }
    if (has_element == 0) ERROR_PRINTF("CSR matrix element (%d, %d) not found, cannot be updated\n", row, col);
}

// Get B{node0, node1} from a H2Pack structure
void H2P_get_Bij_block(H2Pack_p h2pack, const int node0, const int node1, H2P_dense_mat_p Bij)
{
    int *B_p2i_rowptr = h2pack->B_p2i_rowptr;
    int *B_p2i_colidx = h2pack->B_p2i_colidx;
    int *B_p2i_val    = h2pack->B_p2i_val;
    int B_idx = H2P_get_int_CSR_elem(B_p2i_rowptr, B_p2i_colidx, B_p2i_val, node0, node1);
    int need_trans = 0, node0_ = node0, node1_ = node1;
    if (B_idx == 0)
    {
        ERROR_PRINTF("B{%d, %d} does not exist!\n", node0, node1);
        return;
    }
    if (B_idx < 0)
    {
        need_trans = 1;
        B_idx  = -B_idx;
        node0_ = node1;
        node1_ = node0;
    }
    B_idx--;
    int B_nrow = h2pack->B_nrow[B_idx];
    int B_ncol = h2pack->B_ncol[B_idx];
    H2P_dense_mat_resize(Bij, B_nrow, B_ncol);
    if (h2pack->BD_JIT == 0)
    {
        copy_matrix_block(sizeof(DTYPE), B_nrow, B_ncol, h2pack->B_data + h2pack->B_ptr[B_idx], B_ncol, Bij->data, B_ncol);
    } else {
        int   n_point     = h2pack->n_point;
        int   krnl_dim    = h2pack->krnl_dim;
        int   *pt_cluster = h2pack->pt_cluster;
        int   *node_level = h2pack->node_level;
        int   level0      = node_level[node0_];
        int   level1      = node_level[node1_];
        DTYPE *coord      = h2pack->coord;
        void  *krnl_param = h2pack->krnl_param;
        kernel_eval_fptr krnl_eval = h2pack->krnl_eval;
        H2P_dense_mat_p  *J_coord  = h2pack->J_coord;
        // (1) Two nodes are of the same level, compress on both sides
        if (level0 == level1)
        {
            krnl_eval(
                J_coord[node0_]->data, J_coord[node0_]->ncol, J_coord[node0_]->ncol,
                J_coord[node1_]->data, J_coord[node1_]->ncol, J_coord[node1_]->ncol,
                krnl_param, Bij->data, J_coord[node1_]->ncol * krnl_dim
            );
        }
        // (2) node1 is a leaf node and its level is higher than node0's level, 
        //     only compress on node0's side
        if (level0 > level1)
        {
            int pt_s1 = pt_cluster[2 * node1_];
            int pt_e1 = pt_cluster[2 * node1_ + 1];
            int node1_npt = pt_e1 - pt_s1 + 1;
            krnl_eval(
                J_coord[node0_]->data, J_coord[node0_]->ncol, J_coord[node0_]->ncol,
                coord + pt_s1, n_point, node1_npt, 
                krnl_param, Bij->data, node1_npt * krnl_dim
            );
        }
        // (3) node0 is a leaf node and its level is higher than node1's level, 
        //     only compress on node1's side
        if (level0 < level1)
        {
            int pt_s0 = pt_cluster[2 * node0_];
            int pt_e0 = pt_cluster[2 * node0_ + 1];
            int node0_npt = pt_e0 - pt_s0 + 1;
            krnl_eval(
                coord + pt_s0, n_point, node0_npt, 
                J_coord[node1_]->data, J_coord[node1_]->ncol, J_coord[node1_]->ncol,
                krnl_param, Bij->data, J_coord[node1_]->ncol * krnl_dim
            );
        }
    }  // End of "if (h2pack->BD_JIT == 0)"
    if (need_trans) Bij->ld = -Bij->ld;
}

// Get D{node0, node1} from a H2Pack structure
void H2P_get_Dij_block(H2Pack_p h2pack, const int node0, const int node1, H2P_dense_mat_p Dij)
{
    int *D_p2i_rowptr = h2pack->D_p2i_rowptr;
    int *D_p2i_colidx = h2pack->D_p2i_colidx;
    int *D_p2i_val    = h2pack->D_p2i_val;
    int D_idx = H2P_get_int_CSR_elem(D_p2i_rowptr, D_p2i_colidx, D_p2i_val, node0, node1);
    int need_trans = 0, node0_ = node0, node1_ = node1;
    if (D_idx == 0)
    {
        ERROR_PRINTF("D{%d, %d} does not exist!\n", node0, node1);
        return;
    }
    if (D_idx < 0)
    {
        need_trans = 1;
        D_idx  = -D_idx;
        node0_ = node1;
        node1_ = node0;
    }
    D_idx--;
    int D_nrow = h2pack->D_nrow[D_idx];
    int D_ncol = h2pack->D_ncol[D_idx];
    H2P_dense_mat_resize(Dij, D_nrow, D_ncol);
    if (h2pack->BD_JIT == 0)
    {
        copy_matrix_block(sizeof(DTYPE), D_nrow, D_ncol, h2pack->D_data + h2pack->D_ptr[D_idx], D_ncol, Dij->data, D_ncol);
    } else {
        int   n_point     = h2pack->n_point;
        int   krnl_dim    = h2pack->krnl_dim;
        int   *pt_cluster = h2pack->pt_cluster;
        int   pt_s0       = pt_cluster[2 * node0_];
        int   pt_s1       = pt_cluster[2 * node1_];
        int   pt_e0       = pt_cluster[2 * node0_ + 1];
        int   pt_e1       = pt_cluster[2 * node1_ + 1];
        int   node0_npt   = pt_e0 - pt_s0 + 1;
        int   node1_npt   = pt_e1 - pt_s1 + 1;
        DTYPE *coord      = h2pack->coord;
        h2pack->krnl_eval(
            coord + pt_s0, n_point, node0_npt,
            coord + pt_s1, n_point, node1_npt,
            h2pack->krnl_param, Dij->data, node1_npt * krnl_dim
        );
    }  // End of "if (h2pack->BD_JIT == 0)"
    if (need_trans) Dij->ld = -Dij->ld;
}

// Partition work units into multiple blocks s.t. each block has 
// approximately the same amount of work
void H2P_partition_workload(
    const int n_work,  const size_t *work_sizes, const size_t total_size, 
    const int n_block, H2P_int_vec_p blk_displs
)
{
    H2P_int_vec_set_capacity(blk_displs, n_block + 1);
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

// Transpose a DTYPE matrix
void H2P_transpose_dmat(
    const int n_thread, const int src_nrow, const int src_ncol, 
    const DTYPE *src, const int lds, DTYPE *dst, const int ldd
)
{
    if (n_thread == 1)
    {
        for (int i = 0; i < src_ncol; i++)
        {
            DTYPE *dst_irow = dst + i * ldd;
            for (int j = 0; j < src_nrow; j++)
                dst_irow[j] = src[j * lds + i];
        }
    } else {
        if (src_nrow > src_ncol)
        {
            #pragma omp parallel for if(n_thread > 1) num_threads(n_thread)
            for (int i = 0; i < src_ncol; i++)
            {
                DTYPE *dst_irow = dst + i * ldd;
                for (int j = 0; j < src_nrow; j++)
                    dst_irow[j] = src[j * lds + i];
            }
        } else {
            #pragma omp parallel num_threads(n_thread)
            {
                int tid = omp_get_thread_num();
                int blk_spos, blk_len;
                calc_block_spos_len(src_nrow, n_thread, tid, &blk_spos, &blk_len);
                for (int i = 0; i < src_ncol; i++)
                {
                    DTYPE *dst_irow = dst + i * ldd;
                    for (int j = blk_spos; j < blk_spos + blk_len; j++)
                        dst_irow[j] = src[j * lds + i];
                }
            }
        }  // End of "if (src_nrow > src_ncol)"
    }  // End of "if (n_thread == 1)"
}

// Shift the coordinates
void H2P_shift_coord(H2P_dense_mat_p coord, const DTYPE *shift, const DTYPE scale)
{
    for (int i = 0; i < coord->nrow; i++)
    {
        DTYPE *coord_dim_i = coord->data + i * coord->ld;
        #pragma omp simd
        for (int j = 0; j < coord->ncol; j++) coord_dim_i[j] += scale * shift[i];
    }
}
