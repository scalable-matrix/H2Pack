#ifndef __H2PACK_UTILS_H__
#define __H2PACK_UTILS_H__

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "utils.h"

// Functions in this file are used in different H2Pack modules and they are intended 
// for H2Pack internal use only, should not be included by any external program.

#ifdef __cplusplus
extern "C" {
#endif

// Check if two boxes are admissible 
// Input parameters:
//   box0, box1 : Box data, [0 : pt_dim-1] are coordinates of the box corner which is 
//                closest to the original point, [pt_dim : 2*pt_dim-1] are box length
//   pt_dim     : Dimension of point coordinate
//   alpha      : Admissible pair coefficient
// Output parameter:
//   <return>   : If two boxes are admissible 
int H2P_check_box_admissible(const DTYPE *box0, const DTYPE *box1, const int pt_dim, const DTYPE alpha);

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
);

// Copy a block from a matrix to another matrix
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
);

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
);

// Check if a coordinate is in box [-L/2, L/2]^pt_dim
// Input parameters:
//   pt_dim : Dimension of point coordinate
//   coord  : Coordinate
//   L      : Box size
// Output parameter:
//   <return> : If the coordinate is in the box
int point_in_box(const int pt_dim, DTYPE *coord, DTYPE L);

// Generate a random sparse matrix A for calculating y^T := A^T * x^T,
// where A is a random sparse matrix that has no more than max_nnz_col 
// random +1/-1 nonzeros in each column with random position, x and y 
// are row-major matrices. Each row of x/y is a column of x^T/y^T. We 
// can just use SpMV to calculate y^T(:, i) := A^T * x^T(:, i).
// Input parameters:
//   max_nnz_col : Maximum number of nonzeros in each column of A
//   k, n        : A is k-by-n sparse matrix
// Output parameters:
//   A_valbuf : Buffer for storing nonzeros of A^T
//   A_idxbuf : Buffer for storing CSR row_ptr and col_idx arrays of A^T. 
//              A_idxbuf->data[0 : n] stores row_ptr, A_idxbuf->data[n+1 : end]
//              stores col_idx.
void H2P_gen_rand_sparse_mat_trans(
    const int max_nnz_col, const int k, const int n, 
    H2P_dense_mat_t A_valbuf, H2P_int_vec_t A_idxbuf
);

// Calculate y^T := A^T * x^T, where A is a sparse matrix, 
// x and y are row-major matrices. Since x/y is row-major, 
// each of its row is a column of x^T/y^T. We can just use SpMV
// to calculate y^T(:, i) := A^T * x^T(:, i).
// Input parameters:
//   m, n, k  : x is m-by-k matrix, A is k-by-n sparse matrix
//   A_valbuf : Buffer for storing nonzeros of A^T
//   A_idxbuf : Buffer for storing CSR row_ptr and col_idx arrays of A^T. 
//              A_idxbuf->data[0 : n] stores row_ptr, A_idxbuf->data[n+1 : end]
//              stores col_idx.
//   x, ldx   : m-by-k row-major dense matrix, leading dimension ldx
//   ldy      : Leading dimension of y
// Output parameter:
//   y : m-by-n row-major dense matrix, leading dimension ldy
void H2P_calc_sparse_mm_trans(
    const int m, const int n, const int k,
    H2P_dense_mat_t A_valbuf, H2P_int_vec_t A_idxbuf,
    DTYPE *x, const int ldx, DTYPE *y, const int ldy
);

// Generate normal distribution random number, Marsaglia polar method
// Input parameters:
//   mu, sigma : Normal distribution parameters
//   nelem     : Number of random numbers to be generated
// Output parameter:
//   x : Array, size nelem, generated random numbers
void H2P_gen_normal_distribution(const DTYPE mu, const DTYPE sigma, const int nelem, DTYPE *x);

// Quick sorting an integer key-value pair array by key
// Input parameters:
//   key, val : Array, size >= r+1, key-value pairs
//   l, r     : Sort range: [l, r-1]
// Output parameters:
//   key, val : Array, size >= r+1, sorted key-value pairs
void qsort_int_key_val(int *key, int *val, int l, int r);

// Convert an integer COO matrix to a CSR matrix 
// Input parameters:
//   nrow          : Number of rows
//   nnz           : Number of nonzeros in the matrix
//   row, col, val : Size nnz, COO matrix
// Output parameters:
//   row_ptr, col_idx, val_ : Size nrow+1, nnz, nnz, CSR matrix
void H2P_int_COO_to_CSR(
    const int nrow, const int nnz, const int *row, const int *col, 
    const int *val, int *row_ptr, int *col_idx, int *val_
);

// Get element A(row, col) from CSR matrix A
// Input parameters:
//   row_ptr, col_idx, val : CSR matrix arrays
//   row, col : Target position
// Output parameter:
//   <return> : A(row, col) if exists, 0 if not
int H2P_get_int_CSR_elem(
    const int *row_ptr, const int *col_idx, const int *val,
    const int row, const int col
);

// Get B{node0, node1} from a H2Pack structure
// Input parameters:
//   h2pack     : H2Pack structure
//   node{0, 1} : Target node pair
// Output parameter:
//   Bij : Target B{node0, node1} block
void H2P_get_Bij_block(H2Pack_t h2pack, const int node0, const int node1, H2P_dense_mat_t Bij);


// Get D{node0, node1} from a H2Pack structure
// Input parameters:
//   h2pack     : H2Pack structure
//   node{0, 1} : Target node pair
// Output parameter:
//   Dij : Target D{node0, node1} block
void H2P_get_Dij_block(H2Pack_t h2pack, const int node0, const int node1, H2P_dense_mat_t Dij);

// Partition work units into multiple blocks s.t. each block has 
// approximately the same amount of work
// Input parameters:
//   n_work     : Number of work units
//   work_sizes : Size n_work, sizes of work units
//   total_size : Sum of work_sizes
//   n_block    : Maximum number of blocks to partition the work units into
// Output parameter:
//   blk_displs : H2P_int_vec structure, size <= n_block+1, indices of the 
//                first work unit in each block
void H2P_partition_workload(
    const int n_work,  const size_t *work_sizes, const size_t total_size, 
    const int n_block, H2P_int_vec_t blk_displs
);

// Transpose a matrix
// Input parameters:
//   n_thread : Number of threads to use
//   src_nrow : Number of rows of the source matrix
//   src_ncol : Number of columns of the source matrix
//   src      : Source matrix, size >= src_nrow * lds
//   lds      : Leading dimension of source matrix
//   ldd      : Leading dimension of destination matrix
// Output parameter:
//   dst : Destination matrix
void H2P_transpose_dmat(
    const int n_thread, const int src_nrow, const int src_ncol, 
    const DTYPE *src, const int lds, DTYPE *dst, const int ldd
);

#ifdef __cplusplus
}
#endif

#endif

