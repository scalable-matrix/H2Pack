#ifndef __H2PACK_DENSE_MAT_H__
#define __H2PACK_DENSE_MAT_H__

#include "H2Pack_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// A simple dense matrix structure with some basic operations
struct H2P_dense_mat
{
    int   nrow;   // Number of rows
    int   ncol;   // Number of columns
    int   ld;     // Leading dimension, >= ncol
    int   size;   // Size of data, >= nrow * ncol
    DTYPE *data;  // Matrix data
};
typedef struct H2P_dense_mat* H2P_dense_mat_t;

// Initialize a H2P_dense_mat structure
// Input parameters:
//   nrow : Number of rows of the new dense matrix
//   ncol : Number of columns of the new dense matrix
// Output parameter:
//   mat_ : Initialized H2P_dense_mat structure
void H2P_dense_mat_init(H2P_dense_mat_t *mat_, const int nrow, const int ncol);

// Destroy a H2P_dense_mat structure
// Input parameter:
//   mat : H2P_dense_mat structure to be destroyed 
void H2P_dense_mat_destroy(H2P_dense_mat_t mat);

// Copy a block of a dense matrix to another dense matrix
// WARNING: This function DOES NOT perform sanity check!
// Input parameters:
//   src_mat  : Source matrix
//   src_srow : Starting row in the source matrix 
//   src_scol : Starting column in the source matrix 
//   dst_srow : Starting row in the destination matrix 
//   dst_scol : Starting column in the destination matrix 
//   nrow     : Number of rows to be copied
//   ncol     : Number of columns to be copied
// Output parameter:
//   dst_mat : Destination matrix
void H2P_dense_mat_copy_block(
    H2P_dense_mat_t src_mat, H2P_dense_mat_t dst_mat,
    const int src_srow, const int src_scol, 
    const int dst_srow, const int dst_scol, 
    const int nrow, const int ncol
);

// Transpose a dense matrix
// Input parameter:
//   mat : H2P_dense_mat structure to be transposed
// Output parameter:
//   mat : Transposed H2P_dense_mat structure
void H2P_dense_mat_transpose(H2P_dense_mat_t mat);

// Permute rows in a H2P_dense_mat structure
// WARNING: This function DOES NOT perform sanity check!
// Input parameter:
//   mat : H2P_dense_mat structure to be permuted
//   p   : Permutation array. After permutation, the i-th row is the p[i]-th row
//         in the original matrix
// Output parameter:
//   mat : H2P_dense_mat structure with permuted row
void H2P_dense_mat_permute_rows(H2P_dense_mat_t mat, const int *p);

// Print a H2P_dense_mat structure, for debugging
// Input parameter:
//   mat : H2P_dense_mat structure to be printed
void H2P_dense_mat_print(H2P_dense_mat_t mat);

#ifdef __cplusplus
}
#endif

#endif
