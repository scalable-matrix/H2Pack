// ---------------------------------------------------------------
// @brief  : CSRPlus matrix header file 
// @author : Hua Huang <huangh223@gatech.edu>
//           Edmond Chow <echow@cc.gatech.edu>
// 
//     Copyright (c) 2017-2020 Georgia Institute of Technology      
// ----------------------------------------------------------------

#ifndef __CSRPLUS_H__
#define __CSRPLUS_H__

struct CSRP_mat
{
    // Standard CSR arrays and parameters
    int    nrow, ncol, nnz;
    int    *row_ptr;
    int    *col;
    double *val;
    
    // CSRPlus task partitioning information
    int    nblk;        // Number of non-zero blocks 
    int    nthread;     // Number of threads
    int    *nnz_spos;   // First nnz of a block
    int    *nnz_epos;   // Last  nnz (included) of a block
    int    *first_row;  // First row of a block
    int    *last_row;   // Last  row of a block
    int    *fr_intact;  // If the first row of a block is intact
    int    *lr_intact;  // If the last  row of a block is intact
    double *fr_res;     // Partial result of the first row 
    double *lr_res;     // Partial result of the last  row
};

typedef struct CSRP_mat  CSRP_mat_s;
typedef struct CSRP_mat* CSRP_mat_t;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a CSRP_mat structure using a COO matrix
// Note: This function assumes that the input COO matrix is not sorted
// Input parameters:
//   nrow, ncol, nnz : Number of rows, columns and non-zeros
//   row, col, val   : Row indices, column indices and values of non-zeros
// Output parameter:
//   *csrp_mat_  : Pointer to a initialized CSRP_mat structure
void CSRP_init_with_COO_mat(
    const int nrow, const int ncol, const int nnz, const int *row,
    const int *col, const double *val, CSRP_mat_t *csrp_mat_
);

// Free a CSRP_mat structure
// Input parameter:
//   csrp_mat : Pointer to a CSRP_mat structure
void CSRP_free(CSRP_mat_t csrp_mat);

// Partition a CSR matrix into multiple blocks with the same nnz
// for multiple threads execution of SpMV
// Input parameters:
//   csrp_mat : Pointer to a CSRP_mat structure
//   nblk     : Number of non-zero blocks
//   nthread  : Number of threads to be used in SpMV later
// Output parameter:
//   csrp_mat : Pointer to a CSRP_mat structure with partitioning information
void CSRP_partition_multithread(CSRP_mat_t csrp_mat, const int nblk, const int nthread);

// Use first-touch policy to optimize the storage of CSR arrays in a CSRP_mat structure
// Input:
//   csrp_mat : Pointer to a CSRP_mat structure
// Output:
//   csrp_mat : Pointer to a CSRP_mat structure with NUMA optimized storage
void CSRP_optimize_NUMA(CSRP_mat_t csrp_mat);

// Perform OpenMP parallelized CSR SpMV with a CSRP_mat structure
// Input parameters:
//   csrp_mat : Pointer to an initialized and partitioned CSRP_mat structure
//   x        : Input vector
// Output parameter:
//   y : Output vector, will be overwritten by csrp_mat * x 
void CSRP_SpMV(CSRP_mat_t csrp_mat, const double *x, double *y);

#ifdef __cplusplus
}
#endif

#endif

