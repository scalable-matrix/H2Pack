// @brief    : Implementations of some helper functions I use here and there
// @author   : Hua Huang <huangh223@gatech.edu>
// @modified : 2020-02-10

#ifndef __HUANGH223_UTILS_H__
#define __HUANGH223_UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(a, b)  ((a) < (b) ? (a) : (b))
#define MAX(a, b)  ((a) > (b) ? (a) : (b))

#define DBG_PRINTF(fmt, args...)                   \
    do                                             \
    {                                              \
        fprintf(stdout, "%s, line %d: "fmt,        \
                __FILE__, __LINE__, ##args);       \
        fflush(stdout);                            \
    } while (0)

// Get wall-clock time in seconds
// Output parameter:
//   <return> : Wall-clock time in second
double get_wtime_sec();

// Partition an array into multiple same-size blocks and return the 
// start position of a given block
// Input parameters:
//   len  : Length of the array
//   nblk : Total number of blocks to be partitioned
//   iblk : Index of the block whose start position we need.
//          0 <= iblk <= nblk, iblk == 0/nblk return 0/len.
// Output parameter:
//   *blk_spos : The start position of the iblk-th block
//               -1 means invalid parameters
//   *blk_len  : The length of the iblk-th block
void calc_block_spos_len(
    const int len, const int nblk, const int iblk,
    int *blk_spos, int *blk_len
);

// Allocate a piece of aligned memory 
// Input parameters:
//   size      : Size of the memory to be allocated, in bytes
//   alignment : Size of the alignment, in bytes, should be a power of 8
// Output parameter:
//   <return>  : Pointer to the allocated aligned memory
void *malloc_aligned(size_t size, size_t alignment);

// Free a piece of aligned memory allocated by malloc_aligned()
// Input parameter:
//   mem : Pointer to the memory to be free
void free_aligned(void *mem);

// Calculate the 2-norm of a vector
// Input parameters:
//   len : Length of the vector
//   x   : Size >= len, vector
// Output parameter:
//   <return> : 2-norm of vector x
double calc_2norm(const int len, const double *x);

// Calculate the 2-norm of the difference between two vectors 
// and the 2-norm of the reference vector 
// Input parameters:
//   len : Length of the vector
//   x0  : Size >= len, reference vector
//   x1  : Size >= len, target vector to be compared 
// Output parameters:
//   *x0_2norm_  : 2-norm of vector x0
//   *err_2norm_ : 2-norm of vector (x0 - x1)
void calc_err_2norm(
    const int len, const double *x0, const double *x1, 
    double *x0_2norm_, double *err_2norm_
);

// Copy a row-major int matrix block to another row-major int matrix
// Input parameters:
//   src  : Size >= lds * nrow, source matrix
//   lds  : Leading dimension of src, >= ncol
//   nrow : Number of rows to be copied
//   ncol : Number of columns to be copied
//   ldd  : Leading dimension of dst, >= ncol
// Output parameter:
//   dst  : Size >= ldd * nrow, destination matrix
void copy_int_mat_blk(
    const int *src, const int lds, const int nrow, const int ncol, 
    int *dst, const int ldd
);

// Copy a row-major double matrix block to another row-major double matrix
// Input / output parameters are the same as copy_int_mat_blk()
void copy_dbl_mat_blk(
    const double *src, const int lds, const int nrow, const int ncol,  
    double *dst, const int ldd
);

// Print a row-major int matrix block to standard output
// Input parameters:
//   mat      : Size >= ldm * nrow, matrix to be printed 
//   ldm      : Leading dimension of mat, >= ncol
//   nrow     : Number of rows to be printed
//   ncol     : Number of columns to be printed
//   fmt      : Output format string
//   mat_name : Name of the matrix, to be printed
void print_int_mat_blk(
    const int *mat, const int ldm, const int nrow, const int ncol, 
    const char *fmt, const char *mat_name
);

// Print a row-major double matrix block to standard output
// Input / output parameters are the same as copy_int_mat_blk()
void print_dbl_mat_blk(
    const double *mat, const int ldm, const int nrow, const int ncol, 
    const char *fmt, const char *mat_name
);

#ifdef __cplusplus
}
#endif

#endif

