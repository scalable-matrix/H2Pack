// @brief    : Implementations of some helper functions I use here and there
// @author   : Hua Huang <huangh223@gatech.edu>
// @modified : 2020-12-03

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include <sys/time.h>
#include <math.h>

#include "utils.h"

// Get wall-clock time in seconds
double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

// Partition an array into multiple same-size blocks and return the 
// start position of a given block
void calc_block_spos_len(
    const int len, const int nblk, const int iblk,
    int *blk_spos, int *blk_len
)
{
	if (iblk < 0 || iblk > nblk)
    {
        *blk_spos = -1;
        *blk_len  = 0;
        return;
    }
	int rem = len % nblk;
	int bs0 = len / nblk;
	int bs1 = bs0 + 1;
	if (iblk < rem) 
    {
        *blk_spos = bs1 * iblk;
        *blk_len  = bs1;
    } else {
        *blk_spos = bs0 * iblk + rem;
        *blk_len  = bs0;
    }
}

// Allocate a piece of aligned memory 
void *malloc_aligned(size_t size, size_t alignment)
{
    void *ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

// Free a piece of aligned memory allocated by malloc_aligned()
void free_aligned(void *mem)
{
    free(mem);
}

// Calculate the 2-norm of a vector
// Warning: this is a naive implementation, not numerically stable
double calc_2norm(const int len, const double *x)
{
    double res = 0.0;
    for (int i = 0; i < len; i++) res += x[i] * x[i];
    return sqrt(res);
}

// Calculate the 2-norm of the difference between two vectors 
// and the 2-norm of the reference vector 
void calc_err_2norm(
    const int len, const double *x0, const double *x1, 
    double *x0_2norm_, double *err_2norm_
)
{
    double x0_2norm = 0.0, err_2norm = 0.0, diff;
    for (int i = 0; i < len; i++)
    {
        diff = x0[i] - x1[i];
        x0_2norm  += x0[i] * x0[i];
        err_2norm += diff  * diff;
    }
    *x0_2norm_  = sqrt(x0_2norm);
    *err_2norm_ = sqrt(err_2norm);
}

// Copy a row-major matrix block to another row-major matrix
void copy_matrix_block(
    const size_t dt_size, const int nrow, const int ncol, 
    const void *src, const int lds, void *dst, const int ldd
)
{
    const char *src_ = (char*) src;
    char *dst_ = (char*) dst;
    const size_t lds_ = dt_size * (size_t) lds;
    const size_t ldd_ = dt_size * (size_t) ldd;
    const size_t row_msize = dt_size * (size_t) ncol;
    for (int irow = 0; irow < nrow; irow++)
    {
        size_t src_offset = (size_t) irow * lds_;
        size_t dst_offset = (size_t) irow * ldd_;
        memcpy(dst_ + dst_offset, src_ + src_offset, row_msize);
    }
}

// Gather elements from a vector to another vector
void gather_vector_elements(const size_t dt_size, const int nelem, const int *idx, const void *src, void *dst)
{
    if (dt_size == 4)
    {
        const float *src_ = (float*) src;
        float *dst_ = (float*) dst;
        #if defined(_OPENMP)
        #pragma omp simd
        #endif
        for (int i = 0; i < nelem; i++) dst_[i] = src_[idx[i]];
    }
    if (dt_size == 8)
    {
        const double *src_ = (double*) src;
        double *dst_ = (double*) dst;
        #if defined(_OPENMP)
        #pragma omp simd
        #endif
        for (int i = 0; i < nelem; i++) dst_[i] = src_[idx[i]];
    }
    if (dt_size == 16)
    {
        const double _Complex *src_ = (double _Complex*) src;
        double _Complex *dst_ = (double _Complex*) dst;
        #if defined(_OPENMP)
        #pragma omp simd
        #endif
        for (int i = 0; i < nelem; i++) dst_[i] = src_[idx[i]];
    }
}

// Gather rows from a matrix to another matrix
void gather_matrix_rows(
    const size_t dt_size, const int nrow, const int ncol, const int *idx, 
    const void *src, const int lds, void *dst, const int ldd
)
{
    const char *src_ = (char*) src;
    char *dst_ = (char*) dst;
    const size_t lds_ = dt_size * (size_t) lds;
    const size_t ldd_ = dt_size * (size_t) ldd;
    const size_t row_msize = dt_size * (size_t) ncol;
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int irow = 0; irow < nrow; irow++)
    {
        size_t src_offset = (size_t) idx[irow] * lds_;
        size_t dst_offset = (size_t) irow * ldd_;
        memcpy(dst_ + dst_offset, src_ + src_offset, row_msize);
    }
}

// Gather columns from a matrix to another matrix
void gather_matrix_cols(
    const size_t dt_size, const int nrow, const int ncol, const int *idx, 
    const void *src, const int lds, void *dst, const int ldd
)
{
    const char *src_ = (char*) src;
    char *dst_ = (char*) dst;
    const size_t lds_ = dt_size * (size_t) lds;
    const size_t ldd_ = dt_size * (size_t) ldd;
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int irow = 0; irow < nrow; irow++)
    {
        size_t src_offset = (size_t) irow * lds_;
        size_t dst_offset = (size_t) irow * ldd_;
        gather_vector_elements(dt_size, ncol, idx, src_ + src_offset, dst_ + dst_offset);
    }
}

// Print a row-major int matrix block to standard output
void print_int_mat_blk(
    const int *mat, const int ldm, const int nrow, const int ncol, 
    const char *fmt, const char *mat_name
)
{
    printf("%s:\n", mat_name);
    for (int i = 0; i < nrow; i++)
    {
        const int *mat_i = mat + i * ldm;
        for (int j = 0; j < ncol; j++)
        {
            printf(fmt, mat_i[j]);
            printf("  ");
        }
        printf("\n");
    }
    printf("\n");
}

// Print a row-major double matrix block to standard output
void print_dbl_mat_blk(
    const double *mat, const int ldm, const int nrow, const int ncol, 
    const char *fmt, const char *mat_name
)
{
    printf("%s:\n", mat_name);
    for (int i = 0; i < nrow; i++)
    {
        const double *mat_i = mat + i * ldm;
        for (int j = 0; j < ncol; j++)
        {
            printf(fmt, mat_i[j]);
            printf("  ");
        }
        printf("\n");
    }
    printf("\n");
}

