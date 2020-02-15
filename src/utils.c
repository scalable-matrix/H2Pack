// @brief    : Implementations of some helper functions I use here and there
// @author   : Hua Huang <huangh223@gatech.edu>
// @modified : 2020-02-10

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

// For _mm_malloc and _mm_free
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#endif
#if defined(__GNUC__) || (__clang__)
#include <mm_malloc.h>
#endif

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
    return _mm_malloc(size, alignment);
}

// Free a piece of aligned memory allocated by malloc_aligned()
void free_aligned(void *mem)
{
    _mm_free(mem);
}

// Calculate the 2-norm of a vector
double calc_2norm(const int len, const double *x)
{
    double res = 0.0;
    for (int i = 0; i < len; i++)
        res += x[i] * x[i];
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

// Copy a row-major int matrix block to another row-major int matrix
void copy_int_mat_blk(
    const int *src, const int lds, const int nrow, const int ncol, 
    int *dst, const int ldd
)
{
    for (int irow = 0; irow < nrow; irow++)
        memcpy(dst + irow * ldd, src + irow * lds, INT_MSIZE * ncol);
}

// Copy a row-major double matrix block to another row-major double matrix
void copy_dbl_mat_blk(
    const double *src, const int lds, const int nrow, const int ncol,  
    double *dst, const int ldd
)
{
    for (int irow = 0; irow < nrow; irow++)
        memcpy(dst + irow * ldd, src + irow * lds, DBL_MSIZE * ncol);
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

