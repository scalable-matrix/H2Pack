#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "H2Pack_dense_mat.h"

// Partial pivoting QR decomposition, simplified output version
void H2P_partial_pivot_QR(
    H2P_dense_mat_t A, const int tol_rank, const DTYPE tol_norm, 
    int **p_, int *r_
)
{
    DTYPE *R = A->data;
    const int nrow = A->nrow;
    const int ncol = A->ncol;
    const int ldR  = A->ld;
    const int max_iter = MIN(nrow, ncol);
    
    DTYPE *col_norm = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * ncol);
    DTYPE *col_tmp  = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * nrow);
    DTYPE *h_vec    = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * nrow);
    DTYPE *h_R_mv   = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * ncol);
    int *p = (int*) malloc(sizeof(int) * ncol);
    assert(col_norm != NULL && col_tmp != NULL);
    assert(h_vec != NULL && h_R_mv != NULL && p != NULL);
    for (int icol = 0; icol < ncol; icol++)
    {
        p[icol] = icol;
        col_norm[icol] = BLAS_NRM2(nrow, R + icol * ldR, 1);
    }
    
    int stop_rank = MIN(max_iter, tol_rank);
    DTYPE norm_eps = DSQRT((DTYPE) nrow) * 1e-15;
    DTYPE stop_norm = MAX(norm_eps, tol_norm);
    
    int r = -1;
    size_t col_msize = sizeof(DTYPE) * nrow;
    // Main iteration of Household QR
    for (int iter = 0; iter < max_iter; iter++)
    {
        // 1. Find a column with largest 2-norm
        DTYPE norm_p = col_norm[iter];
        int   pivot  = iter;
        for (int j = iter + 1; j < ncol; j++)
        {
            if (col_norm[j] > norm_p)
            {
                norm_p = col_norm[j];
                pivot  = j;
            }
        }
        
        // 2. Check the stop criteria
        if ((norm_p < stop_norm) || (iter > stop_rank))
        {
            r = iter;
            break;
        }
        
        // 3. Swap the column
        if (iter != pivot)
        {
            int tmp_p = p[iter]; p[iter] = p[pivot]; p[pivot] = tmp_p;
            DTYPE tmp_norm  = col_norm[iter];
            col_norm[iter]  = col_norm[pivot];
            col_norm[pivot] = tmp_norm;
            memcpy(col_tmp,         R + iter  * ldR, col_msize);
            memcpy(R + iter  * ldR, R + pivot * ldR, col_msize);
            memcpy(R + pivot * ldR, col_tmp,         col_msize);
        }
        
        // 4. Householder orthogonalization
        int h_len = nrow - iter;
        memcpy(h_vec, R + iter * ldR + iter, sizeof(DTYPE) * h_len);
        DTYPE sign = (h_vec[0] > 0.0) ? 1.0 : -1.0;
        DTYPE h_norm = BLAS_NRM2(h_len, h_vec, 1);
        h_vec[0] = h_vec[0] + sign * h_norm;
        DTYPE inv_h_norm = 1.0 / BLAS_NRM2(h_len, h_vec, 1);
        BLAS_SCAL(h_len, inv_h_norm, h_vec, 1);
        
        // 5. Eliminate the iter-th sub-column and orthogonalize columns 
        //    right to the iter-th column
        R[iter * ldR + iter] = -sign * h_norm;
        memset(R + iter * ldR + iter + 1, 0, sizeof(DTYPE) * (h_len - 1));
        DTYPE *R_block = R + (iter + 1) * ldR + iter;
        int R_block_nrow = h_len;
        int R_block_ncol = ncol - iter - 1;
        BLAS_GEMV(
            CblasColMajor, CblasTrans, R_block_nrow, R_block_ncol,
            1.0, R_block, ldR, h_vec, 1, 0.0, h_R_mv, 1
        );
        BLAS_GER(
            CblasColMajor, R_block_nrow, R_block_ncol, -2.0,
            h_vec, 1, h_R_mv, 1, R_block, ldR
        );
        
        // 6. Update 2-norm of columns right to the iter-th column
        int h_len_m1 = h_len - 1;
        for (int j = iter + 1; j < ncol; j++)
        {
            // Skip small columns
            if (col_norm[j] < stop_norm)
            {
                col_norm[j] = 0.0;
                continue;
            }
            
            // Update column norm: fast update when the new norm is not so small,
            // recalculate when the new norm is small for stability
            DTYPE tmp = R[j * ldR + iter];
            tmp = tmp * tmp;
            tmp = col_norm[j] * col_norm[j] - tmp;
            if (tmp <= 1e-10)  // TODO: Find a better threshold?
            {
                DTYPE *R_col_j = R + j * ldR + (iter + 1);
                col_norm[j] = BLAS_NRM2(h_len_m1, R_col_j, 1);
            } else {
                col_norm[j] = DSQRT(tmp);
            }
        }
    }
    if (r == -1) r = max_iter;
    
    *r_ = r;
    *p_ = p;
    H2P_free_aligned(h_vec);
    H2P_free_aligned(h_R_mv);
    H2P_free_aligned(col_tmp);
    H2P_free_aligned(col_norm);
}
