#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#ifdef USE_MKL
#include <mkl.h>
#define BLAS_SET_NUM_THREADS mkl_set_num_threads
#endif

#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "H2Pack_aux_structs.h"

static inline DTYPE CBLAS_NRM2(const int n, const DTYPE *x)
{
    DTYPE res = 0.0;
    #pragma omp simd
    for (int i = 0; i < n; i++) 
        res += x[i] * x[i];
    return DSQRT(res);
}

static inline DTYPE CBLAS_DOT(const int n, const DTYPE *x, const DTYPE *y)
{
    DTYPE res = 0.0;
    #pragma omp simd
    for (int i = 0; i < n; i++) 
        res += x[i] * y[i];
    return res;
}

// Partial pivoting QR decomposition, simplified output version
// The partial pivoting QR decomposition is of form:
//     A * P = Q * [R11, R12; 0, R22]
// where R11 is an upper-triangular matrix, R12 and R22 are dense matrices,
// P is a permutation matrix. 
// Input parameters:
//   A        : Target matrix, stored in column major
//   tol_rank : QR stopping parameter, maximum column rank, 
//   tol_norm : QR stopping parameter, maximum column 2-norm
//   rel_norm : If tol_norm is relative to the largest column 2-norm in A
//   nthreads : Number of threads used in this function
//   QR_buff  : Size 2 * A->ncol, working buffer for partial pivoting QR
// Output parameters:
//   A : Matrix R: [R11, R12; 0, R22]
//   p : Matrix A column permutation array, A(:, p) = A * P
//   r : Dimension of upper-triangular matrix R11
void H2P_partial_pivot_QR(
    H2P_dense_mat_t A, const int tol_rank, const DTYPE tol_norm, 
    const int rel_norm, int *p, int *r, const int nthreads, DTYPE *QR_buff
)
{
    DTYPE *R = A->data;
    int nrow = A->nrow;
    int ncol = A->ncol;
    int ldR  = A->ld;
    int max_iter = MIN(nrow, ncol);
    
    BLAS_SET_NUM_THREADS(nthreads);
    
    DTYPE *col_norm = QR_buff + 0 * ncol; 
    DTYPE *h_R_mv   = QR_buff + 1 * ncol;
    
    // 1. Find a column with largest 2-norm
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int j = 0; j < ncol; j++)
    {
        p[j] = j;
        col_norm[j] = CBLAS_NRM2(nrow, R + j * ldR);
    }
    DTYPE norm_p = 0.0;
    int pivot = 0;
    for (int j = 0; j < ncol; j++)
    {
        if (col_norm[j] > norm_p)
        {
            norm_p = col_norm[j];
            pivot = j;
        }
    }
    
    int stop_rank = MIN(max_iter, tol_rank);
    DTYPE norm_eps = DSQRT((DTYPE) nrow) * 1e-15;
    DTYPE stop_norm = MAX(norm_eps, tol_norm);
    if (rel_norm) stop_norm *= norm_p;
    
    int rank = -1;
    // Main iteration of Household QR
    for (int i = 0; i < max_iter; i++)
    {   
        // 1. Check the stop criteria
        if ((norm_p < stop_norm) || (i >= stop_rank))
        {
            rank = i;
            break;
        }
        
        // 2. Swap the column
        if (i != pivot)
        {
            int tmp_p = p[i]; p[i] = p[pivot]; p[pivot] = tmp_p;
            DTYPE tmp_norm  = col_norm[i];
            col_norm[i]  = col_norm[pivot];
            col_norm[pivot] = tmp_norm;
            DTYPE *R_i     = R + i     * ldR;
            DTYPE *R_pivot = R + pivot * ldR;
            for (int j = 0; j < nrow; j++)
            {
                DTYPE tmp_R = R_i[j];
                R_i[j]      = R_pivot[j];
                R_pivot[j]  = tmp_R;
            }
        }
        
        // 3. Calculate Householder vector
        int h_len    = nrow - i;
        int h_len_m1 = h_len - 1;
        DTYPE *h_vec = R + i * ldR + i;
        DTYPE sign   = (h_vec[0] > 0.0) ? 1.0 : -1.0;
        DTYPE h_norm = CBLAS_NRM2(h_len, h_vec);
        h_vec[0] = h_vec[0] + sign * h_norm;
        DTYPE inv_h_norm = 1.0 / CBLAS_NRM2(h_len, h_vec);
        #pragma omp simd
        for (int j = 0; j < h_len; j++) h_vec[j] *= inv_h_norm;
        
        // 4. & 5. Householder update & column norm update
        DTYPE *R_block = R + (i + 1) * ldR + i;
        int R_block_nrow = h_len;
        int R_block_ncol = ncol - i - 1;
        #pragma omp parallel for num_threads(nthreads) schedule(guided)
        for (int j = 0; j < R_block_ncol; j++)
        {
            int ji1 = j + i + 1;
            
            DTYPE *R_block_j = R_block + j * ldR;
            DTYPE h_Rj = 2.0 * CBLAS_DOT(R_block_nrow, h_vec, R_block_j);
            
            // 4. Orthogonalize columns right to the i-th column
            #pragma omp simd
            for (int k = 0; k < R_block_nrow; k++)
                R_block_j[k] -= h_Rj * h_vec[k];
            
            // 5. Update i-th column's 2-norm
            if (col_norm[ji1] < stop_norm)
            {
                col_norm[ji1] = 0.0;
                continue;
            }
            DTYPE tmp = R_block_j[0] * R_block_j[0];
            tmp = col_norm[ji1] * col_norm[ji1] - tmp;
            if (tmp <= 1e-10)
            {
                col_norm[ji1] = CBLAS_NRM2(h_len_m1, R_block_j + 1);
            } else {
                // Fast update 2-norm when the new column norm is not so small
                col_norm[ji1] = DSQRT(tmp);
            }
        }
        
        // We don't need h_vec anymore, can overwrite the i-th column of R
        h_vec[0] = -sign * h_norm;
        memset(h_vec + 1, 0, sizeof(DTYPE) * (h_len - 1));
        // Find next pivot 
        pivot  = i + 1;
        norm_p = 0.0;
        for (int j = i + 1; j < ncol; j++)
        {
            if (col_norm[j] > norm_p)
            {
                norm_p = col_norm[j];
                pivot  = j;
            }
        }
    }
    if (rank == -1) rank = max_iter;
    
    *r = rank;
}

// Partial pivoting QR for ID, may need to be upgraded to Strong Rank 
// Revealing QR later.
// Input parameters:
//   A          : Target matrix, stored in column major
//   stop_type  : Partial QR stop criteria: QR_RANK, QR_REL_NRM, or QR_ABS_NRM
//   stop_param : Pointer to partial QR stop parameter
//   nthreads   : Number of threads used in this function
//   QR_buff    : Size 2 * A->ncol, working buffer for partial pivoting QR
// Output parameters:
//   A : Matrix R: [R11, R12]
//   p : Matrix A column permutation array, A(:, p) = A * P
void H2P_ID_QR(
    H2P_dense_mat_t A, const int stop_type, void *stop_param, int *p, 
    const int nthreads, DTYPE *QR_buff
)
{
    // Parse partial QR stop criteria and perform partial QR
    int r, tol_rank, rel_norm;
    DTYPE tol_norm;
    if (stop_type == QR_RANK)
    {
        int *param = (int*) stop_param;
        tol_rank = param[0];
        tol_norm = 1e-15;
        rel_norm = 0;
    }
    if (stop_type == QR_REL_NRM)
    {
        DTYPE *param = (DTYPE*) stop_param;
        tol_rank = MIN(A->nrow, A->ncol);
        tol_norm = param[0];
        rel_norm = 1;
    }
    if (stop_type == QR_ABS_NRM)
    {
        DTYPE *param = (DTYPE*) stop_param;
        tol_rank = MIN(A->nrow, A->ncol);
        tol_norm = param[0];
        rel_norm = 0;
    }
    H2P_partial_pivot_QR(A, tol_rank, tol_norm, rel_norm, p, &r, nthreads, QR_buff);
    
    // Special case: each column's 2-norm is smaller than the threshold
    if (r == 0)
    {
        for (int i = 0; i < A->ncol; i++) p[i] = i;
        A->nrow = 0;
        return;
    }
    
    // Truncate R to be [R11, R12] and return
    A->nrow = r;
}


// Quick sort two array in ascending order 
// Input parameters:
//  key  : Key array
//  val  : Value array
//  l, r : Sort range [l : r]
// Output parameters:
//  key  : Sorted key array
//  val  : Sorted value array
static void H2P_qsort_key_value(int *key, int *val, const int l, const int r)
{
    int i = l, j = r, tmp;
    int mid = key[(i + j) / 2];
    while (i <= j)
    {
        while (key[i] < mid) i++;
        while (key[j] > mid) j--;
        if (i <= j)
        {
            tmp = key[i]; key[i] = key[j]; key[j] = tmp;
            tmp = val[i]; val[i] = val[j]; val[j] = tmp;
            i++; j--;
        }
    }
    if (i < r) H2P_qsort_key_value(key, val, i, r);
    if (j > l) H2P_qsort_key_value(key, val, l, j);
}

// Interpolative Decomposition (ID) using partial QR over rows of a target 
// matrix. Partial pivoting QR may need to be upgraded to SRRQR later. 
void H2P_ID_compress(
    H2P_dense_mat_t A, const int stop_type, void *stop_param, H2P_dense_mat_t *U_, 
    H2P_int_vec_t J, const int nthreads, DTYPE *QR_buff, int *ID_buff
)
{
    // 1. Partial pivoting QR for A^T
    // Note: A is stored in row major style but H2P_ID_QR needs A stored in column
    // major style. We manipulate the size information of A to save some time
    const int nrow = A->nrow;
    const int ncol = A->ncol;
    A->nrow = ncol;
    A->ncol = nrow;
    H2P_int_vec_set_capacity(J, nrow);
    H2P_ID_QR(A, stop_type, stop_param, J->data, nthreads, QR_buff);
    H2P_dense_mat_t R = A; // Note: the output R stored in A is still stored in column major style
    int r = A->nrow;  // Obtained rank
    J->length = r;
    
    // 2. Set permutation indices p0, sort the index subset J[0:r-1]
    int *p0 = ID_buff + 0 * nrow;
    int *p1 = ID_buff + 1 * nrow;
    int *i0 = ID_buff + 2 * nrow;
    int *i1 = ID_buff + 3 * nrow;
    for (int i = 0; i < nrow; i++) 
    {
        p0[J->data[i]] = i;
        i0[i] = i;
    }
    H2P_qsort_key_value(J->data, i0, 0, r - 1);
    for (int i = 0; i < nrow; i++) i1[i0[i]] = i;
    for (int i = 0; i < nrow; i++) p1[i] = i1[p0[i]];
    
    // 3. Form the output U
    H2P_dense_mat_t U;
    if (r == 0)
    {
        // Special case: rank = 0, set U and J as empty
        U->nrow = nrow;
        U->ncol = 0;
        U->ld   = 0;
        U->data = NULL;
    } else {
        if (U_ != NULL)
        {
            // (1) Before permutation, the upper part of U is a diagonal
            H2P_dense_mat_init(&U, nrow, r);
            for (int i = 0; i < r; i++)
            {
                memset(U->data + i * r, 0, sizeof(DTYPE) * r);
                U->data[i * r + i] = 1.0;
            }
            DTYPE *R11 = R->data;
            DTYPE *R12 = R->data + r * R->ld;
            int nrow_R12 = r;
            int ncol_R12 = nrow - r;
            // (2) Solve E = inv(R11) * R12, stored in R12 in column major style
            //     --> equals to what we need: E^T stored in row major style
            BLAS_SET_NUM_THREADS(nthreads);
            CBLAS_TRSM(
                CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                nrow_R12, ncol_R12, 1.0, R11, R->ld, R12, R->ld
            );
            // (3) Reorder E^T's columns according to the sorted J
            DTYPE *UL = U->data + r * r;
            for (int icol = 0; icol < r; icol++)
            {
                DTYPE *R12_icol = R12 + i0[icol];
                DTYPE *UL_icol = UL + icol;
                for (int irow = 0; irow < ncol_R12; irow++)
                    UL_icol[irow * r] = R12_icol[irow * R->ld];
            }
            // (4) Permute U's rows 
            H2P_dense_mat_permute_rows(U, p1);
        }
    }
    
    if (U_ != NULL) *U_ = U;
}
