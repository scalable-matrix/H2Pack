#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>

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

static inline void swap_int(int *x, int *y, int len)
{
    int tmp;
    for (int i = 0; i < len; i++)
    {
        tmp  = x[i];
        x[i] = y[i];
        y[i] = tmp;
    }
}

static inline void swap_DTYPE(DTYPE *x, DTYPE *y, int len)
{
    DTYPE tmp;
    for (int i = 0; i < len; i++)
    {
        tmp  = x[i];
        x[i] = y[i];
        y[i] = tmp;
    }
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
//   QR_buff  : Size A->ncol, working buffer for partial pivoting QR
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
    
    DTYPE *col_norm = QR_buff; 
    
    // Find a column with largest 2-norm
    #pragma omp parallel for if (nthreads > 1) \
    num_threads(nthreads) schedule(static)
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
    
    // Scale the stopping norm
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
            swap_int(p + i, p + pivot, 1);
            swap_DTYPE(col_norm + i, col_norm + pivot, 1);
            swap_DTYPE(R + i * ldR, R + pivot * ldR, nrow);
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
        #pragma omp parallel for if (nthreads > 1) \
        num_threads(nthreads) schedule(guided)
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

// H2P_partial_pivot_QR operated on column blocks for tensor kernel matrix.
// Each column block has kdim columns. In each step, we swap kdim columns 
// and do kdim Householder orthogonalization.
// Size of QR_buff: (2*kdim+2)*A->nrow + (kdim+1)*A->ncol
// BLAS-3 approach, ref: https://doi.org/10.1145/1513895.1513904
void H2P_partial_pivot_QR_kdim(
    H2P_dense_mat_t A, const int kdim, const int tol_rank, const DTYPE tol_norm, 
    const int rel_norm, int *p, int *r, const int nthreads, DTYPE *QR_buff
)
{
    DTYPE *R = A->data;
    int nrow = A->nrow;
    int ncol = A->ncol;
    int nblk = ncol / kdim;
    int ldR  = A->ld;
    int max_iter = MIN(nrow, ncol) / kdim;
    
    BLAS_SET_NUM_THREADS(nthreads);
    
    for (int j = 0; j < ncol; j++) p[j] = j;
    
    // Partition the work buffer 
    DTYPE *blk_norm = QR_buff;
    DTYPE *Vblk = blk_norm + ncol;
    DTYPE *Wblk = Vblk + kdim * nrow;
    DTYPE *VV   = Wblk + kdim * nrow;
    DTYPE *WVV  = VV   + nrow;
    DTYPE *WR   = WVV  + nrow;
    
    // Find a column with largest 2-norm as a scaling factor
    #pragma omp parallel for if(nthreads > 1) \
    num_threads(nthreads) schedule(static)
    for (int j = 0; j < ncol; j++)
        blk_norm[j] = CBLAS_NRM2(nrow, R + j * ldR);
    
    // Find a column block with largest 2-norm as the first pivot
    DTYPE norm_p = 0.0;
    int pivot = 0;
    for (int j = 0; j < nblk; j++)
    {
        DTYPE tmp = 0.0;
        for (int k = 0; k < kdim; k++) 
        {
            int idx = kdim * j + k;
            tmp += blk_norm[idx] * blk_norm[idx]; 
        }
        blk_norm[j] = DSQRT(tmp);
        if (blk_norm[j] > norm_p)
        {
            norm_p = blk_norm[j];
            pivot  = j;
        }
    }
    
    // Scale the stopping norm
    int stop_rank   = MIN(max_iter, tol_rank);
    DTYPE norm_eps  = DSQRT((DTYPE) nrow) * 1e-15;
    DTYPE stop_norm = MAX(norm_eps, tol_norm);
    if (rel_norm) stop_norm *= norm_p;
    
    // Main iteration of Household QR
    int rank = -1;
    for (int i = 0; i < max_iter; i++)
    {   
        // 1. Check the stop criteria
        if ((norm_p < stop_norm) || (i >= stop_rank))
        {
            rank = i * kdim;
            break;
        }
        
        // 2. Swap the column
        if (i != pivot)
        {
            swap_int(p + i * kdim, p + pivot * kdim, kdim);
            swap_DTYPE(blk_norm + i, blk_norm + pivot, 1);
            DTYPE *R_i = R + i * kdim * ldR;
            DTYPE *R_pivot = R + pivot * kdim * ldR;
            swap_DTYPE(R_i, R_pivot, ldR * kdim);
        }
        
        int VWblk_nrow = nrow - i * kdim;
        
        // 3. Do kdim times of consecutive Householder orthogonalize on the current column block
        for (int ii = i * kdim; ii < i * kdim + kdim; ii++)
        {
            // 3.1 Calculate Householder vector
            int h_len    = nrow - ii;
            DTYPE *h_vec = R + ii * ldR + ii;
            DTYPE sign   = (h_vec[0] > 0.0) ? 1.0 : -1.0;
            DTYPE h_norm = CBLAS_NRM2(h_len, h_vec);
            h_vec[0] = h_vec[0] + sign * h_norm;
            DTYPE inv_h_norm = 1.0 / CBLAS_NRM2(h_len, h_vec);
            #pragma omp simd
            for (int j = 0; j < h_len; j++) h_vec[j] *= inv_h_norm;
            
            // 3.2 Householder update current column block
            int blk_i = ii - i * kdim;
            DTYPE *R_block = R + (ii + 1) * ldR + ii;
            int R_block_nrow = h_len;
            int R_block_ncol = ncol - ii - 1;
            for (int j = 0; j < kdim - blk_i - 1; j++)
            {
                int ji1 = j + ii + 1;
                
                DTYPE *R_block_j = R_block + j * ldR;
                DTYPE h_Rj = 2.0 * CBLAS_DOT(R_block_nrow, h_vec, R_block_j);
                
                // Orthogonalize columns right to the ii-th column
                #pragma omp simd
                for (int k = 0; k < R_block_nrow; k++)
                    R_block_j[k] -= h_Rj * h_vec[k];
            }
            
            // Save the Householder vector for block update
            for (int j = 0; j < blk_i; j++) Vblk[blk_i * VWblk_nrow + j] = 0.0;
            memcpy(Vblk + blk_i * VWblk_nrow + blk_i, h_vec, sizeof(DTYPE) * h_len);
            
            // We don't need h_vec anymore, can overwrite the i-th column of R
            h_vec[0] = -sign * h_norm;
            memset(h_vec + 1, 0, sizeof(DTYPE) * (h_len - 1));
        }
        
        // 4. Construct W, use V & W for block Householder update 
        #pragma omp simd
        for (int j = 0; j < VWblk_nrow; j++) Wblk[j] = -2.0 * Vblk[j];
        for (int ii = 1; ii < kdim; ii++)
        {
            DTYPE *Vii = Vblk + ii * VWblk_nrow;
            DTYPE *Wii = Wblk + ii * VWblk_nrow;
            CBLAS_GEMV(
                CblasColMajor, CblasTrans, VWblk_nrow, ii, 
                1.0, Vblk, VWblk_nrow, Vii, 1, 0.0, VV, 1
            );
            CBLAS_GEMV(
                CblasColMajor, CblasNoTrans, VWblk_nrow, ii,
                1.0, Wblk, VWblk_nrow, VV, 1, 0.0, WVV, 1    
            );
            #pragma omp simd
            for (int j = 0; j < VWblk_nrow; j++)
                Wii[j] = -2.0 * (Vii[j] + WVV[j]);
        }
        int R_blk_scol = i * kdim + kdim;
        int R_blk_ncol = ncol - R_blk_scol;
        DTYPE *R_blk = R + R_blk_scol * ldR + (i * kdim);
        int R_col_blk_128KB = 128 * 1024;
        R_col_blk_128KB /= (int) sizeof(DTYPE);
        R_col_blk_128KB /= (VWblk_nrow * kdim * 3);
        if (R_col_blk_128KB < 2) R_col_blk_128KB = 8;
        for (int R_col_offset = 0; R_col_offset < R_blk_ncol; R_col_offset += R_col_blk_128KB)
        {
            int R_col_blksize = R_col_blk_128KB;
            if (R_col_blksize + R_col_offset > R_blk_ncol) R_col_blksize = R_blk_ncol - R_col_offset;
            CBLAS_GEMM(
                CblasColMajor, CblasTrans, CblasNoTrans,
                kdim, R_col_blksize, VWblk_nrow, 
                1.0, Wblk, VWblk_nrow, R_blk + ldR * R_col_offset, ldR, 
                0.0, WR, kdim
            );
            CBLAS_GEMM(
                CblasColMajor, CblasNoTrans, CblasNoTrans, 
                VWblk_nrow, R_col_blksize, kdim, 
                1.0, Vblk, VWblk_nrow, WR, kdim, 
                1.0, R_blk + ldR * R_col_offset, ldR
            );
        }
        
        // 5. Update i-th column's 2-norm
        #pragma omp parallel for if(nthreads > 1) \
        num_threads(nthreads) schedule(guided)
        for (int j = i + 1; j < nblk; j++)
        {
            if (blk_norm[j] < stop_norm)
            {
                blk_norm[j] = 0.0;
                continue;
            }
            // R(i * kdim, j * kdim + k)
            DTYPE *R_block = R + i * kdim + j * kdim * ldR;
            DTYPE tmp = blk_norm[j] * blk_norm[j];
            for (int k0 = 0; k0 < kdim; k0++)
            {
                for (int k1 = 0; k1 < kdim; k1++)
                {
                    int idx = k0 * ldR + k1;
                    tmp -= R_block[idx] * R_block[idx];
                }
            }
            
            if (tmp <= 1e-10)
            {
                const int nrm_len = nrow - (i + 1) * kdim;
                tmp = 0.0;
                for (int k = 0; k < kdim; k++)
                {
                    // R((i + 1) * kdim, j * kdim + k)
                    DTYPE *R_block_k = R + (j*kdim+k) * ldR + (i+1) * kdim;
                    DTYPE tmp1 = CBLAS_NRM2(nrm_len, R_block_k);
                    tmp += tmp1 * tmp1;
                }
                blk_norm[j] = DSQRT(tmp);
            } else {
                // Fast update 2-norm when the new column norm is not so small
                blk_norm[j] = DSQRT(tmp);
            }
        }
        
        // 6. Find next pivot 
        pivot  = i + 1;
        norm_p = 0.0;
        for (int j = i + 1; j < nblk; j++)
        {
            if (blk_norm[j] > norm_p)
            {
                norm_p = blk_norm[j];
                pivot  = j;
            }
        }
    }
    if (rank == -1) rank = max_iter * kdim;
    
    *r = rank;
}


// Partial pivoting QR for ID
// Input parameters:
//   A          : Target matrix, stored in column major
//   stop_type  : Partial QR stop criteria: QR_RANK, QR_REL_NRM, or QR_ABS_NRM
//   stop_param : Pointer to partial QR stop parameter
//   nthreads   : Number of threads used in this function
//   QR_buff    : Working buffer for partial pivoting QR. If kdim == 1, size A->ncol.
//                If kdim > 1, size (2*kdim+2)*A->nrow + (kdim+1)*A->ncol.
//   kdim       : Dimension of tensor kernel's return (column block size)
// Output parameters:
//   A : Matrix R: [R11, R12]
//   p : Matrix A column permutation array, A(:, p) = A * P
void H2P_ID_QR(
    H2P_dense_mat_t A, const int stop_type, void *stop_param, int *p, 
    const int nthreads, DTYPE *QR_buff, const int kdim
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
        tol_norm = param[0] * DSQRT((DTYPE) kdim);
        rel_norm = 1;
    }
    if (stop_type == QR_ABS_NRM)
    {
        DTYPE *param = (DTYPE*) stop_param;
        tol_rank = MIN(A->nrow, A->ncol);
        tol_norm = param[0];
        rel_norm = 0;
    }
    // Use H2P_partial_pivot_QR_kdim() for kdim == 1 also works,
    // but the performance is worse than H2P_partial_pivot_QR()
    if (kdim == 1)
    {
        H2P_partial_pivot_QR(
            A, tol_rank, tol_norm, rel_norm, 
            p, &r, nthreads, QR_buff
        );
    } else {
        H2P_partial_pivot_QR_kdim(
            A, kdim, tol_rank, tol_norm, rel_norm, 
            p, &r, nthreads, QR_buff
        );
    }
    
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
    H2P_int_vec_t J, const int nthreads, DTYPE *QR_buff, int *ID_buff, const int kdim
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
    H2P_ID_QR(A, stop_type, stop_param, J->data, nthreads, QR_buff, kdim);
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
    if (kdim > 1)
    {
        J->data[0] /= kdim;
        for (int i = 1; i < J->length / kdim; i++)
            J->data[i] = J->data[i * kdim] / kdim;
        J->length /= kdim;
    }
    
    if (U_ != NULL) *U_ = U;
}
