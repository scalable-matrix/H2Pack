// ---------------------------------------------------------------
// @brief  : CSRPlus matrix implementation file 
// @author : Hua Huang <huangh223@gatech.edu>
//           Edmond Chow <echow@cc.gatech.edu>
// 
//     Copyright (c) 2017-2020 Georgia Institute of Technology      
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "CSRPlus.h"

static void qsort_int_dbl_pair(int *key, double *val, int l, int r)
{
    int i = l, j = r, tmp_key;
    int mid_key = key[(l + r) / 2];
    double tmp_val;
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
    if (i < r) qsort_int_dbl_pair(key, val, i, r);
    if (j > l) qsort_int_dbl_pair(key, val, l, j);
}

// Initialize a CSRP_mat structure using a COO matrix
void CSRP_init_with_COO_mat(
    const int nrow, const int ncol, const int nnz, const int *row,
    const int *col, const double *val, CSRP_mat_t *csrp_mat_
)
{
    CSRP_mat_t csrp_mat = (CSRP_mat_t) malloc(sizeof(CSRP_mat_s));
    
    csrp_mat->nrow    = nrow;
    csrp_mat->ncol    = ncol;
    csrp_mat->nnz     = nnz;
    csrp_mat->row_ptr = (int*)    malloc(sizeof(int)    * (nrow + 1));
    csrp_mat->col     = (int*)    malloc(sizeof(int)    * nnz);
    csrp_mat->val     = (double*) malloc(sizeof(double) * nnz);
    assert(csrp_mat->row_ptr != NULL);
    assert(csrp_mat->col     != NULL);
    assert(csrp_mat->val     != NULL);
    
    csrp_mat->nnz_spos  = NULL;
    csrp_mat->nnz_epos  = NULL;
    csrp_mat->first_row = NULL;
    csrp_mat->last_row  = NULL;
    csrp_mat->fr_intact = NULL;
    csrp_mat->lr_intact = NULL;
    csrp_mat->fr_res    = NULL;
    csrp_mat->lr_res    = NULL;
    
    int    *row_ptr = csrp_mat->row_ptr;
    int    *col_    = csrp_mat->col;
    double *val_    = csrp_mat->val;
    memset(row_ptr, 0, sizeof(int) * (nrow + 1));
    // Get the number of non-zeros in each row
    for (int i = 0; i < nnz; i++)
        row_ptr[row[i] + 1]++;
    // Calculate the displacement of 1st non-zero in each row
    for (int i = 2; i <= nrow; i++)
        row_ptr[i] += row_ptr[i - 1];
    // Use row_ptr to bucket sort col[] and val[]
    for (int i = 0; i < nnz; i++)
    {
        int idx = row_ptr[row[i]];
        col_[idx] = col[i];
        val_[idx] = val[i];
        row_ptr[row[i]]++;
    }
    // Reset row_ptr
    for (int i = nrow; i >= 1; i--)
        row_ptr[i] = row_ptr[i - 1];
    row_ptr[0] = 0;
    // Sort the non-zeros in each row according to column indices
    #pragma omp parallel for
    for (int i = 0; i < nrow; i++)
        qsort_int_dbl_pair(col_, val_, row_ptr[i], row_ptr[i + 1] - 1);
    
    *csrp_mat_ = csrp_mat;
}

// Free a CSRP_mat structure
void CSRP_free(CSRP_mat_t csrp_mat)
{
    if (csrp_mat == NULL) return;
    free(csrp_mat->row_ptr);
    free(csrp_mat->col);
    free(csrp_mat->val);
    free(csrp_mat->nnz_spos);
    free(csrp_mat->nnz_epos);
    free(csrp_mat->first_row);
    free(csrp_mat->last_row);
    free(csrp_mat->fr_intact);
    free(csrp_mat->lr_intact);
    free(csrp_mat->fr_res);
    free(csrp_mat->lr_res);
}

static void partition_block_equal(const int len, const int nblk, int *displs)
{
    int bs0 = len / nblk;
    int rem = len % nblk;
    int bs1 = (rem > 0) ? bs0 + 1 : bs0;
    displs[0] = 0;
    for (int i = 0; i < rem; i++)
        displs[i + 1] = displs[i] + bs1;
    for (int i = rem; i < nblk; i++)
        displs[i + 1] = displs[i] + bs0;
}

static int calc_lower_bound(const int *a, int n, int x) 
{
    int l = 0, h = n;
    while (l < h) 
    {
        int mid = l + (h - l) / 2;
        if (x <= a[mid]) h = mid;
        else l = mid + 1;
    }
    return l;
}

// Partition a CSR matrix into multiple blocks with the same nnz
// for multiple threads execution of SpMV
void CSRP_partition_multithread(CSRP_mat_t csrp_mat, const int nblk, const int nthread)
{
    csrp_mat->nblk      = nblk;
    csrp_mat->nthread   = nthread;
    csrp_mat->nnz_spos  = (int*)    malloc(sizeof(int)    * nblk);
    csrp_mat->nnz_epos  = (int*)    malloc(sizeof(int)    * nblk);
    csrp_mat->first_row = (int*)    malloc(sizeof(int)    * nblk);
    csrp_mat->last_row  = (int*)    malloc(sizeof(int)    * nblk);
    csrp_mat->fr_intact = (int*)    malloc(sizeof(int)    * nblk);
    csrp_mat->lr_intact = (int*)    malloc(sizeof(int)    * nblk);
    csrp_mat->fr_res    = (double*) malloc(sizeof(double) * nblk);
    csrp_mat->lr_res    = (double*) malloc(sizeof(double) * nblk);
    assert(csrp_mat->nnz_spos  != NULL);
    assert(csrp_mat->nnz_epos  != NULL);
    assert(csrp_mat->first_row != NULL);
    assert(csrp_mat->last_row  != NULL);
    assert(csrp_mat->fr_intact != NULL);
    assert(csrp_mat->lr_intact != NULL);
    assert(csrp_mat->fr_res    != NULL);
    assert(csrp_mat->lr_res    != NULL);
    
    int nnz  = csrp_mat->nnz;
    int nrow = csrp_mat->nrow;
    int *row_ptr = csrp_mat->row_ptr;
    
    int *nnz_displs = (int *) malloc((nblk + 1) * sizeof(int));
    partition_block_equal(nnz, nblk, nnz_displs);
    
    for (int iblk = 0; iblk < nblk; iblk++)
    {
        int iblk_nnz_spos = nnz_displs[iblk];
        int iblk_nnz_epos = nnz_displs[iblk + 1] - 1;
        
        int spos_in_row = calc_lower_bound(row_ptr, nrow + 1, iblk_nnz_spos);
        int epos_in_row = calc_lower_bound(row_ptr, nrow + 1, iblk_nnz_epos);
        if (row_ptr[spos_in_row] > iblk_nnz_spos) spos_in_row--;
        if (row_ptr[epos_in_row] > iblk_nnz_epos) epos_in_row--;
        // Note: It is possible that the last nnz is the first nnz in a row,
        // and there are some empty rows between the last row and previous non-empty row
        while (row_ptr[epos_in_row] == row_ptr[epos_in_row + 1]) epos_in_row++;
        
        csrp_mat->nnz_spos[iblk]  = iblk_nnz_spos;
        csrp_mat->nnz_epos[iblk]  = iblk_nnz_epos;
        csrp_mat->first_row[iblk] = spos_in_row;
        csrp_mat->last_row[iblk]  = epos_in_row;
        
        if ((epos_in_row - spos_in_row) >= 1)
        {
            int fr_intact = (iblk_nnz_spos == row_ptr[spos_in_row]);
            int lr_intact = (iblk_nnz_epos == row_ptr[epos_in_row + 1] - 1);
            
            csrp_mat->fr_intact[iblk] = fr_intact;
            csrp_mat->lr_intact[iblk] = lr_intact;
        } else {
            // Mark that this thread only handles a segment of a row 
            csrp_mat->fr_intact[iblk] = 0;
            csrp_mat->lr_intact[iblk] = -1;
        }
    }
    
    csrp_mat->last_row[nblk - 1] = nrow - 1;
    csrp_mat->nnz_epos[nblk - 1] = row_ptr[nrow] - 1;
    
    free(nnz_displs);
}

// Use Use first-touch policy to optimize the storage of CSR arrays in a CSRP_mat structure
void CSRP_optimize_NUMA(CSRP_mat_t csrp_mat)
{
    int nnz     = csrp_mat->nnz;
    int nrow    = csrp_mat->nrow;
    int nblk    = csrp_mat->nblk;
    int nthread = csrp_mat->nthread;
    
    int    *row_ptr = (int*)    malloc(sizeof(int)    * (nrow + 1));
    int    *col     = (int*)    malloc(sizeof(int)    * nnz);
    double *val     = (double*) malloc(sizeof(double) * nnz);
    assert(row_ptr != NULL);
    assert(col     != NULL);
    assert(val     != NULL);
    
    #pragma omp parallel num_threads(nthread)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < nrow + 1; i++)
            row_ptr[i] = csrp_mat->row_ptr[i];
        
        #pragma omp for schedule(static)
        for (int iblk = 0; iblk < nblk; iblk++)
        {
            int nnz_spos = csrp_mat->nnz_spos[iblk];
            int nnz_epos = csrp_mat->nnz_epos[iblk];
            for (int i = nnz_spos; i <= nnz_epos; i++)
            {
                col[i] = csrp_mat->col[i];
                val[i] = csrp_mat->val[i];
            }
        }
    }
    
    free(csrp_mat->row_ptr);
    free(csrp_mat->col);
    free(csrp_mat->val);
    
    csrp_mat->row_ptr = row_ptr;
    csrp_mat->col     = col;
    csrp_mat->val     = val;
}

static double CSR_SpMV_row_seg(
    const int seg_len, const int *__restrict col, 
    const double *__restrict val, const double *__restrict x
)
{
    register double res = 0.0;
    #pragma omp simd
    for (int idx = 0; idx < seg_len; idx++)
        res += val[idx] * x[col[idx]];
    return res;
}

static void CSR_SpMV_row_block(
    const int srow, const int erow,
    const int *row_ptr, const int *col, const double *val, 
    const double *__restrict x, double *__restrict y
)
{
    for (int irow = srow; irow < erow; irow++)
    {
        register double res = 0.0;
        #pragma omp simd
        for (int idx = row_ptr[irow]; idx < row_ptr[irow + 1]; idx++)
            res += val[idx] * x[col[idx]];
        y[irow] = res;
    }
}

static void CSRP_SpMV_block(CSRP_mat_t csrp_mat, const int iblk, const double *x, double *y)
{
    int    *row_ptr   = csrp_mat->row_ptr;
    int    *col       = csrp_mat->col;
    int    *first_row = csrp_mat->first_row;
    int    *last_row  = csrp_mat->last_row;
    int    *nnz_spos  = csrp_mat->nnz_spos;
    int    *nnz_epos  = csrp_mat->nnz_epos;
    int    *fr_intact = csrp_mat->fr_intact;
    int    *lr_intact = csrp_mat->lr_intact;
    double *val       = csrp_mat->val;
    double *fr_res    = csrp_mat->fr_res;
    double *lr_res    = csrp_mat->lr_res;
    
    if (first_row[iblk] == last_row[iblk])
    {
        // This thread handles a segment on 1 row
        
        int iblk_nnz_spos = nnz_spos[iblk];
        int iblk_nnz_epos = nnz_epos[iblk];
        int iblk_seg_len  = iblk_nnz_epos - iblk_nnz_spos + 1;
        
        fr_res[iblk] = CSR_SpMV_row_seg(iblk_seg_len, col + iblk_nnz_spos, val + iblk_nnz_spos, x);
        lr_res[iblk] = 0.0;
        y[first_row[iblk]] = 0.0;
    } else {
        // This thread handles segments on multiple rows
        
        int first_intact_row = first_row[iblk];
        int last_intact_row  = last_row[iblk];
        
        if (fr_intact[iblk] == 0)
        {
            int iblk_nnz_spos = nnz_spos[iblk];
            int iblk_nnz_epos = row_ptr[first_intact_row + 1];
            int iblk_seg_len  = iblk_nnz_epos - iblk_nnz_spos;
            
            fr_res[iblk] = CSR_SpMV_row_seg(iblk_seg_len, col + iblk_nnz_spos, val + iblk_nnz_spos, x);
            y[first_intact_row] = 0.0;
            first_intact_row++;
        }
        
        if (lr_intact[iblk] == 0)
        {
            int iblk_nnz_spos = row_ptr[last_intact_row];
            int iblk_nnz_epos = nnz_epos[iblk];
            int iblk_seg_len  = iblk_nnz_epos - iblk_nnz_spos + 1;
            
            lr_res[iblk] = CSR_SpMV_row_seg(iblk_seg_len, col + iblk_nnz_spos, val + iblk_nnz_spos, x);
            y[last_intact_row] = 0.0;
            last_intact_row--;
        }
        
        CSR_SpMV_row_block(
            first_intact_row, last_intact_row + 1,
            row_ptr, col, val, x, y
        );
    }
}

// Perform OpenMP parallelized CSR SpMV with a CSRP_mat structure
void CSRP_SpMV(CSRP_mat_t csrp_mat, const double *x, double *y)
{
    int nblk    = csrp_mat->nblk;
    int nthread = csrp_mat->nthread;
    
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (int iblk = 0; iblk < nblk; iblk++)
        CSRP_SpMV_block(csrp_mat, iblk, x, y);
    
    for (int iblk = 0; iblk < nblk; iblk++)
    {
        if (csrp_mat->fr_intact[iblk] == 0)
        {
            int first_row = csrp_mat->first_row[iblk];
            y[first_row] += csrp_mat->fr_res[iblk];
        }
        
        if (csrp_mat->lr_intact[iblk] == 0)
        {
            int last_row = csrp_mat->last_row[iblk];
            y[last_row] += csrp_mat->lr_res[iblk];
        }
    }
}

