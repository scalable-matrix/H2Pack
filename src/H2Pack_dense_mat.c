#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "H2Pack_dense_mat.h"

// Initialize a H2P_dense_mat structure
void H2P_dense_mat_init(H2P_dense_mat_t *mat_, const int nrow, const int ncol)
{
    H2P_dense_mat_t mat = (H2P_dense_mat_t) malloc(sizeof(struct H2P_dense_mat));
    assert(mat != NULL);
    
    mat->nrow = MAX(0, nrow);
    mat->ncol = MAX(0, ncol);
    mat->ld   = mat->ncol;
    mat->size = mat->nrow * mat->ncol;
    if (mat->size > 0)
    {
        mat->data = H2P_malloc_aligned(sizeof(DTYPE) * mat->size);
        assert(mat->data != NULL);
    } else {
        mat->data = NULL;
    }
    
    *mat_ = mat;
}

// Destroy a H2P_dense_mat structure
void H2P_dense_mat_destroy(H2P_dense_mat_t mat)
{
    H2P_free_aligned(mat->data);
    mat->data = NULL;
    mat->size = 0;
}

// Copy a block of a dense matrix to another dense matrix
void H2P_dense_mat_copy_block(
    H2P_dense_mat_t src_mat, H2P_dense_mat_t dst_mat,
    const int src_srow, const int src_scol, 
    const int dst_srow, const int dst_scol, 
    const int nrow, const int ncol
)
{
    int src_ld = src_mat->ld;
    int dst_ld = dst_mat->ld;
    DTYPE *src_ptr = src_mat->data + src_srow * src_ld + src_scol;
    DTYPE *dst_ptr = dst_mat->data + dst_srow * dst_ld + dst_scol;
    for (int irow = 0; irow < nrow; irow++)
    {
        memcpy(dst_ptr, src_ptr, sizeof(DTYPE) * ncol);
        src_ptr += src_ld;
        dst_ptr += dst_ld;
    }
}

// Transpose a dense matrix
void H2P_dense_mat_transpose(H2P_dense_mat_t mat)
{
    DTYPE *mat_dst = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * mat->nrow * mat->ncol);
    assert(mat_dst != NULL);
    
    DTYPE *mat_src = mat->data;
    int src_ld = mat->ld;
    int dst_ld = mat->nrow;
    for (int icol = 0; icol < mat->ncol; icol++)
    {
        #pragma omp simd
        for (int irow = 0; irow < mat->nrow; irow++)
        {
            int src_idx = irow * src_ld + icol;
            int dst_idx = icol * dst_ld + irow;
            mat_dst[dst_idx] = mat_src[src_idx];
        }
    }
    
    int dst_nrow = mat->ncol;
    int dst_ncol = mat->nrow;
    mat->nrow = dst_nrow;
    mat->ncol = dst_ncol;
    mat->ld   = dst_ncol;
    mat->size = dst_nrow * dst_ncol;
    H2P_free_aligned(mat->data);
    mat->data = mat_dst;
}

// Permute rows in a H2P_dense_mat structure
void H2P_dense_mat_permute_rows(H2P_dense_mat_t mat, const int *p)
{
    DTYPE *mat_dst = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * mat->nrow * mat->ncol);
    assert(mat_dst != NULL);
    
    for (int irow = 0; irow < mat->nrow; irow++)
    {
        DTYPE *src_row = mat->data + p[irow] * mat->ld;
        DTYPE *dst_row = mat_dst + irow * mat->ncol;
        memcpy(dst_row, src_row, sizeof(DTYPE) * mat->ncol);
    }
    
    mat->ld   = mat->ncol;
    mat->size = mat->nrow * mat->ncol;
    H2P_free_aligned(mat->data);
    mat->data = mat_dst;
}

// Print a H2P_dense_mat structure, for debugging
// Input parameter:
//   mat : H2P_dense_mat structure to be printed
void H2P_dense_mat_print(H2P_dense_mat_t mat)
{
    for (int irow = 0; irow < mat->nrow; irow++)
    {
        DTYPE *mat_row = mat->data + irow * mat->ld;
        for (int icol = 0; icol < mat->ncol; icol++) printf("%.2lf ", mat_row[icol]);
        printf("\n");
    }
}
