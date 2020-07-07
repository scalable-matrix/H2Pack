#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "H2Pack_config.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_utils.h"
#include "linalg_lib_wrapper.h"
#include "utils.h"

// ========================== H2P_tree_node ========================== //

// Initialize a H2P_tree_node structure
void H2P_tree_node_init(H2P_tree_node_t *node_, const int dim)
{
    const int max_child = 1 << dim;
    H2P_tree_node_t node = (H2P_tree_node_t) malloc(sizeof(struct H2P_tree_node));
    ASSERT_PRINTF(node != NULL, "Failed to allocate H2P_tree_node structure\n");
    node->children = (void**) malloc(sizeof(H2P_tree_node_t) * max_child);
    node->enbox    = (DTYPE*) malloc(sizeof(DTYPE) * dim * 2);
    ASSERT_PRINTF(
        node->children != NULL && node->enbox != NULL,
        "Failed to allocate arrays in a H2P_tree_node structure\n"
    );
    for (int i = 0; i < max_child; i++) 
        node->children[i] = NULL;
    *node_ = node;
}

// Recursively destroy a H2P_tree_node node and its children nodes
void H2P_tree_node_destroy(H2P_tree_node_t node)
{
    if (node == NULL) return;
    for (int i = 0; i < node->n_child; i++)
    {
        H2P_tree_node_t child_i = (H2P_tree_node_t) node->children[i];
        if (child_i != NULL) H2P_tree_node_destroy(child_i);
        free(child_i);
    }
    free(node->children);
    free(node->enbox);
}

// ------------------------------------------------------------------- // 


// =========================== H2P_int_vec =========================== //

// Initialize a H2P_int_vec structure
void H2P_int_vec_init(H2P_int_vec_t *int_vec_, int capacity)
{
    if (capacity < 0) capacity = 128;
    H2P_int_vec_t int_vec = (H2P_int_vec_t) malloc(sizeof(struct H2P_int_vec));
    ASSERT_PRINTF(int_vec != NULL, "Failed to allocate H2P_int_vec structure\n");
    int_vec->data = (int*) malloc(sizeof(int) * capacity);
    ASSERT_PRINTF(int_vec->data != NULL, "Failed to allocate integer vector of size %d\n", capacity);
    int_vec->capacity = capacity;
    int_vec->length = 0;
    *int_vec_ = int_vec;
}

// Destroy a H2P_int_vec structure
void H2P_int_vec_destroy(H2P_int_vec_t int_vec)
{
    if (int_vec == NULL) return;
    free(int_vec->data);
    int_vec->data     = NULL;
    int_vec->capacity = 0;
    int_vec->length   = 0;
}

// Concatenate values in a H2P_int_vec to another H2P_int_vec
void H2P_int_vec_concatenate(H2P_int_vec_t dst_vec, H2P_int_vec_t src_vec)
{
    int s_len = src_vec->length;
    int d_len = dst_vec->length;
    int new_length = s_len + d_len;
    if (new_length > dst_vec->capacity)
        H2P_int_vec_set_capacity(dst_vec, new_length);
    memcpy(dst_vec->data + d_len, src_vec->data, sizeof(int) * s_len);
    dst_vec->length = new_length;
}

// Gather elements in a H2P_int_vec to another H2P_int_vec
void H2P_int_vec_gather(H2P_int_vec_t src_vec, H2P_int_vec_t idx, H2P_int_vec_t dst_vec)
{
    H2P_int_vec_set_capacity(dst_vec, idx->length);
    for (int i = 0; i < idx->length; i++)
        dst_vec->data[i] = src_vec->data[idx->data[i]];
    dst_vec->length = idx->length;
}

// ------------------------------------------------------------------- // 


// ======================= H2P_partition_vars ======================== //

// Initialize a H2P_partition_vars structure
void H2P_partition_vars_init(H2P_partition_vars_t *part_vars_)
{
    H2P_partition_vars_t part_vars = (H2P_partition_vars_t) malloc(sizeof(struct H2P_partition_vars));
    H2P_int_vec_init(&part_vars->r_adm_pairs,   10240);
    H2P_int_vec_init(&part_vars->r_inadm_pairs, 10240);
    part_vars->curr_po_idx = 0;
    part_vars->max_level   = 0;
    part_vars->n_leaf_node = 0;
    *part_vars_ = part_vars;
}

// Destroy a H2P_partition_vars structure
void H2P_partition_vars_destroy(H2P_partition_vars_t part_vars)
{
    H2P_int_vec_destroy(part_vars->r_adm_pairs);
    H2P_int_vec_destroy(part_vars->r_inadm_pairs);
    free(part_vars->r_adm_pairs);
    free(part_vars->r_inadm_pairs);
    free(part_vars);
}

// ------------------------------------------------------------------- // 


// ========================== H2P_dense_mat ========================== //

// Initialize a H2P_dense_mat structure
void H2P_dense_mat_init(H2P_dense_mat_t *mat_, const int nrow, const int ncol)
{
    H2P_dense_mat_t mat = (H2P_dense_mat_t) malloc(sizeof(struct H2P_dense_mat));
    ASSERT_PRINTF(mat != NULL, "Failed to allocate H2P_dense_mat structure\n");
    
    mat->nrow = MAX(0, nrow);
    mat->ncol = MAX(0, ncol);
    mat->ld   = mat->ncol;
    mat->size = mat->nrow * mat->ncol;
    if (mat->size > 0)
    {
        mat->data = malloc_aligned(sizeof(DTYPE) * mat->size, 64);
        ASSERT_PRINTF(mat->data != NULL, "Failed to allocate %d * %d dense matrix\n", nrow, ncol);
    } else {
        mat->data = NULL;
    }
    
    *mat_ = mat;
}

// Destroy a H2P_dense_mat structure
void H2P_dense_mat_destroy(H2P_dense_mat_t mat)
{
    if (mat == NULL) return;
    free_aligned(mat->data);
    mat->data = NULL;
    mat->size = 0;
    mat->nrow = 0;
    mat->ncol = 0;
    mat->ld   = 0;
}

// Copy the data in a H2P_dense_mat structure to another H2P_dense_mat structure
void H2P_dense_mat_copy(H2P_dense_mat_t src_mat, H2P_dense_mat_t dst_mat)
{
    H2P_dense_mat_resize(dst_mat, src_mat->nrow, src_mat->ncol);
    for (int i = 0; i < src_mat->nrow; i++)
    {
        DTYPE *src_ptr = src_mat->data + i * src_mat->ld;
        DTYPE *dst_ptr = dst_mat->data + i * dst_mat->ld;
        memcpy(dst_ptr, src_ptr, sizeof(DTYPE) * src_mat->ncol);
    }
}

// Permute rows in a H2P_dense_mat structure
void H2P_dense_mat_permute_rows(H2P_dense_mat_t mat, const int *p)
{
    DTYPE *mat_dst = (DTYPE*) malloc_aligned(sizeof(DTYPE) * mat->nrow * mat->ncol, 64);
    ASSERT_PRINTF(mat_dst != NULL, "Failed to allocate buffer of size %d * %d\n", mat->nrow, mat->ncol);
    
    for (int irow = 0; irow < mat->nrow; irow++)
    {
        DTYPE *src_row = mat->data + p[irow] * mat->ld;
        DTYPE *dst_row = mat_dst + irow * mat->ncol;
        memcpy(dst_row, src_row, sizeof(DTYPE) * mat->ncol);
    }
    
    mat->ld   = mat->ncol;
    mat->size = mat->nrow * mat->ncol;
    free_aligned(mat->data);
    mat->data = mat_dst;
}

// Select rows in a H2P_dense_mat structure
void H2P_dense_mat_select_rows(H2P_dense_mat_t mat, H2P_int_vec_t row_idx)
{
    for (int irow = 0; irow < row_idx->length; irow++)
    {
        DTYPE *src = mat->data + row_idx->data[irow] * mat->ld;
        DTYPE *dst = mat->data + irow * mat->ld;
        if (src != dst) memcpy(dst, src, sizeof(DTYPE) * mat->ncol);
    }
    mat->nrow = row_idx->length;
}

// Select columns in a H2P_dense_mat structure
void H2P_dense_mat_select_columns(H2P_dense_mat_t mat, H2P_int_vec_t col_idx)
{
    for (int irow = 0; irow < mat->nrow; irow++)
    {
        DTYPE *mat_row = mat->data + irow * mat->ld;
        for (int icol = 0; icol < col_idx->length; icol++)
            mat_row[icol] = mat_row[col_idx->data[icol]];
    }
    mat->ncol = col_idx->length;
    for (int irow = 1; irow < mat->nrow; irow++)
    {
        DTYPE *src = mat->data + irow * mat->ld;
        DTYPE *dst = mat->data + irow * mat->ncol;
        memmove(dst, src, sizeof(DTYPE) * mat->ncol);
    }
    mat->ld = mat->ncol;
}

// Normalize columns in a H2P_dense_mat structure
void H2P_dense_mat_normalize_columns(H2P_dense_mat_t mat, H2P_dense_mat_t workbuf)
{
    int nrow = mat->nrow, ncol = mat->ncol;
    H2P_dense_mat_resize(workbuf, 1, ncol);
    DTYPE *inv_2norm = workbuf->data;
    
    /*
    #pragma omp simd
    for (int icol = 0; icol < ncol; icol++) 
        inv_2norm[icol] = mat->data[icol] * mat->data[icol];
    for (int irow = 1; irow < nrow; irow++)
    {
        double *mat_row = mat->data + irow * mat->ld;
        #pragma omp simd
        for (int icol = 0; icol < ncol; icol++) 
            inv_2norm[icol] += mat_row[icol] * mat_row[icol];
    }
    
    #pragma omp simd
    for (int icol = 0; icol < ncol; icol++) 
        inv_2norm[icol] = 1.0 / DSQRT(inv_2norm[icol]);
    */

    // Slower, but more accurate
    for (int icol = 0; icol < ncol; icol++)
        inv_2norm[icol] = 1.0 / CBLAS_NRM2(nrow, mat->data + icol, ncol);
    
    for (int irow = 0; irow < nrow; irow++)
    {
        double *mat_row = mat->data + irow * mat->ld;
        #pragma omp simd
        for (int icol = 0; icol < ncol; icol++) 
            mat_row[icol] *= inv_2norm[icol];
    }
}

// Perform GEMM C := alpha * op(A) * op(B) + beta * C
void H2P_dense_mat_gemm(
    const DTYPE alpha, const DTYPE beta, const int transA, const int transB, 
    H2P_dense_mat_t A, H2P_dense_mat_t B, H2P_dense_mat_t C
)
{
    int M, N, KA, KB;
    CBLAS_TRANSPOSE transA_, transB_;
    if (transA == 0)
    {
        transA_ = CblasNoTrans;
        M  = A->nrow;
        KA = A->ncol;
    } else {
        transA_ = CblasTrans;
        M  = A->ncol;
        KA = A->nrow;
    }
    if (transB == 0)
    {
        transB_ = CblasNoTrans;
        N  = B->ncol;
        KB = B->nrow;
    } else {
        transB_ = CblasTrans;
        N  = B->nrow;
        KB = B->ncol;
    }
    if (KA != KB)
    {
        ERROR_PRINTF("GEMM size mismatched: A[%d * %d], B[%d * %d]\n", M, KA, KB, N);
        return;
    }
    if (beta == 0.0 && (C->nrow < M || C->ncol < N)) H2P_dense_mat_resize(C, M, N);
    CBLAS_GEMM(
        CblasRowMajor, transA_, transB_, M, N, KA,
        alpha, A->data, A->ld, B->data, B->ld, 
        beta,  C->data, C->ld
    );
}

// Create a block diagonal matrix created by aligning the input matrices along the diagonal
void H2P_dense_mat_blkdiag(H2P_dense_mat_t *mats, H2P_int_vec_t idx, H2P_dense_mat_t new_mat)
{
    int nrow = 0, ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2P_dense_mat_t mat_i = mats[idx->data[i]];
        nrow += mat_i->nrow;
        ncol += mat_i->ncol;
    }
    H2P_dense_mat_resize(new_mat, nrow, ncol);
    memset(new_mat->data, 0, sizeof(DTYPE) * nrow * ncol);
    nrow = 0; ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2P_dense_mat_t mat_i = mats[idx->data[i]];
        int nrow_i = mat_i->nrow;
        int ncol_i = mat_i->ncol;
        DTYPE *dst = new_mat->data + nrow * new_mat->ld + ncol;
        H2P_copy_matrix_block(nrow_i, ncol_i, mat_i->data, mat_i->ld, dst, new_mat->ld);
        nrow += nrow_i;
        ncol += ncol_i;
    }
}

// Vertically concatenates the input matrices
void H2P_dense_mat_vertcat(H2P_dense_mat_t *mats, H2P_int_vec_t idx, H2P_dense_mat_t new_mat)
{
    int nrow = 0, ncol = mats[idx->data[0]]->ncol;
    for (int i = 0; i < idx->length; i++)
    {
        H2P_dense_mat_t mat_i = mats[idx->data[i]];
        if (mat_i->ncol != ncol)
        {
            ERROR_PRINTF("%d-th matrix has %d columns, 1st matrix has %d columns\n", i+1, mat_i->ncol, ncol);
            return;
        }
        nrow += mat_i->nrow;
    }
    H2P_dense_mat_resize(new_mat, nrow, ncol);
    nrow = 0; ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2P_dense_mat_t mat_i = mats[idx->data[i]];
        int nrow_i = mat_i->nrow;
        int ncol_i = mat_i->ncol;
        DTYPE *dst = new_mat->data + nrow * new_mat->ld;
        H2P_copy_matrix_block(nrow_i, ncol_i, mat_i->data, mat_i->ld, dst, new_mat->ld);
        nrow += nrow_i;
    }
}

// Horizontally concatenates the input matrices
void H2P_dense_mat_horzcat(H2P_dense_mat_t *mats, H2P_int_vec_t idx, H2P_dense_mat_t new_mat)
{
    int nrow = mats[idx->data[0]]->nrow, ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2P_dense_mat_t mat_i = mats[idx->data[i]];
        if (mat_i->nrow != nrow)
        {
            ERROR_PRINTF("%d-th matrix has %d rows, 1st matrix has %d rows\n", i+1, mat_i->nrow, nrow);
            return;
        }
        ncol += mat_i->ncol;
    }
    H2P_dense_mat_resize(new_mat, nrow, ncol);
    memset(new_mat->data, 0, sizeof(DTYPE) * nrow * ncol);
    nrow = 0; ncol = 0;
    for (int i = 0; i < idx->length; i++)
    {
        H2P_dense_mat_t mat_i = mats[idx->data[i]];
        int nrow_i = mat_i->nrow;
        int ncol_i = mat_i->ncol;
        DTYPE *dst = new_mat->data + ncol;
        H2P_copy_matrix_block(nrow_i, ncol_i, mat_i->data, mat_i->ld, dst, new_mat->ld);
        ncol += mat_i->ncol;
    }
}

// Print a H2P_dense_mat structure, for debugging
void H2P_dense_mat_print(H2P_dense_mat_t mat)
{
    for (int irow = 0; irow < mat->nrow; irow++)
    {
        DTYPE *mat_row = mat->data + irow * mat->ld;
        for (int icol = 0; icol < mat->ncol; icol++) printf("% .4lf  ", mat_row[icol]);
        printf("\n");
    }
}

// ------------------------------------------------------------------- // 


// ========================== H2P_thread_buf ========================= //

void H2P_thread_buf_init(H2P_thread_buf_t *thread_buf_, const int krnl_mat_size)
{
    H2P_thread_buf_t thread_buf = (H2P_thread_buf_t) malloc(sizeof(struct H2P_thread_buf));
    ASSERT_PRINTF(thread_buf != NULL, "Failed to allocate H2P_thread_buf structure\n");
    H2P_int_vec_init(&thread_buf->idx0, 1024);
    H2P_int_vec_init(&thread_buf->idx1, 1024);
    H2P_dense_mat_init(&thread_buf->mat0, 1024, 1);
    H2P_dense_mat_init(&thread_buf->mat1, 1024, 1);
    H2P_dense_mat_init(&thread_buf->mat2, 1024, 1);
    thread_buf->y = (DTYPE*) malloc_aligned(sizeof(DTYPE) * krnl_mat_size, 64);
    ASSERT_PRINTF(thread_buf->y != NULL, "Failed to allocate y of size %d in H2P_thread_buf\n", krnl_mat_size);
    *thread_buf_ = thread_buf;
}

void H2P_thread_buf_destroy(H2P_thread_buf_t thread_buf)
{
    if (thread_buf == NULL) return;
    H2P_int_vec_destroy(thread_buf->idx0);
    H2P_int_vec_destroy(thread_buf->idx1);
    H2P_dense_mat_destroy(thread_buf->mat0);
    H2P_dense_mat_destroy(thread_buf->mat1);
    H2P_dense_mat_destroy(thread_buf->mat2);
    free_aligned(thread_buf->y);
}

void H2P_thread_buf_reset(H2P_thread_buf_t thread_buf)
{
    if (thread_buf == NULL) return;
    H2P_int_vec_destroy(thread_buf->idx0);
    H2P_int_vec_destroy(thread_buf->idx1);
    H2P_dense_mat_destroy(thread_buf->mat0);
    H2P_dense_mat_destroy(thread_buf->mat1);
    H2P_int_vec_set_capacity(thread_buf->idx0, 1024);
    H2P_int_vec_set_capacity(thread_buf->idx1, 1024);
    H2P_dense_mat_resize(thread_buf->mat0, 1024, 1);
    H2P_dense_mat_resize(thread_buf->mat1, 1024, 1);
    H2P_dense_mat_resize(thread_buf->mat2, 1024, 1);
}

// ------------------------------------------------------------------- // 