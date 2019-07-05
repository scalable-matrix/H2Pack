#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "H2Pack_aux_structs.h"

// ========================== H2P_tree_node ========================== //

// Initialize a H2P_tree_node structure
void H2P_tree_node_init(H2P_tree_node_t *node_, const int dim)
{
    const int max_child = 1 << dim;
    H2P_tree_node_t node = (H2P_tree_node_t) malloc(sizeof(struct H2P_tree_node));
    assert(node != NULL);
    node->children = (void**) malloc(sizeof(H2P_tree_node_t) * max_child);
    node->enbox = (DTYPE*) malloc(sizeof(DTYPE) * dim * 2);
    assert(node->children != NULL && node->enbox != NULL);
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
    if (capacity < 0 || capacity > 65536) capacity = 128;
    H2P_int_vec_t int_vec = (H2P_int_vec_t) malloc(sizeof(struct H2P_int_vec));
    assert(int_vec != NULL);
    int_vec->capacity = capacity;
    int_vec->length = 0;
    int_vec->data = (int*) malloc(sizeof(int) * capacity);
    assert(int_vec->data != NULL);
    *int_vec_ = int_vec;
}

// Destroy a H2P_int_vec structure
void H2P_int_vec_destroy(H2P_int_vec_t int_vec)
{
    if (int_vec == NULL) return;
    free(int_vec->data);
    int_vec->capacity = 0;
    int_vec->length = 0;
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


// ========================== H2P_dense_mat ========================== //

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
    if (mat == NULL) return;
    H2P_free_aligned(mat->data);
    mat->data = NULL;
    mat->size = 0;
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
    
    for (int irow = 0; irow < nrow; irow++)
    {
        double *mat_row = mat->data + irow * mat->ld;
        #pragma omp simd
        for (int icol = 0; icol < ncol; icol++) 
            mat_row[icol] *= inv_2norm[icol];
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
    assert(thread_buf != NULL);
    H2P_int_vec_init(&thread_buf->idx0, 1024);
    H2P_int_vec_init(&thread_buf->idx1, 1024);
    H2P_int_vec_init(&thread_buf->idx2, 1024);
    H2P_int_vec_init(&thread_buf->idx3, 1024);
    H2P_int_vec_init(&thread_buf->idx4, 1024);
    H2P_dense_mat_init(&thread_buf->mat0, 1024, 1);
    H2P_dense_mat_init(&thread_buf->mat1, 1024, 1);
    H2P_dense_mat_init(&thread_buf->mat2, 1024, 1);
    thread_buf->y = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * krnl_mat_size);
    assert(thread_buf->y != NULL);
    *thread_buf_ = thread_buf;
}

void H2P_thread_buf_destroy(H2P_thread_buf_t thread_buf)
{
    if (thread_buf == NULL) return;
    H2P_int_vec_destroy(thread_buf->idx0);
    H2P_int_vec_destroy(thread_buf->idx1);
    H2P_int_vec_destroy(thread_buf->idx2);
    H2P_int_vec_destroy(thread_buf->idx3);
    H2P_int_vec_destroy(thread_buf->idx4);
    H2P_dense_mat_destroy(thread_buf->mat0);
    H2P_dense_mat_destroy(thread_buf->mat1);
    H2P_dense_mat_destroy(thread_buf->mat2);
    H2P_free_aligned(thread_buf->y);
}

// ------------------------------------------------------------------- // 