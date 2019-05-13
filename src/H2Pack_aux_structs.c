#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

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

// Set the capacity of an initialized H2P_int_vec structure
void H2P_int_vec_set_capacity(H2P_int_vec_t int_vec, const int capacity)
{
    if (capacity > int_vec->capacity)
    {
        int_vec->capacity = capacity;
        int *new_data = (int*) malloc(sizeof(int) * int_vec->capacity);
        assert(new_data != NULL);
        memcpy(new_data, int_vec->data, sizeof(int) * int_vec->length);
        free(int_vec->data);
        int_vec->data = new_data;
    }
}

// Push an integer to the tail of a H2P_int_vec
void H2P_int_vec_push_back(H2P_int_vec_t int_vec, int value)
{
    if (int_vec->capacity == int_vec->length)
        H2P_int_vec_set_capacity(int_vec, int_vec->capacity * 2);
    int_vec->data[int_vec->length] = value;
    int_vec->length++;
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

// Resize an initialized H2P_dense_mat structure, original data in 
// the dense matrix will be unavailable after this operation
void H2P_dense_mat_resize(H2P_dense_mat_t mat, const int nrow, const int ncol)
{
    int new_size = nrow * ncol;
    mat->nrow = nrow;
    mat->ncol = ncol;
    mat->ld   = ncol;
    if (new_size > mat->size)
    {
        mat->size = new_size;
        H2P_free_aligned(mat->data);
        mat->data = H2P_malloc_aligned(sizeof(DTYPE) * mat->size);
        assert(mat->data != NULL);
    }
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
        for (int icol = 0; icol < mat->ncol; icol++) printf("% .10lf  ", mat_row[icol]);
        printf("\n");
    }
}

// ------------------------------------------------------------------- // 
