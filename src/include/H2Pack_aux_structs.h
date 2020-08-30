#ifndef __H2PACK_AUX_STRUCTS_H__
#define __H2PACK_AUX_STRUCTS_H__

#include "H2Pack_config.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================== H2 tree linked list node ========================== //
struct H2P_tree_node
{
    int   n_child;        // Number of children nodes 
    int   n_node;         // Number of nodes this sub-tree has
    int   po_idx;         // Post-order traversal index of this node
    int   level;          // Level of this node on the tree (root == 0)
    int   height;         // Height of this node on the tree (leaf node == 0)
    int   pt_cluster[2];  // The start and end indices of points belong to this node
    void  **children;     // Size 2^dim, all children nodes of this node
    DTYPE *enbox;         // Size 2*dim, box that encloses all points of this node. 
                          // enbox[0 : dim-1] are the smallest corner coordinate,
                          // enbox[dim : 2*dim-1] are the size of this box.
};
typedef struct H2P_tree_node* H2P_tree_node_p;

// Initialize an H2P_tree_node structure
// Input parameter:
//   dim : Dimension of point coordinate
// Output parameter:
//   node_ : Initialized H2P_tree_node structure
void H2P_tree_node_init(H2P_tree_node_p *node_, const int dim);

// Recursively destroy an H2P_tree_node node and its children nodes
// Input parameter:
//   node : H2P_tree_node structure to be destroyed
void H2P_tree_node_destroy(H2P_tree_node_p *node_);

// ------------------------------------------------------------------------------ // 


// =============== Integer vector, similar to std::vector in C++ ================ //

struct H2P_int_vec
{
    int capacity;    // Capacity of this vector
    int length;      // Current length of this vector
    int *data;       // Data in this vector
};
typedef struct H2P_int_vec* H2P_int_vec_p;

// Initialize an H2P_int_vec structure
// Input parameter:
//   capacity : Initial capacity of the vector. If (capacity <= 0 || capacity >= 65536),
//              capacity will be set as 128.
// Output parameter:
//   int_vec_ : Initialized H2P_int_vec structure
void H2P_int_vec_init(H2P_int_vec_p *int_vec_, int capacity);

// Destroy an H2P_int_vec structure
// Input parameter:
//   int_vec : H2P_int_vec structure to be destroyed
void H2P_int_vec_destroy(H2P_int_vec_p *int_vec_);

// Reset an H2P_int_vec structure to its default capacity and release the memory
// Input parameter:
//   int_vec : H2P_int_vec structure to be reset
void H2P_int_vec_reset(H2P_int_vec_p int_vec);

// Set the capacity of an initialized H2P_int_vec structure. If new 
// capacity > original capacity, allocate a new data buffer and copy 
// values to the new buffer. Otherwise, do nothing.
// Frequently used, inline it.
// Input parameters:
//   int_vec  : Initialized H2P_int_vec structure
//   capacity : New capacity
// Output parameter:
//   int_vec  : H2P_int_vec structure with adjusted capacity
static inline void H2P_int_vec_set_capacity(H2P_int_vec_p int_vec, const int capacity)
{
    if (capacity > int_vec->capacity)
    {
        int *new_data = (int*) malloc(sizeof(int) * capacity);
        ASSERT_PRINTF(new_data != NULL, "Failed to reallocate integer vector of size %d", capacity);
        int_vec->capacity = capacity;
        memcpy(new_data, int_vec->data, sizeof(int) * int_vec->length);
        free(int_vec->data);
        int_vec->data = new_data;
    }
}

// Push an integer to the tail of an H2P_int_vec, inline it for frequently use
// Input parameters:
//   int_vec : H2P_int_vec structure
//   value   : Value to be pushed 
// Output parameter:
//   int_vec : H2P_int_vec structure with the pushed value
static inline void H2P_int_vec_push_back(H2P_int_vec_p int_vec, int value)
{
    if (int_vec->capacity == int_vec->length)
        H2P_int_vec_set_capacity(int_vec, int_vec->capacity * 2);
    int_vec->data[int_vec->length] = value;
    int_vec->length++;
}

// Concatenate values in an H2P_int_vec to another H2P_int_vec
// Input parameters:
//   dst_vec : Destination H2P_int_vec structure
//   src_vec : Source H2P_int_vec structure
// Output parameter:
//   dst_vec : Destination H2P_int_vec structure
void H2P_int_vec_concatenate(H2P_int_vec_p dst_vec, H2P_int_vec_p src_vec);

// Gather elements in an H2P_int_vec to another H2P_int_vec
// Input parameters:
//   src_vec : Source H2P_int_vec structure
//   idx     : Indices of elements to be gathered
// Output parameter:
//   dst_vec : Destination H2P_int_vec structure
void H2P_int_vec_gather(H2P_int_vec_p src_vec, H2P_int_vec_p idx, H2P_int_vec_p dst_vec);

// ------------------------------------------------------------------------------ // 


// ========== Working variables and arrays used in point partitioning =========== //

struct H2P_partition_vars
{
    int curr_po_idx;                // Post-order traversal index
    int max_level;                  // Maximum level of the H2 tree
    int n_leaf_node;                // Number of leaf nodes
    int curr_leaf_idx;              // Index of this leaf node
    int min_adm_level;              // Minimum level of reduced admissible pair
    H2P_int_vec_p r_inadm_pairs;    // Reduced inadmissible pairs
    H2P_int_vec_p r_adm_pairs;      // Reduced admissible pairs
};
typedef struct H2P_partition_vars* H2P_partition_vars_p;

// Initialize an H2P_partition_vars structure
void H2P_partition_vars_init(H2P_partition_vars_p *vars_);

// Destroy an H2P_partition_vars structure
void H2P_partition_vars_destroy(H2P_partition_vars_p *vars_);

// ------------------------------------------------------------------------------ // 


// ========== Simple dense matrix structure with some basic operations ========== //

struct H2P_dense_mat
{
    int   nrow;   // Number of rows
    int   ncol;   // Number of columns
    int   ld;     // Leading dimension, >= ncol
    int   size;   // Size of data, >= nrow * ncol
    DTYPE *data;  // Matrix data
};
typedef struct H2P_dense_mat* H2P_dense_mat_p;

// Initialize an H2P_dense_mat structure
// Input parameters:
//   nrow : Number of rows of the new dense matrix
//   ncol : Number of columns of the new dense matrix
// Output parameter:
//   mat_ : Initialized H2P_dense_mat structure
void H2P_dense_mat_init(H2P_dense_mat_p *mat_, const int nrow, const int ncol);

// Destroy an H2P_dense_mat structure
// Input parameter:
//   mat : H2P_dense_mat structure to be destroyed 
void H2P_dense_mat_destroy(H2P_dense_mat_p *mat_);

// Reset an H2P_dense_mat structure to its default size (0-by-0) and release the memory
// Input parameter:
//   mat : H2P_dense_mat structure to be reset 
void H2P_dense_mat_reset(H2P_dense_mat_p mat);

// Resize an initialized H2P_dense_mat structure, original data in 
// the dense matrix will be unavailable after this operation.
// Frequently used, inline it.
// Input parameters:
//   nrow : Number of rows of the new dense matrix
//   ncol : Number of columns of the new dense matrix
// Output parameter:
//   mat  : Resized H2P_dense_mat structure
static inline void H2P_dense_mat_resize(H2P_dense_mat_p mat, const int nrow, const int ncol)
{
    int new_size = nrow * ncol;
    mat->nrow = nrow;
    mat->ncol = ncol;
    mat->ld   = ncol;
    if (new_size > mat->size)
    {
        free_aligned(mat->data);
        mat->data = (DTYPE*) malloc_aligned(sizeof(DTYPE) * new_size, 64);
        ASSERT_PRINTF(mat->data != NULL, "Failed to reallocate %d * %d dense matrix\n", nrow, ncol);
        mat->size = new_size;
    }
}

// Copy the data in an H2P_dense_mat structure to another H2P_dense_mat structure
// Input parameter:
//   src_mat : Source H2P_dense_mat
// Output parameter:
//   dst_mat : Destination H2P_dense_mat
void H2P_dense_mat_copy(H2P_dense_mat_p src_mat, H2P_dense_mat_p dst_mat);

// Permute rows in an H2P_dense_mat structure
// WARNING: This function DOES NOT perform sanity check!
// Input parameters:
//   mat : H2P_dense_mat structure to be permuted
//   p   : Permutation array. After permutation, the i-th row is the p[i]-th row
//         in the original matrix
// Output parameter:
//   mat : H2P_dense_mat structure with permuted row
void H2P_dense_mat_permute_rows(H2P_dense_mat_p mat, const int *p);

// Select rows in an H2P_dense_mat structure
// WARNING: This function DOES NOT perform sanity check!
// Input parameters:
//   mat     : H2P_dense_mat structure to be selected
//   row_idx : Row index array, sorted in ascending order. The i-th row in the
//             new matrix is the row_idx->data[i]-th row in the original matrix
// Output parameter:
//   mat : H2P_dense_mat structure with selected rows
void H2P_dense_mat_select_rows(H2P_dense_mat_p mat, H2P_int_vec_p row_idx);

// Select columns in an H2P_dense_mat structure
// WARNING: This function DOES NOT perform sanity check!
// Input parameters:
//   mat     : H2P_dense_mat structure to be selected
//   col_idx : Column index array, sorted in ascending order. The i-th column in the
//             new matrix is the col_idx->data[i]-th column in the original matrix
// Output parameter:
//   mat : H2P_dense_mat structure with selected columns
void H2P_dense_mat_select_columns(H2P_dense_mat_p mat, H2P_int_vec_p col_idx);

// Normalize columns in an H2P_dense_mat structure
// Input parameters:
//   mat     : H2P_dense_mat structure to be normalized columns
//   workbuf : H2P_dense_mat structure as working buffer
// Output parameter:
//   mat     : H2P_dense_mat structure with normalized columns
void H2P_dense_mat_normalize_columns(H2P_dense_mat_p mat, H2P_dense_mat_p workbuf);

// Perform GEMM C := alpha * op(A) * op(B) + beta * C
// Input parameters:
//   alpha, beta    : Scaling factors
//   transA, transB : If A / B need to be transposed
//   A, B           : Source matrices
// Output parameter:
//   C : Result matrix, need to be properly resized before entering if beta != 0
void H2P_dense_mat_gemm(
    const DTYPE alpha, const DTYPE beta, const int transA, const int transB, 
    H2P_dense_mat_p A, H2P_dense_mat_p B, H2P_dense_mat_p C
);

// Create a block diagonal matrix created by aligning the input matrices along the diagonal
// Input parameters:
//   mats : Size unknown, candidate H2P_dense_mat_t matrices
//   idx  : Size unknown, indices of the input matrices in the candidate set
// Output parameter:
//   new_mat : The result matrix
void H2P_dense_mat_blkdiag(H2P_dense_mat_p *mats, H2P_int_vec_p idx, H2P_dense_mat_p new_mat);

// Vertically concatenates the input matrices
// Input / output parameters are the same as H2P_dense_mat_blkdiag()
void H2P_dense_mat_vertcat(H2P_dense_mat_p *mats, H2P_int_vec_p idx, H2P_dense_mat_p new_mat);

// Horizontally concatenates the input matrices
// Input / output parameters are the same as H2P_dense_mat_blkdiag()
void H2P_dense_mat_horzcat(H2P_dense_mat_p *mats, H2P_int_vec_p idx, H2P_dense_mat_p new_mat);

// Print an H2P_dense_mat structure, for debugging
// Input parameter:
//   mat : H2P_dense_mat structure to be printed
void H2P_dense_mat_print(H2P_dense_mat_p mat);

// ------------------------------------------------------------------------------ // 


// ================ Thread-local buffer used in build and matvec ================ //

struct H2P_thread_buf
{
    H2P_int_vec_p   idx0;   // H2P_build_H2_UJ_proxy, H2P_build_HSS_UJ_hybrid
    H2P_int_vec_p   idx1;   // H2P_build_H2_UJ_proxy, H2P_build_HSS_UJ_hybrid
    H2P_dense_mat_p mat0;   // H2P_build_H2_UJ_proxy, H2P_build_HSS_UJ_hybrid
    H2P_dense_mat_p mat1;   // H2P_build_H2_UJ_proxy, H2P_build_HSS_UJ_hybrid, H2P_matvec
    H2P_dense_mat_p mat2;   // H2P_build_HSS_UJ_hybrid
    DTYPE  *y;              // Used in H2P_matvec
    double timer;           // Used for profiling
};
typedef struct H2P_thread_buf* H2P_thread_buf_p;

// Initialize an H2P_thread_buf structure
// Input parameter:
//   krnl_mat_size : Size of the kernel matrix
// Output parameter:
//   thread_buf_ : Initialized H2P_thread_buf structure
void H2P_thread_buf_init(H2P_thread_buf_p *thread_buf_, const int krnl_mat_size);

// Destroy an H2P_thread_buf structure
// Input parameter:
//   thread_buf : H2P_thread_buf structure to be destroyed 
void H2P_thread_buf_destroy(H2P_thread_buf_p *thread_buf_);

// Reset an H2P_thread_buf structure (release memory)
// Input parameter:
//   thread_buf : H2P_thread_buf structure to be reset 
void H2P_thread_buf_reset(H2P_thread_buf_p thread_buf);

// ------------------------------------------------------------------------------ // 

#ifdef __cplusplus
}
#endif

#endif
