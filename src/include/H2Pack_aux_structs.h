#ifndef __H2PACK_AUX_STRUCTS_H__
#define __H2PACK_AUX_STRUCTS_H__

#include "H2Pack_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================== H2 tree linked list node ========================== //
struct H2P_tree_node
{
    int   n_child;     // Number of children nodes 
    int   n_node;      // Number of nodes this sub-tree has
    int   po_idx;      // Post-order traversal index of this node
    int   level;       // Level of this node on the tree (root == 0)
    int   height;      // Height of this node on the tree (leaf node == 0)
    int   cluster[2];  // The start and end indices of points belong to this node
    void  **children;  // Size 2^dim, all children nodes of this node
    DTYPE *enbox;      // Size 2*dim, box that encloses all points of this node. 
                       // enbox[0 : dim-1] are the smallest corner coordinate,
                       // enbox[dim : 2*dim-1] are the size of this box.
};
typedef struct H2P_tree_node* H2P_tree_node_t;

// Initialize a H2P_tree_node structure
// Input parameter:
//   dim : Dimension of point coordinate
// Output parameter:
//   node_ : Initialized H2P_tree_node structure
void H2P_tree_node_init(H2P_tree_node_t *node_, const int dim);

// Recursively destroy a H2P_tree_node node and its children nodes
// Input parameter:
//   node : H2P_tree_node structure to be destroyed
void H2P_tree_node_destroy(H2P_tree_node_t node);

// ------------------------------------------------------------------------------ // 


// =============== Integer vector, similar to std::vector in C++ ================ //

struct H2P_int_vec
{
    int capacity;    // Capacity of this vector
    int length;      // Current length of this vector
    int *data;       // Data in this vector
};
typedef struct H2P_int_vec* H2P_int_vec_t;

// Initialize a H2P_int_vec structure
// Input parameter:
//   capacity : Initial capacity of the vector. If (capacity <= 0 || capacity >= 65536),
//              capacity will be set as 128.
// Output parameter:
//   int_vec_ : Initialized H2P_int_vec structure
void H2P_int_vec_init(H2P_int_vec_t *int_vec_, int capacity);

// Destroy a H2P_int_vec structure
// Input parameter:
//   int_vec : H2P_int_vec structure to be destroyed
void H2P_int_vec_destroy(H2P_int_vec_t int_vec);

// Set the capacity of an initialized H2P_int_vec structure. 
// If new capacity > original capacity, allocate a new data buffer and 
// copy values to the new buffer. Otherwise, do nothing.
// Input parameters:
//   int_vec  : Initialized H2P_int_vec structure
//   capacity : New capacity
// Output parameter:
//   int_vec  : H2P_int_vec structure with adjusted capacity
void H2P_int_vec_set_capacity(H2P_int_vec_t int_vec, const int capacity);

// Push an integer to the tail of a H2P_int_vec
// Input parameters:
//   int_vec : H2P_int_vec structure
//   value   : Value to be pushed 
// Output parameter:
//   int_vec : H2P_int_vec structure with the pushed value
void H2P_int_vec_push_back(H2P_int_vec_t int_vec, int value);

// Concatenate values in a H2P_int_vec to another H2P_int_vec
// Input parameters:
//   dst_vec : Destination H2P_int_vec structure
//   src_vec : Source H2P_int_vec structure
// Output parameter:
//   dst_vec : Destination H2P_int_vec structure
void H2P_int_vec_concatenate(H2P_int_vec_t dst_vec, H2P_int_vec_t src_vec);

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
typedef struct H2P_dense_mat* H2P_dense_mat_t;

// Initialize a H2P_dense_mat structure
// Input parameters:
//   nrow : Number of rows of the new dense matrix
//   ncol : Number of columns of the new dense matrix
// Output parameter:
//   mat_ : Initialized H2P_dense_mat structure
void H2P_dense_mat_init(H2P_dense_mat_t *mat_, const int nrow, const int ncol);

// Destroy a H2P_dense_mat structure
// Input parameter:
//   mat : H2P_dense_mat structure to be destroyed 
void H2P_dense_mat_destroy(H2P_dense_mat_t mat);

// Resize an initialized H2P_dense_mat structure, original data in 
// the dense matrix will be unavailable after this operation
// Input parameters:
//   nrow : Number of rows of the new dense matrix
//   ncol : Number of columns of the new dense matrix
// Output parameter:
//   mat  : Resized H2P_dense_mat structure
void H2P_dense_mat_resize(H2P_dense_mat_t mat, const int nrow, const int ncol);

// Permute rows in a H2P_dense_mat structure
// WARNING: This function DOES NOT perform sanity check!
// Input parameter:
//   mat : H2P_dense_mat structure to be permuted
//   p   : Permutation array. After permutation, the i-th row is the p[i]-th row
//         in the original matrix
// Output parameter:
//   mat : H2P_dense_mat structure with permuted row
void H2P_dense_mat_permute_rows(H2P_dense_mat_t mat, const int *p);

// Select rows in a H2P_dense_mat structure
// WARNING: This function DOES NOT perform sanity check!
// Input parameter:
//   mat     : H2P_dense_mat structure to be selected
//   row_idx : Row index array, sorted in ascending order. The i-th row in the
//             new matrix is the row_idx->data[i]-th row in the original matrix
// Output parameter:
//   mat : H2P_dense_mat structure with selected row
void H2P_dense_mat_select_rows(H2P_dense_mat_t mat, H2P_int_vec_t row_idx);

// Transpose a dense matrix
// Input parameter:
//   mat : H2P_dense_mat structure to be transposed
// Output parameter:
//   mat : Transposed H2P_dense_mat structure
void H2P_dense_mat_transpose(H2P_dense_mat_t mat);

// Print a H2P_dense_mat structure, for debugging
// Input parameter:
//   mat : H2P_dense_mat structure to be printed
void H2P_dense_mat_print(H2P_dense_mat_t mat);

// ------------------------------------------------------------------------------ // 


#ifdef __cplusplus
}
#endif

#endif
