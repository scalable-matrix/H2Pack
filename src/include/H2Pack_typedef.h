#ifndef __H2PACK_TYPEDEF_H__
#define __H2PACK_TYPEDEF_H__

#include "H2Pack_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Structure of H2 matrix tree node
struct H2TreeNode
{
    int   n_child;     // Number of children nodes 
    int   n_node;      // Number of nodes this sub-tree has
    int   po_idx;      // Post-order traversal index of this node
    int   level;       // Level of this node on the tree (root == 0)
    int   cluster[2];  // The start and end indices of points belong to this node
    void  **children;  // Size 2^dim, all children nodes of this node
    DTYPE *enbox;      // Size 2*dim, box that encloses all points of this node. 
                       // enbox[0 : dim-1] are the smallest corner coordinate,
                       // enbox[dim : 2*dim-1] are the size of this box.
};
typedef struct H2TreeNode* H2TreeNode_t;

// Structure of H2 matrix tree flatten representation
struct H2Pack
{
    // H2 matrix tree flatten representation
    int   dim;              // Dimension of point coordinate
    int   n_point;          // Number of points for the kernel matrix
    int   n_node;           // Number of nodes in this H2 tree
    int   n_leaf_node;      // Number of leaf nodes in this H2 tree
    int   max_child;        // Maximum number of children per node, == 2^dim
    int   max_level;        // Maximum level of this H2 tree
    int   min_adm_level;    // Minimum level of reduced admissible pair
    int   n_r_inadm_pair;   // Number of reduced inadmissible pairs 
    int   n_r_adm_pair;     // Number of reduced admissible pairs 
    DTYPE reltol;           // Relative 2-norm tolerance of H2 approximation
    int   *parent;          // Size n_node, parent index of each node
    int   *children;        // Size n_node * max_child, indices of a node's children nodes
    int   *cluster;         // Size n_node * 2, start and end indices of points belong to each node
    int   *n_child;         // Size n_node, number of children nodes of each node
    int   *node_level;      // Size n_node, level of each node
    int   *level_n_node;    // Size max_level, number of nodes in each level
    int   *level_nodes;     // Size max_level * n_leaf_node, indices of nodes on each level
    int   *leaf_nodes;      // Size n_leaf_node, leaf node indices
    int   *r_inadm_pairs;   // Size unknown, Reduced inadmissible pairs 
    int   *r_adm_pairs;     // Size unknown, Reduced admissible pairs 
    int   *node_adm_list;   // Size n_node * n_node, full admissible node list for each node
    int   *node_adm_cnt;    // Size n_node, number of admissible nodes for each node
    DTYPE *coord;           // Size n_point * dim, sorted point coordinates
    DTYPE *enbox;           // Size n_node * (2*dim), enclosing box data of each node
    
    // Statistic data
    size_t mem_bytes;       // Memory usage in bytes
    double timers[4];       // Partition, get admissible pairs, construct H2, matvec
};
typedef struct H2Pack* H2Pack_t;

// Integer vector, similar to std::vector in C++
struct H2P_int_vector
{
    int capacity;    // Capacity of this vector
    int length;      // Current length of this vector
    int *data;       // Data in this vector
};
typedef struct H2P_int_vector* H2P_int_vector_t;

// Global variables used in functions in H2Pack_partition.c
// Use this structure as a namespace in H2Pack_partition.c 
struct H2P_partition_global_vars
{
    int curr_po_idx;    // Post-order traversal index
    int max_level;      // Maximum level of the H2 tree
    int n_leaf_node;    // Number of leaf nodes
    int curr_leaf_idx;  // Index of this leaf node
    int min_adm_level;  // Minimum level of reduced admissible pair
    H2P_int_vector_t r_inadm_pairs;  // Reduced inadmissible pairs
    H2P_int_vector_t r_adm_pairs;    // Reduced admissible pairs
};

// Initialize a H2TreeNode structure
// Input parameter:
//   dim : Dimension of point coordinate
// Output parameter:
//   node_ : Initialized H2TreeNode structure
void H2P_TreeNode_init(H2TreeNode_t *node_, const int dim);

// Recursively destroy a H2TreeNode node and its children nodes
// Input parameter:
//   node : H2 matrix tree node
void H2P_TreeNode_destroy(H2TreeNode_t node);

// Initialize a H2Pack structure
// Input parameters:
//   dim     : Dimension of point coordinate
//   rel_tol : Relative 2-norm tolerance of H2 approximation
// Output parameter:
//   h2pack_ : Initialized H2Pack structure
void H2P_init(H2Pack_t *h2pack_, const int dim, const DTYPE rel_tol);

// Destroy a H2Pack structure
// Input parameter:
//   h2pack : H2Pack structure to be destroyed
void H2P_destroy(H2Pack_t h2pack);

// Initialize a H2P_int_vector structure
// Input parameter:
//   capacity : Initial capacity of the vector. If (capacity <= 0 || capacity >= 65536),
//              capacity will be set as 128.
// Output parameter:
//   int_vec_ : Initialized H2P_int_vector structure
void H2P_int_vector_init(H2P_int_vector_t *int_vec_, int capacity);

// Destroy a H2P_int_vector structure
// Input parameter:
//   int_vec : H2P_int_vector structure
void H2P_int_vector_destroy(H2P_int_vector_t int_vec);

// Push an integer to the tail of a H2P_int_vector
// Input parameters:
//   int_vec : H2P_int_vector structure
//   value   : Value to be pushed 
// Output parameter:
//   int_vec : H2P_int_vector structure with the pushed value
void H2P_int_vector_push_back(H2P_int_vector_t int_vec, int value);

#ifdef __cplusplus
}
#endif

#endif
