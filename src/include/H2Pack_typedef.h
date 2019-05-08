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
    int   min_far_level;    // Minimum level of reduced far-field box pair
    DTYPE reltol;           // Relative 2-norm tolerance of H2 approximation
    int   *parent;          // Size n_node, parent index of each node
    int   *children;        // Size n_node * max_child, indices of a node's children nodes
    int   *cluster;         // Size n_node * 2, start and end indices of points belong to each node
    int   *n_child;         // Size n_node, number of children nodes of each node
    int   *node_level;      // Size n_node, level of each node
    int   *level_n_node;    // Size max_level, number of nodes in each level
    int   *level_nodes;     // Size max_level * n_leaf_node, indices of nodes on each level
    int   *coord_idx0;      // Size n_point, index of each point before sorting
    DTYPE *coord;           // Size n_point * dim, sorted point coordinates
    DTYPE *enbox;           // Size n_node * (2*dim), enclosing box data of each node
    
    // Statistic data
    size_t mem_bytes;       // Memory usage in bytes
};
typedef struct H2Pack* H2Pack_t;

// Initialize a H2Pack structure
// Input parameters:
//   dim    : Dimension of point coordinate
//   reltol : Relative 2-norm tolerance of H2 approximation
// Output parameters:
//   h2pack_ : Initialized H2Pack structure
void H2P_init(H2Pack_t *h2pack_, const int dim, const DTYPE rel_tol);

// Destroy a H2Pack structure
// Input parameter:
//   h2pack : H2Pack structure to be destroyed
void H2P_destroy(H2Pack_t h2pack);

#ifdef __cplusplus
}
#endif

#endif
