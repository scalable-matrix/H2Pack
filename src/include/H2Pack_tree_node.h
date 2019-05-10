#ifndef __H2PACK_TREE_NODE_H__
#define __H2PACK_TREE_NODE_H__

#include "H2Pack_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// H2 tree linked list node
struct H2P_tree_node
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

#ifdef __cplusplus
}
#endif

#endif
