#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_partition.h"

H2TreeNode_t H2P_partition_bisec(
    int level, int coord_s, int coord_e, const int dim, 
    const DTYPE max_leaf_size, const int max_leaf_points, DTYPE *enbox,
    DTYPE *coord, DTYPE *coord_tmp, int *coord_idx0, 
    int *curr_po_idx, int *max_level, int *n_leaf_node
)
{
    
}

void H2P_tree_to_array(H2TreeNode_t node, H2Pack_t h2pack)
{
    
}

// Destroy a H2 matrix tree
// Input parameter:
//   node : H2 matrix tree node
void H2P_destroy_tree(H2TreeNode_t node)
{
    for (int i = 0; i < node->n_child; i++)
    {
        H2TreeNode_t child_i = (H2TreeNode_t) node->children[i];
        if (child_i != NULL) H2P_destroy_tree(child_i);
        free(child_i);
    }
    free(node->children);
    free(node->enbox);
}

// Partition points for a H2 tree
void H2P_partitionPoints(
    H2Pack_t h2pack, const int n_point, const DTYPE *coord,
    const int max_leaf_points, const DTYPE max_leaf_size
)
{
    const int dim = h2pack->dim;
    
    // 1. Copy input point coordinates
    h2pack->n_point    = n_point;
    h2pack->coord_idx0 = (int*)   malloc(sizeof(int)   * n_point);
    h2pack->coord      = (DTYPE*) malloc(sizeof(DTYPE) * n_point * dim);
    assert(h2pack->coord_idx0 != NULL && h2pack->coord != NULL);
    memcpy(h2pack->coord, coord, sizeof(DTYPE) * n_point * dim);
    for (int i = 0; i < n_point; i++) 
        h2pack->coord_idx0[i] = i;
    h2pack->mem_bytes += sizeof(int)   * n_point;
    h2pack->mem_bytes += sizeof(DTYPE) * n_point * dim;
    
    // 2. Partition points for H2 tree using linked list 
    DTYPE *coord_tmp = (DTYPE*) malloc(sizeof(DTYPE) * n_point * dim);
    assert(coord_tmp != NULL);
    int curr_po_idx = 0;
    int max_level   = 0;
    int n_leaf_node = 0;
    H2TreeNode_t root;
    root = H2P_partition_bisec(
        0, 0, n_point-1, dim, max_leaf_size, max_leaf_points, NULL,
        h2pack->coord, coord_tmp, h2pack->coord_idx0, 
        &curr_po_idx, &max_level, &n_leaf_node
    );
    free(coord_tmp);
    
    // 3. Convert linked list H2 tree partition to arrays
    int n_node    = root->n_node;
    int max_child = 1;
    for (int i = 0; i < dim; i++) max_child *= 2;
    h2pack->n_node       = n_node;
    h2pack->n_leaf_node  = n_leaf_node;
    h2pack->max_child    = max_child;
    h2pack->max_level    = max_level;
    h2pack->parent       = malloc(sizeof(int)   * n_node);
    h2pack->children     = malloc(sizeof(int)   * n_node * max_child);
    h2pack->cluster      = malloc(sizeof(int)   * n_node * 2);
    h2pack->n_child      = malloc(sizeof(int)   * n_node);
    h2pack->node_level   = malloc(sizeof(int)   * n_node);
    h2pack->level_n_node = malloc(sizeof(int)   * max_level);
    h2pack->level_nodes  = malloc(sizeof(int)   * max_level * n_leaf_node);
    h2pack->enbox        = malloc(sizeof(DTYPE) * n_node * 2 * dim);
    assert(h2pack->parent      != NULL && h2pack->children     != NULL);
    assert(h2pack->cluster     != NULL && h2pack->n_child      != NULL);
    assert(h2pack->node_level  != NULL && h2pack->level_n_node != NULL);
    assert(h2pack->level_nodes != NULL && h2pack->enbox        != NULL);
    h2pack->mem_bytes += sizeof(int)   * n_node    * (max_child   + 5);
    h2pack->mem_bytes += sizeof(int)   * max_level * (n_leaf_node + 1);
    h2pack->mem_bytes += sizeof(DTYPE) * n_node    * (dim * 2);
    H2P_tree_to_array(root, h2pack);
    H2P_destroy_tree(root);
}
