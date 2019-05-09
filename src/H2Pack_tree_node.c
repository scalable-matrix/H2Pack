#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack_config.h"
#include "H2Pack_tree_node.h"

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
