#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"

// Initialize a H2TreeNode structure
void H2P_TreeNode_init(H2TreeNode_t *node_, const int dim)
{
    const int max_child = 1 << dim;
    H2TreeNode_t node = (H2TreeNode_t) malloc(sizeof(struct H2TreeNode));
    assert(node != NULL);
    node->children = (void**) malloc(sizeof(H2TreeNode_t) * max_child);
    node->enbox = (DTYPE*) malloc(sizeof(DTYPE) * dim * 2);
    assert(node->children != NULL && node->enbox != NULL);
    for (int i = 0; i < max_child; i++) 
        node->children[i] = NULL;
    *node_ = node;
}

// Recursively destroy a H2TreeNode node and its children nodes
void H2P_TreeNode_destroy(H2TreeNode_t node)
{
    for (int i = 0; i < node->n_child; i++)
    {
        H2TreeNode_t child_i = (H2TreeNode_t) node->children[i];
        if (child_i != NULL) H2P_TreeNode_destroy(child_i);
        free(child_i);
    }
    free(node->children);
    free(node->enbox);
}

// Initialize a H2Pack structure
void H2P_init(H2Pack_t *h2pack_, const int dim, const DTYPE reltol)
{
    H2Pack_t h2pack = (H2Pack_t) malloc(sizeof(struct H2Pack));
    assert(h2pack != NULL);
    
    h2pack->dim       = dim;
    h2pack->reltol    = reltol;
    h2pack->mem_bytes = 0;
    h2pack->max_child = 1 << dim;
    memset(h2pack->timers, 0, sizeof(double) * 4);
    
    h2pack->parent       = NULL;
    h2pack->children     = NULL;
    h2pack->cluster      = NULL;
    h2pack->n_child      = NULL;
    h2pack->node_level   = NULL;
    h2pack->level_n_node = NULL;
    h2pack->level_nodes  = NULL;
    h2pack->leaf_nodes   = NULL;
    h2pack->r_nf_pairs   = NULL;
    h2pack->r_ff_pairs   = NULL;
    h2pack->coord        = NULL;
    h2pack->enbox        = NULL;
    
    *h2pack_ = h2pack;
}

// Destroy a H2Pack structure
void H2P_destroy(H2Pack_t h2pack)
{
    free(h2pack->parent);
    free(h2pack->children);
    free(h2pack->cluster);
    free(h2pack->n_child);
    free(h2pack->node_level);
    free(h2pack->level_n_node);
    free(h2pack->level_nodes);
    free(h2pack->leaf_nodes);
    free(h2pack->r_nf_pairs);
    free(h2pack->r_ff_pairs);
    free(h2pack->coord);
    free(h2pack->enbox);
}

// Initialize a H2P_int_vector structure
void H2P_int_vector_init(H2P_int_vector_t *int_vec_, int capacity)
{
    if (capacity < 0 || capacity > 65536) capacity = 128;
    H2P_int_vector_t int_vec = (H2P_int_vector_t) malloc(sizeof(struct H2P_int_vector));
    assert(int_vec != NULL);
    int_vec->capacity = capacity;
    int_vec->length = 0;
    int_vec->data = (int*) malloc(sizeof(int) * capacity);
    assert(int_vec->data != NULL);
    *int_vec_ = int_vec;
}

// Destroy a H2P_int_vector structure
void H2P_int_vector_destroy(H2P_int_vector_t int_vec)
{
    free(int_vec->data);
    int_vec->capacity = 0;
    int_vec->length = 0;
}

// Push an integer to the tail of a H2P_int_vector
void H2P_int_vector_push_back(H2P_int_vector_t int_vec, int value)
{
    if (int_vec->capacity == int_vec->length)
    {
        int_vec->capacity *= 2;
        int *new_data = (int*) malloc(sizeof(int) * int_vec->capacity);
        assert(new_data != NULL);
        memcpy(new_data, int_vec->data, sizeof(int) * int_vec->length);
        free(int_vec->data);
        int_vec->data = new_data;
    }
    int_vec->data[int_vec->length] = value;
    int_vec->length++;
}
