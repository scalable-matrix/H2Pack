#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "H2Pack_typedef.h"
#include "H2Pack_partition.h"

// Hierarchical partitioning of the given points.
// Tree nodes are indexed in post order.
// Input parameters:
//   level           : Level of current node (root == 0)
//   coord_s         : Index of the first point in this box
//   coord_e         : Index of the last point in this box
//   dim             : Dimension of point coordinate
//   max_leaf_size   : Maximum box size for leaf nodes
//   max_leaf_points : Maximum number of points for leaf nodes
//   enbox           : Box that encloses all points in this node. 
//                     enbox[0 : dim-1] are the corner with the smallest
//                     x/y/z/... coordinates. enbox[dim : 2*dim-1] are the 
//                     sizes of this box.
//   coord           : Array, size H2Pack->n_point * dim, point coordinates.
//   coord_tmp       : Temporary array for sorting coord
// Output parameters:
//   coord           : Sorted coordinates
//   <return>        : Information of current node
int curr_po_idx;  // Post-order traversal index
int max_level;    // Maximum level of the H2 tree
int n_leaf_node;  // Number of leaf nodes
H2TreeNode_t H2P_bisection_partition_points(
    int level, int coord_s, int coord_e, const int dim, 
    const DTYPE max_leaf_size, const int max_leaf_points, 
    DTYPE *enbox, DTYPE *coord, DTYPE *coord_tmp
)
{
    int n_point   = coord_e - coord_s + 1;
    int max_child = 1 << dim;
    if (level > max_level) max_level = level;
    
    // 1. Check the enclosing box
    int alloc_enbox = 0;
    if (enbox == NULL)
    {
        alloc_enbox = 1;
        enbox = (DTYPE*) malloc(sizeof(DTYPE) * dim * 2);
        DTYPE *center = (DTYPE*) malloc(sizeof(DTYPE) * dim);
        memset(center, 0, sizeof(DTYPE) * dim);
        // Calculate the center of points in this box
        for (int i = 0; i < n_point; i++)
        {
            DTYPE *coord_i = coord + (coord_s + i) * dim;
            for (int j = 0; j < dim; j++) 
                center[j] += coord_i[j];
        }
        DTYPE semi_box_size = 0.0;
        DTYPE npts = (DTYPE) n_point;
        for (int j = 0; j < dim; j++) center[j] /= npts;
        // Calculate the box size
        for (int i = 0; i < n_point; i++)
        {
            DTYPE *coord_i = coord + (coord_s + i) * dim;
            for (int j = 0; j < dim; j++)
            {
                DTYPE tmp = DABS(coord_i[j] - center[j]);
                semi_box_size = MAX(semi_box_size, tmp);
            }
        }
        for (int j = 0; j < dim; j++)
        {
            enbox[j] = center[j] - semi_box_size;
            enbox[dim + j] = 2 * semi_box_size;
        }
        free(center);
    }
    DTYPE box_size = enbox[dim];
    
    // 2. If the size of current box or the number of points in current box
    //    is smaller than the threshold, set current box as a leaf node
    if ((n_point <= max_leaf_points) || (box_size <= max_leaf_size))
    {
        H2TreeNode_t node;
        H2P_TreeNode_init(&node, dim);
        node->cluster[0]  = coord_s;
        node->cluster[1]  = coord_e;
        node->n_child     = 0;
        node->n_node      = 1;
        node->po_idx      = curr_po_idx;
        node->level       = level;
        memcpy(node->enbox, enbox, sizeof(DTYPE) * dim * 2);
        curr_po_idx++;
        n_leaf_node++;
        if (alloc_enbox) free(enbox);
        return node;
    }
    
    // 3. Bisection partition points in current box
    int *rel_idx   = (int*) malloc(sizeof(int) * n_point * dim);
    int *child_idx = (int*) malloc(sizeof(int) * n_point);
    assert(rel_idx != NULL && child_idx != NULL);
    memset(child_idx, 0, sizeof(int) * n_point);
    for (int i = 0; i < n_point; i++)
    {
        DTYPE *coord_i = coord + (coord_s + i) * dim;
        int *rel_idx_i = rel_idx + i * dim;
        int pow2 = 1;
        for (int j = 0; j < dim; j++)
        {
            DTYPE rel_coord = coord_i[j] - enbox[j];
            rel_idx_i[j] = floor(2 * rel_coord / enbox[dim + j]);
            if (rel_idx_i[j] == 2) rel_idx_i[j] = 1;
            child_idx[i] += rel_idx_i[j] * pow2;
            pow2 *= 2;
        }
    }
    
    // 4. Get the number of points in each sub-box, then bucket sort all 
    //    points according to the sub-box a point in
    int *sub_rel_idx = (int*) malloc(sizeof(int) * max_child * dim);
    int *sub_n_point = (int*) malloc(sizeof(int) * max_child);
    int *sub_displs  = (int*) malloc(sizeof(int) * (max_child + 1));
    assert(sub_rel_idx != NULL && sub_n_point != NULL && sub_displs != NULL);
    memset(sub_n_point, 0, sizeof(int) * max_child);
    for (int i = 0; i < n_point; i++)
    {
        int child_idx_i = child_idx[i];
        sub_n_point[child_idx_i]++;
        memcpy(sub_rel_idx + child_idx_i * dim, rel_idx + i * dim, sizeof(int) * dim);
    }
    H2P_exclusive_scan(max_child, sub_n_point, sub_displs);
    DTYPE *coord_tmp_ptr = coord_tmp + coord_s * dim;
    DTYPE *coord_ptr     = coord     + coord_s * dim;
    memcpy(coord_tmp_ptr, coord_ptr, sizeof(DTYPE) * n_point * dim);
    for (int i = 0; i < n_point; i++)
    {
        int child_idx_i = child_idx[i];
        DTYPE *src_ptr = coord_tmp + dim * (coord_s + i);
        DTYPE *dst_ptr = coord     + dim * (coord_s + sub_displs[child_idx_i]);
        memcpy(dst_ptr, src_ptr, sizeof(DTYPE) * dim);
        sub_displs[child_idx_i]++;
    }
    
    // 5. Prepare enclosing box data for each sub-box
    int n_child = 0;
    DTYPE *sub_box      = (DTYPE*) malloc(sizeof(DTYPE) * max_child * dim * 2);
    int   *sub_coord_se = (int*)   malloc(sizeof(int)   * max_child * 2);
    assert(sub_box != NULL && sub_coord_se != NULL);
    H2P_exclusive_scan(max_child, sub_n_point, sub_displs);
    for (int i = 0; i < max_child; i++)
    {
        if (sub_n_point[i] == 0) continue;
        n_child++;
        DTYPE *sub_box_i = sub_box + i * dim * 2;
        int *sub_rel_idx_i = sub_rel_idx + i * dim;
        for (int j = 0; j < dim; j++)
        {
            sub_box_i[j]       = enbox[j] + 0.5 * enbox[dim + j] * sub_rel_idx_i[j];
            sub_box_i[dim + j] = 0.5 * enbox[dim + j];
        }
        sub_coord_se[2 * i + 0] = coord_s + sub_displs[i];
        sub_coord_se[2 * i + 1] = coord_s + sub_displs[i + 1] - 1;
    }
    
    // 6. Recursively partition each sub-box
    H2TreeNode_t node;
    H2P_TreeNode_init(&node, dim);
    int n_node = 1;
    for (int i = 0; i < n_child; i++)
    {
        int coord_s_i = sub_coord_se[2 * i + 0];
        int coord_e_i = sub_coord_se[2 * i + 1];
        DTYPE *sub_box_i = sub_box + i * dim * 2;
        node->children[i] = H2P_bisection_partition_points(
            level + 1, coord_s_i, coord_e_i, dim, 
            max_leaf_size, max_leaf_points, 
            sub_box_i, coord, coord_tmp
        );
        H2TreeNode_t child_node_i = (H2TreeNode_t) node->children[i];
        n_node += child_node_i->n_node;
    }
    
    // 7. Store information of this node
    node->cluster[0]  = coord_s;
    node->cluster[1]  = coord_e;
    node->n_child     = n_child;
    node->n_node      = n_node;
    node->po_idx      = curr_po_idx;
    node->level       = level;
    memcpy(node->enbox, enbox, sizeof(DTYPE) * dim * 2);
    curr_po_idx++;
    
    // 8. Free temporary arrays
    free(sub_coord_se);
    free(sub_box);
    free(sub_displs);
    free(sub_n_point);
    free(sub_rel_idx);
    free(child_idx);
    free(rel_idx);
    if (alloc_enbox) free(enbox);
    
    return node;
}

// Convert a linked list H2 tree to arrays
// Input parameters:
//   node   : Current node of linked list H2 tree
// Output parameters:
//   h2pack : H2Pack structure
int curr_leaf_idx = 0;  // Index of leaf node
void H2P_tree_to_array(H2TreeNode_t node, H2Pack_t h2pack)
{
    int dim       = h2pack->dim;
    int dimx2     = dim * 2;
    int max_child = 1 << dim;
    int node_idx  = node->po_idx;
    int n_child   = node->n_child;
    int level     = node->level;
    
    // 1. Recursively convert sub-trees to arrays
    for (int i = 0; i < node->n_child; i++)
    {
        H2TreeNode_t child_i = (H2TreeNode_t) node->children[i];
        H2P_tree_to_array(child_i, h2pack);
    }
    
    // 2. Copy information of current node to arrays
    if (n_child == 0)
    {
        h2pack->leaf_nodes[curr_leaf_idx] = node_idx;
        curr_leaf_idx++;
    }
    int *node_children = h2pack->children + node_idx * max_child;
    for (int i = 0; i < n_child; i++)
    {
        H2TreeNode_t child_i = (H2TreeNode_t) node->children[i];
        int child_idx = child_i->po_idx;
        node_children[i] = child_idx;
        h2pack->parent[child_idx] = node_idx;
    }
    for (int i = n_child; i < max_child; i++)
        node_children[i] = -1;
    h2pack->cluster[node_idx * 2 + 0] = node->cluster[0];
    h2pack->cluster[node_idx * 2 + 1] = node->cluster[1];
    memcpy(h2pack->enbox + node_idx * dimx2, node->enbox, sizeof(DTYPE) * dimx2);
    h2pack->node_level[node_idx] = level;
    h2pack->n_child[node_idx] = node->n_child;
    int level_idx = level * h2pack->n_leaf_node + h2pack->level_n_node[level];
    h2pack->level_nodes[level_idx] = node_idx;
    h2pack->level_n_node[level]++;
}

// Partition points for a H2 tree
void H2P_partition_points(
    H2Pack_t h2pack, const int n_point, const DTYPE *coord,
    const int max_leaf_points, const DTYPE max_leaf_size
)
{
    const int dim = h2pack->dim;
    double st = H2P_get_wtime_sec();
    
    // 1. Copy input point coordinates
    h2pack->n_point = n_point;
    h2pack->coord   = (DTYPE*) malloc(sizeof(DTYPE) * n_point * dim);
    assert(h2pack->coord != NULL);
    memcpy(h2pack->coord, coord, sizeof(DTYPE) * n_point * dim);
    h2pack->mem_bytes += sizeof(DTYPE) * n_point * dim;
    
    // 2. Partition points for H2 tree using linked list 
    DTYPE *coord_tmp = (DTYPE*) malloc(sizeof(DTYPE) * n_point * dim);
    assert(coord_tmp != NULL);
    curr_po_idx = 0;
    max_level   = 0;
    n_leaf_node = 0;
    H2TreeNode_t root = H2P_bisection_partition_points(
        0, 0, n_point-1, dim, 
        max_leaf_size, max_leaf_points, 
        NULL, h2pack->coord, coord_tmp
    );
    free(coord_tmp);
    
    // 3. Convert linked list H2 tree partition to arrays
    int n_node    = root->n_node;
    int max_child = 1 << dim;
    h2pack->n_node       = n_node;
    h2pack->n_leaf_node  = n_leaf_node;
    h2pack->max_child    = max_child;
    h2pack->max_level    = max_level++;
    h2pack->parent       = malloc(sizeof(int)   * n_node);
    h2pack->children     = malloc(sizeof(int)   * n_node * max_child);
    h2pack->cluster      = malloc(sizeof(int)   * n_node * 2);
    h2pack->n_child      = malloc(sizeof(int)   * n_node);
    h2pack->node_level   = malloc(sizeof(int)   * n_node);
    h2pack->level_n_node = malloc(sizeof(int)   * max_level);
    h2pack->level_nodes  = malloc(sizeof(int)   * max_level * n_leaf_node);
    h2pack->leaf_nodes   = malloc(sizeof(int)   * n_leaf_node);
    h2pack->enbox        = malloc(sizeof(DTYPE) * n_node * 2 * dim);
    assert(h2pack->parent      != NULL && h2pack->children     != NULL);
    assert(h2pack->cluster     != NULL && h2pack->n_child      != NULL);
    assert(h2pack->node_level  != NULL && h2pack->level_n_node != NULL);
    assert(h2pack->level_nodes != NULL && h2pack->leaf_nodes   != NULL);
    assert(h2pack->enbox       != NULL);
    h2pack->mem_bytes += sizeof(int)   * n_node    * (max_child   + 5);
    h2pack->mem_bytes += sizeof(int)   * max_level * (2 * n_leaf_node + 1);
    h2pack->mem_bytes += sizeof(DTYPE) * n_node    * (dim * 2);
    curr_leaf_idx = 0;
    memset(h2pack->level_n_node, 0, sizeof(int) * max_level);
    H2P_tree_to_array(root, h2pack);
    h2pack->parent[n_node - 1] = -1;  // Root node doesn't have parent
    H2P_TreeNode_destroy(root);  // We don't need the linked list H2 tree anymore
    
    double et = H2P_get_wtime_sec();
    h2pack->timers[0] = et - st;
}
