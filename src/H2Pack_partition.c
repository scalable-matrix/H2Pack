#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "H2Pack_utils.h"
#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_partition.h"

// Use this structure as a namespace for global variables in this file
struct H2P_partition_vars
{
    int curr_po_idx;    // Post-order traversal index
    int max_level;      // Maximum level of the H2 tree
    int n_leaf_node;    // Number of leaf nodes
    int curr_leaf_idx;  // Index of this leaf node
    int min_adm_level;  // Minimum level of reduced admissible pair
    H2P_int_vec_t r_inadm_pairs;  // Reduced inadmissible pairs
    H2P_int_vec_t r_adm_pairs;    // Reduced admissible pairs
};
struct H2P_partition_vars partition_vars;

// Perform exclusive scan for an integer array
// Input parameters:
//   n : Length of the input array
//   x : Input array
// Output parameters:
//   res : Output array, length n+1
void H2P_exclusive_scan(const int n, int *x, int *res)
{
    res[0] = 0;
    for (int i = 1; i <= n; i++) 
        res[i] = res[i - 1] + x[i - 1];
}

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
H2P_tree_node_t H2P_bisection_partition_points(
    int level, int coord_s, int coord_e, const int dim, 
    const DTYPE max_leaf_size, const int max_leaf_points, 
    DTYPE *enbox, DTYPE *coord, DTYPE *coord_tmp
)
{
    int n_point   = coord_e - coord_s + 1;
    int max_child = 1 << dim;
    if (level > partition_vars.max_level) partition_vars.max_level = level;
    
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
        H2P_tree_node_t node;
        H2P_tree_node_init(&node, dim);
        node->cluster[0]  = coord_s;
        node->cluster[1]  = coord_e;
        node->n_child     = 0;
        node->n_node      = 1;
        node->po_idx      = partition_vars.curr_po_idx;
        node->level       = level;
        memcpy(node->enbox, enbox, sizeof(DTYPE) * dim * 2);
        partition_vars.curr_po_idx++;
        partition_vars.n_leaf_node++;
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
            rel_idx_i[j] = DFLOOR(2 * rel_coord / enbox[dim + j]);
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
    H2P_tree_node_t node;
    H2P_tree_node_init(&node, dim);
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
        H2P_tree_node_t child_node_i = (H2P_tree_node_t) node->children[i];
        n_node += child_node_i->n_node;
    }
    
    // 7. Store information of this node
    node->cluster[0]  = coord_s;
    node->cluster[1]  = coord_e;
    node->n_child     = n_child;
    node->n_node      = n_node;
    node->po_idx      = partition_vars.curr_po_idx;
    node->level       = level;
    memcpy(node->enbox, enbox, sizeof(DTYPE) * dim * 2);
    partition_vars.curr_po_idx++;
    
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
//   h2pack : H2Pack structure with H2 tree partitioning in arrays
void H2P_tree_to_array(H2P_tree_node_t node, H2Pack_t h2pack)
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
        H2P_tree_node_t child_i = (H2P_tree_node_t) node->children[i];
        H2P_tree_to_array(child_i, h2pack);
    }
    
    // 2. Copy information of current node to arrays
    if (n_child == 0)
    {
        h2pack->leaf_nodes[partition_vars.curr_leaf_idx] = node_idx;
        partition_vars.curr_leaf_idx++;
    }
    int *node_children = h2pack->children + node_idx * max_child;
    for (int i = 0; i < n_child; i++)
    {
        H2P_tree_node_t child_i = (H2P_tree_node_t) node->children[i];
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
    partition_vars.curr_po_idx = 0;
    partition_vars.max_level   = 0;
    partition_vars.n_leaf_node = 0;
    H2P_tree_node_t root = H2P_bisection_partition_points(
        0, 0, n_point-1, dim, 
        max_leaf_size, max_leaf_points, 
        NULL, h2pack->coord, coord_tmp
    );
    free(coord_tmp);
    
    // 3. Convert linked list H2 tree partition to arrays
    int n_node    = root->n_node;
    int max_child = 1 << dim;
    int max_level = partition_vars.max_level;
    h2pack->n_node       = n_node;
    h2pack->n_leaf_node  = partition_vars.n_leaf_node;
    h2pack->max_child    = max_child;
    h2pack->max_level    = max_level++;
    h2pack->parent       = malloc(sizeof(int)   * n_node);
    h2pack->children     = malloc(sizeof(int)   * n_node * max_child);
    h2pack->cluster      = malloc(sizeof(int)   * n_node * 2);
    h2pack->n_child      = malloc(sizeof(int)   * n_node);
    h2pack->node_level   = malloc(sizeof(int)   * n_node);
    h2pack->level_n_node = malloc(sizeof(int)   * max_level);
    h2pack->level_nodes  = malloc(sizeof(int)   * max_level * h2pack->n_leaf_node);
    h2pack->leaf_nodes   = malloc(sizeof(int)   * h2pack->n_leaf_node);
    h2pack->enbox        = malloc(sizeof(DTYPE) * n_node * 2 * dim);
    assert(h2pack->parent      != NULL && h2pack->children     != NULL);
    assert(h2pack->cluster     != NULL && h2pack->n_child      != NULL);
    assert(h2pack->node_level  != NULL && h2pack->level_n_node != NULL);
    assert(h2pack->level_nodes != NULL && h2pack->leaf_nodes   != NULL);
    assert(h2pack->enbox       != NULL);
    h2pack->mem_bytes += sizeof(int)   * n_node    * (max_child + 5);
    h2pack->mem_bytes += sizeof(int)   * max_level * (2 * h2pack->n_leaf_node + 1);
    h2pack->mem_bytes += sizeof(DTYPE) * n_node    * (dim * 2);
    partition_vars.curr_leaf_idx = 0;
    memset(h2pack->level_n_node, 0, sizeof(int) * max_level);
    H2P_tree_to_array(root, h2pack);
    h2pack->parent[n_node - 1] = -1;  // Root node doesn't have parent
    H2P_tree_node_destroy(root);  // We don't need the linked list H2 tree anymore
    
    double et = H2P_get_wtime_sec();
    h2pack->timers[0] = et - st;
}

// Check if two boxes are admissible 
// Input parameters:
//   box0, box1 : Box data
//   dim        : Dimension of point coordinate
//   alpha      : Admissible pair coefficient
// Output parameter:
//   <return>   : If two boxes are admissible 
int H2P_check_box_admissible(
    const DTYPE *box0, const DTYPE *box1, 
    const int dim, const DTYPE alpha
)
{
    for (int i = 0; i < dim; i++)
    {
        // Radius of each box's i-th dimension
        DTYPE r0 = box0[dim + i];
        DTYPE r1 = box1[dim + i];
        // Center of each box's i-th dimension
        DTYPE c0 = box0[i] + 0.5 * r0;
        DTYPE c1 = box1[i] + 0.5 * r1;
        DTYPE min_r = MIN(r0, r1);
        DTYPE dist  = DABS(c0 - c1);
        if (dist >= alpha * min_r + 0.5 * (r0 + r1)) return 1;
    }
    return 0;
}

// Calculate reduced (in)admissible pairs of a H2 tree
// Input parameter:
//   h2pack : H2Pack structure with H2 tree partitioning in arrays
//   alpha  : Admissible pair coefficient
//   n0, n1 : Node pair
// Output parameter:
//   h2pack : H2Pack structure reduced (in)admissible pairs
void H2P_calc_reduced_adm_pairs(H2Pack_t h2pack, const DTYPE alpha, const int n0, const int n1)
{
    int   dim           = h2pack->dim;
    int   max_child     = h2pack->max_child;
    int   min_adm_level = h2pack->min_adm_level;
    int   *children     = h2pack->children;
    int   *n_child      = h2pack->n_child;
    int   *node_level   = h2pack->node_level;
    DTYPE *enbox        = h2pack->enbox;
    
    if (n0 == n1)
    {
        // Self box interaction
        
        // 1. Leaf node, nothing to do
        int n_child_n0 = n_child[n0];
        if (n_child_n0 == 0) return;
        
        // 2. Non-leaf node, check each children node
        int *child_node = children + n0 * max_child;
        // (1) Children node self box interaction
        for (int i = 0; i < n_child_n0; i++)
        {
            int child_idx = child_node[i];
            H2P_calc_reduced_adm_pairs(h2pack, alpha, child_idx, child_idx);
        }
        // (2) Interaction between different children nodes
        for (int i = 0; i < n_child_n0; i++)
        {
            int child_idx_i = child_node[i];
            for (int j = i + 1; j < n_child_n0; j++)
            {
                int child_idx_j = child_node[j];
                H2P_calc_reduced_adm_pairs(h2pack, alpha, child_idx_i, child_idx_j);
            }
        }
    } else {
        // Interaction between two different nodes
        int n_child_n0 = n_child[n0];
        int n_child_n1 = n_child[n1];
        int level_n0   = node_level[n0];
        int level_n1   = node_level[n1];
        
        // 1. Admissible pair and the level of both node is larger than 
        //    the minimum level of reduced admissible box pair 
        DTYPE *enbox_n0 = enbox + n0 * dim * 2;
        DTYPE *enbox_n1 = enbox + n1 * dim * 2;
        if (H2P_check_box_admissible(enbox_n0, enbox_n1, dim, alpha) &&
            (level_n0 >= min_adm_level) && (level_n1 >= min_adm_level))
        {
            H2P_int_vec_push_back(partition_vars.r_adm_pairs, n0);
            H2P_int_vec_push_back(partition_vars.r_adm_pairs, n1);
            partition_vars.min_adm_level = MIN(partition_vars.min_adm_level, level_n0);
            partition_vars.min_adm_level = MIN(partition_vars.min_adm_level, level_n1);
            return;
        }
        
        // 2. Two inadmissible leaf node
        if ((n_child_n0 == 0) && (n_child_n1 == 0))
        {
            H2P_int_vec_push_back(partition_vars.r_inadm_pairs, n0);
            H2P_int_vec_push_back(partition_vars.r_inadm_pairs, n1);
            return;
        }
        
        // 3. n0 is leaf node, n1 is non-leaf node: check n0 with n1's children
        if ((n_child_n0 == 0) && (n_child_n1 > 0))
        {
            int *child_n1 = children + n1 * max_child;
            for (int j = 0; j < n_child_n1; j++)
            {
                int n1_child_j = child_n1[j];
                H2P_calc_reduced_adm_pairs(h2pack, alpha, n0, n1_child_j);
            }
            return;
        }
        
        // 4. n0 is non-leaf node, n1 is leaf node: check n1 with n0's children
        if ((n_child_n0 > 0) && (n_child_n1 == 0))
        {
            int *child_n0 = children + n0 * max_child;
            for (int i = 0; i < n_child_n0; i++)
            {
                int n0_child_i = child_n0[i];
                H2P_calc_reduced_adm_pairs(h2pack, alpha, n0_child_i, n1);
            }
            return;
        }
        
        // 5. Neither n0 nor n1 is leaf node, check their children
        if ((n_child_n0 > 0) && (n_child_n1 > 0))
        {
            int *child_n0 = children + n0 * max_child;
            int *child_n1 = children + n1 * max_child;
            for (int i = 0; i < n_child_n0; i++)
            {
                int n0_child_i = child_n0[i];
                for (int j = 0; j < n_child_n1; j++)
                {
                    int n1_child_j = child_n1[j];
                    H2P_calc_reduced_adm_pairs(h2pack, alpha, n0_child_i, n1_child_j);
                }
            }
        }
    }
}

// Calculate full admissible pair list for each node
// For each box i at k-th level, find all admissible nodes j if it satisfies 
// either of the following conditions: 
// 1. j is at k-th level and (i, j) is an admissible pair, which requires that 
//    at least one pair of ancestors of i and j is a far-field pair.
// 2. j is a leaf node, (i, j) is an admissible pair, and j's level is 
//    higher than i.
// 3. i is a leaf node and j's level is lower than i. In this case, (i, j) 
//    is already in the reduced far-field pair list.
// Input parameter:
//   h2pack : H2Pack structure with reduced (in)admissible pairs of a H2 tree
// Output parameter:
//   h2pack : H2Pack structure with full admissible pair list for each node
void H2P_calc_full_adm_lists(H2Pack_t h2pack)
{
    int n_node        = h2pack->n_node;
    int max_child     = h2pack->max_child;
    int *parent       = h2pack->parent;
    int *children     = h2pack->children;
    int *n_child      = h2pack->n_child;
    int n_r_adm_pair  = h2pack->n_r_adm_pair;
    int *r_adm_pairs  = h2pack->r_adm_pairs;
    int *level_n_node = h2pack->level_n_node;
    int *level_nodes  = h2pack->level_nodes;
    
    // 1. Allocate node_adm_list and node_adm_cnt
    h2pack->node_adm_list = (int*) malloc(sizeof(int) * n_node * n_node);
    h2pack->node_adm_cnt  = (int*) malloc(sizeof(int) * n_node);
    assert(h2pack->node_adm_list != NULL && h2pack->node_adm_cnt != NULL);
    memset(h2pack->node_adm_cnt, 0, sizeof(int) * n_node);
    int *node_adm_list = h2pack->node_adm_list;
    int *node_adm_cnt  = h2pack->node_adm_cnt;
    
    // 2. Prepare reduce admissible list for each node
    int *node_r_adm_list = (int*) malloc(sizeof(int) * n_node * n_node);
    int *node_r_adm_cnt  = (int*) malloc(sizeof(int) * n_node);
    assert(node_r_adm_list != NULL && node_r_adm_cnt != NULL);
    memset(node_r_adm_cnt, 0, sizeof(int) * n_node);
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int n0 = r_adm_pairs[2 * i + 0];
        int n1 = r_adm_pairs[2 * i + 1];
        int idx0 = node_r_adm_cnt[n0];
        int idx1 = node_r_adm_cnt[n1];
        node_r_adm_list[n0 * n_node + idx0] = n1;
        node_r_adm_list[n1 * n_node + idx1] = n0;
        node_r_adm_cnt[n0] = idx0 + 1;
        node_r_adm_cnt[n1] = idx1 + 1;
    }
    
    // 3. Generate admissible lists level by level
    for (int i = 1; i <= h2pack->max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * h2pack->n_leaf_node;
        for (int j = 0; j < level_n_node[i]; j++)
        {
            int node = level_i_nodes[j];
            int *adm_list = node_adm_list + node * n_node;
            
            // (1) Admissible nodes inherent from parent
            int parent_idx = parent[node];
            int *parent_adm_list = node_adm_list + parent_idx * n_node;
            for (int k = 0; k < node_adm_cnt[parent_idx]; k++)
            {
                int parent_adm_k = parent_adm_list[k];
                int n_child_k    = n_child[parent_adm_k];
                int *children_k  = children + parent_adm_k * max_child;
                if (n_child_k > 0)
                {
                    // Condition 1
                    int cnt = node_adm_cnt[node];
                    for (int kk = 0; kk < n_child_k; kk++)
                    {
                        adm_list[cnt] = children_k[kk];
                        cnt++;
                    }
                    node_adm_cnt[node] += n_child_k;
                } else {
                    // Condition 2
                    int cnt = node_adm_cnt[node];
                    adm_list[cnt] = parent_adm_k;
                    node_adm_cnt[node]++;
                }
            }
            
            // (2) Condition 3 (?)
            int r_adm_cnt = node_r_adm_cnt[node];
            int *r_adm_list = node_r_adm_list + node * n_node;
            int cnt = node_adm_cnt[node];
            memcpy(adm_list + cnt, r_adm_list, sizeof(int) * r_adm_cnt);
            node_adm_cnt[node] += r_adm_cnt;
        }
    }
    
    free(node_r_adm_cnt);
    free(node_r_adm_list);
}

// Calculate reduced (in)admissible pairs and full admissible pairs of a H2 tree
void H2P_calc_admissible_pairs(H2Pack_t h2pack)
{
    double st = H2P_get_wtime_sec();
    
    // 1. Calculate reduced (in)admissible pairs
    int estimated_n_pair = h2pack->n_node * h2pack->max_child;
    H2P_int_vec_init(&partition_vars.r_inadm_pairs, estimated_n_pair);
    H2P_int_vec_init(&partition_vars.r_adm_pairs,   estimated_n_pair);
    // TODO: Change min_adm_level according to the tree structure
    // If h2pack->min_adm_level != 0, partition_vars.min_adm_level is useless
    h2pack->min_adm_level = 0;
    partition_vars.min_adm_level = h2pack->max_level;
    int root_idx = h2pack->n_node - 1;
    H2P_calc_reduced_adm_pairs(h2pack, ALPHA_H2, root_idx, root_idx);
    if (h2pack->min_adm_level == 0)
        h2pack->min_adm_level = partition_vars.min_adm_level;
    
    // 2. Copy reduced (in)admissible pairs from H2P_int_vec to h2pack arrays
    h2pack->n_r_inadm_pair = partition_vars.r_inadm_pairs->length / 2;
    h2pack->n_r_adm_pair   = partition_vars.r_adm_pairs->length   / 2;
    size_t r_inadm_pair_msize = sizeof(int) * h2pack->n_r_inadm_pair * 2;
    size_t r_adm_pair_msize   = sizeof(int) * h2pack->n_r_adm_pair   * 2;
    h2pack->r_inadm_pairs = (int*) malloc(r_inadm_pair_msize);
    h2pack->r_adm_pairs   = (int*) malloc(r_adm_pair_msize);
    assert(h2pack->r_inadm_pairs != NULL && h2pack->r_adm_pairs != NULL);
    memcpy(h2pack->r_inadm_pairs, partition_vars.r_inadm_pairs->data, r_inadm_pair_msize);
    memcpy(h2pack->r_adm_pairs,   partition_vars.r_adm_pairs->data,   r_adm_pair_msize);
    H2P_int_vec_destroy(partition_vars.r_inadm_pairs);
    H2P_int_vec_destroy(partition_vars.r_adm_pairs);
    
    // 3. Calculate full admissible pair list for each node
    H2P_calc_full_adm_lists(h2pack);
    
    double et = H2P_get_wtime_sec();
    h2pack->timers[1] = et - st;
}
