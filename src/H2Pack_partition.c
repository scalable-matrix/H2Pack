#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_utils.h"
#include "H2Pack_partition.h"
#include "utils.h"

// Hierarchical partitioning of the given points.
// Tree nodes are indexed in post order.
// Input parameters:
//   level           : Level of current node (root == 0)
//   coord_s         : Index of the first point in this box
//   coord_e         : Index of the last point in this box
//   pt_dim          : Dimension of point coordinate
//   xpt_dim         : Dimension of extended point coordinate (for RPY)
//   n_point         : Total number of points
//   max_leaf_size   : Maximum box size for leaf nodes
//   max_leaf_points : Maximum number of points for leaf nodes
//   enbox           : Box that encloses all points in this node. 
//                     enbox[0 : pt_dim-1] are the corner with the smallest
//                     x/y/z/... coordinates. enbox[pt_dim : 2*pt_dim-1] are  
//                     the sizes of this box.
//   coord           : Array, size n_point * pt_dim, point coordinates.
//   coord_tmp       : Temporary array for sorting coord
//   coord_idx       : Array, size n_point, original index of each point
//   coord_idx_tmp   : Temporary array for sorting coord_idx
//   part_vars       : Structure for storing working variables and arrays in point partitioning
// Output parameters:
//   coord           : Sorted coordinates
//   <return>        : Information of current node
H2P_tree_node_t H2P_bisection_partition_points(
    int level, int coord_s, int coord_e, const int pt_dim, const int xpt_dim, const int n_point, 
    const DTYPE max_leaf_size, const int max_leaf_points, DTYPE *enbox, 
    DTYPE *coord, DTYPE *coord_tmp, int *coord_idx, int *coord_idx_tmp, 
    H2P_partition_vars_t part_vars
)
{
    int node_npts = coord_e - coord_s + 1;
    int max_child = 1 << pt_dim;
    if (level > part_vars->max_level) part_vars->max_level = level;
    
    // 1. Check the enclosing box
    int alloc_enbox = 0;
    if (enbox == NULL)
    {
        alloc_enbox = 1;
        enbox = (DTYPE*) malloc(sizeof(DTYPE) * pt_dim * 2);
        DTYPE *center = (DTYPE*) malloc(sizeof(DTYPE) * pt_dim);
        memset(center, 0, sizeof(DTYPE) * pt_dim);
        // Calculate the center of points in this box
        for (int j = 0; j < pt_dim; j++)
        {
            DTYPE *coord_dim_j = coord + j * n_point;
            for (int i = coord_s; i <= coord_e; i++)
                center[j] += coord_dim_j[i];
        }
        DTYPE semi_box_size = 0.0;
        DTYPE npts = (DTYPE) node_npts;
        for (int j = 0; j < pt_dim; j++) center[j] /= npts;
        // Calculate the box size
        for (int j = 0; j < pt_dim; j++)
        {
            DTYPE *coord_dim_j = coord + j * n_point;
            DTYPE center_j = center[j];
            for (int i = coord_s; i <= coord_e; i++)
            {
                DTYPE tmp = DABS(coord_dim_j[i] - center_j);
                semi_box_size = MAX(semi_box_size, tmp);
            }
        }
        semi_box_size = semi_box_size + 1e-8;
        // Give the center a small random shift to prevent highly symmetric point distributions
        // Use a fixed random seed here to get the same partitioning for repeated tests
        DTYPE shift_scale = semi_box_size * 1e-3;
        srand48(19241112);
        for (int j = 0; j < pt_dim; j++) 
        {
            DTYPE shift_j = (DTYPE) drand48() * 0.5 + 0.25;
            center[j] += shift_j * shift_scale;
        }
        // Recalculate the box size
        semi_box_size = 0.0;
        for (int j = 0; j < pt_dim; j++)
        {
            DTYPE *coord_dim_j = coord + j * n_point;
            DTYPE center_j = center[j];
            for (int i = coord_s; i <= coord_e; i++)
            {
                DTYPE tmp = DABS(coord_dim_j[i] - center_j);
                semi_box_size = MAX(semi_box_size, tmp);
            }
        }
        semi_box_size = semi_box_size + 1e-8;
        for (int j = 0; j < pt_dim; j++)
        {
            enbox[j] = center[j] - semi_box_size - 2e-12;
            enbox[pt_dim + j] = 2 * semi_box_size + 4e-12;
        }
        free(center);
    }  // End of "if (enbox == NULL)"
    DTYPE box_size = enbox[pt_dim];
    
    // 2. If the size of current box or the number of points in current box
    //    is smaller than the threshold, set current box as a leaf node
    if ((node_npts <= max_leaf_points) || (box_size <= max_leaf_size))
    {
        H2P_tree_node_t node;
        H2P_tree_node_init(&node, pt_dim);
        node->pt_cluster[0] = coord_s;
        node->pt_cluster[1] = coord_e;
        node->n_child = 0;
        node->n_node  = 1;
        node->po_idx  = part_vars->curr_po_idx;
        node->level   = level;
        node->height  = 0;
        memcpy(node->enbox, enbox, sizeof(DTYPE) * pt_dim * 2);
        part_vars->curr_po_idx++;
        part_vars->n_leaf_node++;
        if (alloc_enbox) free(enbox);
        return node;
    }
    
    // 3. Bisection partition points in current box
    int *rel_idx   = (int*) malloc(sizeof(int) * node_npts * pt_dim);
    int *child_idx = (int*) malloc(sizeof(int) * node_npts);
    ASSERT_PRINTF(
        rel_idx != NULL && child_idx != NULL, 
        "Failed to allocate index arrays of size %d for bisection partitioning\n", node_npts * (pt_dim + 1)
    );
    memset(child_idx, 0, sizeof(int) * node_npts);
    int pow2 = 1;
    for (int j = 0; j < pt_dim; j++)
    {
        DTYPE enbox_corner_j = enbox[j];
        DTYPE enbox_width_j  = enbox[pt_dim + j];
        DTYPE *coord_dim_j_s = coord   + j * n_point + coord_s;
        int   *rel_idx_dim_j = rel_idx + j * node_npts;
        for (int i = 0; i < node_npts; i++)
        {
            DTYPE rel_coord  = coord_dim_j_s[i] - enbox_corner_j;
            rel_idx_dim_j[i] = DFLOOR(2.0 * rel_coord / enbox_width_j);
            if (rel_idx_dim_j[i] == 2) rel_idx_dim_j[i] = 1;
            child_idx[i] += rel_idx_dim_j[i] * pow2;
        }
        pow2 *= 2;
    }
    
    // 4. Get the number of points in each sub-box, then bucket sort all 
    //    points according to the sub-box a point in
    int *sub_rel_idx   = (int*) malloc(sizeof(int) * max_child * pt_dim);
    int *sub_node_npts = (int*) malloc(sizeof(int) * max_child);
    int *sub_displs    = (int*) malloc(sizeof(int) * (max_child + 1));
    ASSERT_PRINTF(
        sub_rel_idx != NULL && sub_node_npts != NULL && sub_displs != NULL,
        "Failed to allocate working buffer of size %d for sub-nodes\n",
        max_child * (pt_dim + 2)
    );
    memset(sub_node_npts, 0, sizeof(int) * max_child);
    for (int i = 0; i < node_npts; i++)
    {
        int child_idx_i = child_idx[i];
        sub_node_npts[child_idx_i]++;
        for (int j = 0; j < pt_dim; j++)
            sub_rel_idx[j * max_child + child_idx_i] = rel_idx[j * node_npts + i];
    }
    sub_displs[0] = 0;
    for (int i = 1; i <= max_child; i++)
        sub_displs[i] = sub_displs[i - 1] + sub_node_npts[i - 1];
    // Notice: we need to copy both coordinates and extended information
    for (int j = 0; j < xpt_dim; j++)
    {
        int dim_j_offset = j * n_point + coord_s;
        DTYPE *src = coord     + dim_j_offset;
        DTYPE *dst = coord_tmp + dim_j_offset;
        memcpy(dst, src, sizeof(DTYPE) * node_npts);
    }
    memcpy(coord_idx_tmp + coord_s, coord_idx + coord_s, sizeof(int) * node_npts);
    for (int i = 0; i < node_npts; i++)
    {
        int child_idx_i = child_idx[i];
        int src_idx = coord_s + i;
        int dst_idx = coord_s + sub_displs[child_idx_i];
        DTYPE *coord_src = coord_tmp + src_idx;
        DTYPE *coord_dst = coord     + dst_idx;
        // Notice: we need to copy both coordinates and extended information
        for (int j = 0; j < xpt_dim; j++)
            coord_dst[j * n_point] = coord_src[j * n_point];
        coord_idx[dst_idx] = coord_idx_tmp[src_idx];
        sub_displs[child_idx_i]++;
    }
    
    // 5. Prepare enclosing box data for each sub-box
    int n_child = 0;
    DTYPE *sub_box      = (DTYPE*) malloc(sizeof(DTYPE) * max_child * pt_dim * 2);
    int   *sub_coord_se = (int*)   malloc(sizeof(int)   * max_child * 2);
    ASSERT_PRINTF(
        sub_box != NULL && sub_coord_se != NULL,
        "Failed to allocate working buffer of size %d for sub-nodes\n",
        max_child * (pt_dim + 1) * 2
    );
    sub_displs[0] = 0;
    for (int i = 1; i <= max_child; i++)
        sub_displs[i] = sub_displs[i - 1] + sub_node_npts[i - 1];
    for (int i = 0; i < max_child; i++)
    {
        if (sub_node_npts[i] == 0) continue;
        DTYPE *sub_box_child = sub_box + n_child * pt_dim * 2;
        int *sub_rel_idx_i = sub_rel_idx + i;
        for (int j = 0; j < pt_dim; j++)
        {
            sub_box_child[j] = enbox[j] + 0.5 * enbox[pt_dim + j] * sub_rel_idx_i[j * max_child] - 1e-12;
            sub_box_child[pt_dim + j] = 0.5 * enbox[pt_dim + j] + 2e-12;
        }
        sub_coord_se[2 * n_child + 0] = coord_s + sub_displs[i];
        sub_coord_se[2 * n_child + 1] = coord_s + sub_displs[i + 1] - 1;
        n_child++;
    }
    
    // 6. Recursively partition each sub-box
    H2P_tree_node_t node;
    H2P_tree_node_init(&node, pt_dim);
    int n_node = 1, max_child_height = 0;
    for (int i = 0; i < n_child; i++)
    {
        int coord_s_i = sub_coord_se[2 * i + 0];
        int coord_e_i = sub_coord_se[2 * i + 1];
        DTYPE *sub_box_i = sub_box + i * pt_dim * 2;
        node->children[i] = H2P_bisection_partition_points(
            level + 1, coord_s_i, coord_e_i, pt_dim, xpt_dim, n_point, 
            max_leaf_size, max_leaf_points, sub_box_i, 
            coord, coord_tmp, coord_idx, coord_idx_tmp, part_vars
        );
        H2P_tree_node_t child_node_i = (H2P_tree_node_t) node->children[i];
        n_node += child_node_i->n_node;
        max_child_height = MAX(max_child_height, child_node_i->height);
    }
    
    // 7. Store information of this node
    node->pt_cluster[0] = coord_s;
    node->pt_cluster[1] = coord_e;
    node->n_child = n_child;
    node->n_node  = n_node;
    node->po_idx  = part_vars->curr_po_idx;
    node->level   = level;
    node->height  = max_child_height + 1;
    memcpy(node->enbox, enbox, sizeof(DTYPE) * pt_dim * 2);
    part_vars->curr_po_idx++;

    // 8. Free temporary arrays
    free(sub_coord_se);
    free(sub_box);
    free(sub_displs);
    free(sub_node_npts);
    free(sub_rel_idx);
    free(child_idx);
    free(rel_idx);
    if (alloc_enbox) free(enbox);
    
    return node;
}

// Convert a linked list H2 tree to arrays
// Input parameter:
//   node   : Current node of linked list H2 tree
// Output parameter:
//   h2pack : H2Pack structure with H2 tree partitioning in arrays
void H2P_tree_to_array(H2P_tree_node_t node, H2Pack_t h2pack)
{
    int pt_dim    = h2pack->pt_dim;
    int pt_dim2   = pt_dim * 2;
    int max_child = 1 << pt_dim;
    int node_idx  = node->po_idx;
    int n_child   = node->n_child;
    int level     = node->level;
    int height    = node->height;
    
    // 1. Recursively convert sub-trees to arrays
    for (int i = 0; i < node->n_child; i++)
    {
        H2P_tree_node_t child_i = (H2P_tree_node_t) node->children[i];
        H2P_tree_to_array(child_i, h2pack);
    }
    
    // 2. Copy information of current node to arrays
    int *node_children = h2pack->children + node_idx * max_child;
    for (int i = 0; i < n_child; i++)
    {
        H2P_tree_node_t child_i = (H2P_tree_node_t) node->children[i];
        int child_idx = child_i->po_idx;
        node_children[i] = child_idx;
        h2pack->parent[child_idx] = node_idx;
    }
    for (int i = n_child; i < max_child; i++) node_children[i] = -1;
    h2pack->pt_cluster[node_idx * 2 + 0] = node->pt_cluster[0];
    h2pack->pt_cluster[node_idx * 2 + 1] = node->pt_cluster[1];
    memcpy(h2pack->enbox + node_idx * pt_dim2, node->enbox, sizeof(DTYPE) * pt_dim2);
    h2pack->node_level[node_idx]  = level;
    h2pack->node_height[node_idx] = height;
    h2pack->n_child[node_idx] = node->n_child;
    int level_idx  = level  * h2pack->n_leaf_node + h2pack->level_n_node[level];
    int height_idx = height * h2pack->n_leaf_node + h2pack->height_n_node[height];
    h2pack->level_nodes[level_idx]   = node_idx;
    h2pack->height_nodes[height_idx] = node_idx;
    h2pack->level_n_node[level]++;
    h2pack->height_n_node[height]++;
}

// Calculate reduced (in)admissible pairs of a H2 tree
// Input parameters:
//   h2pack    : H2Pack structure with H2 tree partitioning in arrays
//   alpha     : Admissible pair coefficient
//   n0, n1    : Node pair
//   part_vars : Structure for storing working variables and arrays in point partitioning
// Output parameter:
//   h2pack : H2Pack structure reduced (in)admissible pairs
void H2P_calc_reduced_adm_pairs(H2Pack_t h2pack, const DTYPE alpha, const int n0, const int n1, H2P_partition_vars_t part_vars)
{
    int   pt_dim        = h2pack->pt_dim;
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
            H2P_calc_reduced_adm_pairs(h2pack, alpha, child_idx, child_idx, part_vars);
        }
        // (2) Interaction between different children nodes
        for (int i = 0; i < n_child_n0; i++)
        {
            int child_idx_i = child_node[i];
            for (int j = i + 1; j < n_child_n0; j++)
            {
                int child_idx_j = child_node[j];
                H2P_calc_reduced_adm_pairs(h2pack, alpha, child_idx_i, child_idx_j, part_vars);
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
        DTYPE *enbox_n0 = enbox + n0 * pt_dim * 2;
        DTYPE *enbox_n1 = enbox + n1 * pt_dim * 2;
        if (H2P_check_box_admissible(enbox_n0, enbox_n1, pt_dim, alpha) &&
            (level_n0 >= min_adm_level) && (level_n1 >= min_adm_level))
        {
            H2P_int_vec_push_back(part_vars->r_adm_pairs, n0);
            H2P_int_vec_push_back(part_vars->r_adm_pairs, n1);
            int max_level_n01 = MAX(level_n0, level_n1);
            part_vars->min_adm_level = MIN(part_vars->min_adm_level, max_level_n01);
            return;
        }
        
        // 2. Two inadmissible leaf node
        if ((n_child_n0 == 0) && (n_child_n1 == 0))
        {
            H2P_int_vec_push_back(part_vars->r_inadm_pairs, n0);
            H2P_int_vec_push_back(part_vars->r_inadm_pairs, n1);
            return;
        }
        
        // 3. n0 is leaf node, n1 is non-leaf node: check n0 with n1's children
        if ((n_child_n0 == 0) && (n_child_n1 > 0))
        {
            int *child_n1 = children + n1 * max_child;
            for (int j = 0; j < n_child_n1; j++)
            {
                int n1_child_j = child_n1[j];
                H2P_calc_reduced_adm_pairs(h2pack, alpha, n0, n1_child_j, part_vars);
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
                H2P_calc_reduced_adm_pairs(h2pack, alpha, n0_child_i, n1, part_vars);
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
                    H2P_calc_reduced_adm_pairs(h2pack, alpha, n0_child_i, n1_child_j, part_vars);
                }
            }
        }
    }  // End of "if (n0 == n1)"
}

// Calculate the inadmissible node list for each node, required by HSS construction
// Input parameter:
//   h2pack : H2Pack structure with H2 tree partitioning in arrays
// Output parameter:
//   h2pack : H2Pack structure with calculated node inadmissible lists
void H2P_calc_node_inadm_lists(H2Pack_t h2pack)
{
    int   pt_dim        = h2pack->pt_dim;
    int   max_neighbor  = h2pack->max_neighbor;
    int   n_node        = h2pack->n_node;
    int   max_level     = h2pack->max_level;
    int   min_adm_level = h2pack->HSS_min_adm_level;
    int   n_leaf_node   = h2pack->n_leaf_node;
    int   *leaf_nodes   = h2pack->height_nodes;
    int   *node_level   = h2pack->node_level;
    int   *level_n_node = h2pack->level_n_node;
    int   *level_nodes  = h2pack->level_nodes;
    DTYPE *enbox        = h2pack->enbox;

    H2P_int_vec_t target_list;
    H2P_int_vec_init(&target_list, 2 * n_leaf_node);
    int *node_inadm_lists = (int*) malloc(sizeof(int) * n_node * max_neighbor);
    int *node_n_r_inadm   = (int*) malloc(sizeof(int) * n_node);
    ASSERT_PRINTF(
        node_inadm_lists != NULL && node_n_r_inadm != NULL,
        "Failed to allocate node inadmissible list array of size %d", n_node * max_neighbor
    );
    memset(node_n_r_inadm, 0, sizeof(int) * n_node);

    for (int i = max_level; i >= min_adm_level; i--)
    {
        // 1. Prepare target node list of this level
        target_list->length = 0;
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        // (1) All nodes on this level
        for (int j = 0; j < level_n_node[i]; j++)
            H2P_int_vec_push_back(target_list, level_i_nodes[j]);
        // (2) All leaf nodes higher than this level
        for (int j = 0; j < n_leaf_node; j++)
        {
            int leaf_j = leaf_nodes[j];
            if (node_level[leaf_j] < i) 
                H2P_int_vec_push_back(target_list, leaf_j);
        }

        // 2. Find H2 inadmissible nodes in the target node list
        for (int j = 0; j < level_n_node[i]; j++)
        {
            int node = level_i_nodes[j];
            DTYPE *node_enbox = enbox + node * 2 * pt_dim;
            int *node_inadm_list = node_inadm_lists + node * max_neighbor;
            int n_r_inadm = 0;
            for (int k = 0; k < target_list->length; k++)
            {
                int target_k = target_list->data[k];
                DTYPE *target_enbox = enbox + target_k * 2 * pt_dim;
                int is_H2_adm  = H2P_check_box_admissible(node_enbox, target_enbox, pt_dim, ALPHA_H2);
                int is_HSS_adm = H2P_check_box_admissible(node_enbox, target_enbox, pt_dim, ALPHA_HSS);
                if (is_H2_adm == 0 && is_HSS_adm == 1 && target_k != node)
                {
                    node_inadm_list[n_r_inadm] = target_k;
                    n_r_inadm++;
                }
                node_n_r_inadm[node] = n_r_inadm;
            }
        }  // End of j loop
    }  // End of i loop

    h2pack->node_inadm_lists = node_inadm_lists;
    h2pack->node_n_r_inadm   = node_n_r_inadm;
    H2P_int_vec_destroy(target_list);
}

// Calculate reduced (in)admissible pairs for HSS
void H2P_HSS_calc_adm_inadm_pairs(H2Pack_t h2pack)
{
    H2P_partition_vars_t part_vars;
    H2P_partition_vars_init(&part_vars);

    // Calculate reduced HSS (in)admissible pairs
    int H2_min_adm_level = h2pack->min_adm_level;
    h2pack->min_adm_level = 0;
    part_vars->min_adm_level = h2pack->max_level;
    H2P_calc_reduced_adm_pairs(h2pack, ALPHA_HSS, h2pack->root_idx, h2pack->root_idx, part_vars);
    h2pack->min_adm_level     = H2_min_adm_level;
    h2pack->HSS_min_adm_level = part_vars->min_adm_level;
    ASSERT_PRINTF(h2pack->HSS_min_adm_level == 1, "HSS matrix minimal admissible level should be 1!\n");

    // Copy reduced HSS (in)admissible pairs from H2P_int_vec to h2pack arrays
    h2pack->HSS_n_r_inadm_pair = part_vars->r_inadm_pairs->length / 2;
    h2pack->HSS_n_r_adm_pair   = part_vars->r_adm_pairs->length   / 2;
    size_t HSS_r_inadm_pairs_msize = sizeof(int) * h2pack->HSS_n_r_inadm_pair * 2;
    size_t HSS_r_adm_pairs_msize   = sizeof(int) * h2pack->HSS_n_r_adm_pair   * 2;
    h2pack->HSS_r_inadm_pairs = (int*) malloc(HSS_r_inadm_pairs_msize);
    h2pack->HSS_r_adm_pairs   = (int*) malloc(HSS_r_adm_pairs_msize);
    ASSERT_PRINTF(
        h2pack->HSS_r_inadm_pairs != NULL && h2pack->HSS_r_adm_pairs != NULL,
        "Failed to allocate arrays of size %d and %d for HSS (in)admissible pairs\n",
        h2pack->HSS_n_r_inadm_pair * 2, h2pack->HSS_n_r_adm_pair   * 2
    );
    memcpy(h2pack->HSS_r_inadm_pairs, part_vars->r_inadm_pairs->data, HSS_r_inadm_pairs_msize);
    memcpy(h2pack->HSS_r_adm_pairs,   part_vars->r_adm_pairs->data,   HSS_r_adm_pairs_msize);

    // Calculate inadmissible node list for each node
    H2P_calc_node_inadm_lists(h2pack);

    H2P_partition_vars_destroy(part_vars);
}

// Partition points for a H2 tree
void H2P_partition_points(
    H2Pack_t h2pack, const int n_point, const DTYPE *coord, 
    int max_leaf_points, DTYPE max_leaf_size
)
{
    const int pt_dim  = h2pack->pt_dim;
    const int xpt_dim = h2pack->xpt_dim;
    double st, et;
    
    st = get_wtime_sec();

    H2P_partition_vars_t part_vars;
    H2P_partition_vars_init(&part_vars);
    
    // 1. Copy input point coordinates
    h2pack->n_point = n_point;
    if (max_leaf_points <= 0)
    {
        if (pt_dim == 2) max_leaf_points = 200;
        else max_leaf_points = 400;
    }
    h2pack->max_leaf_points = max_leaf_points;
    h2pack->max_leaf_size   = max_leaf_size;
    h2pack->coord_idx       = (int*)   malloc(sizeof(int)   * n_point);
    h2pack->coord           = (DTYPE*) malloc(sizeof(DTYPE) * n_point * xpt_dim);
    ASSERT_PRINTF(
        h2pack->coord != NULL && h2pack->coord_idx != NULL,
        "Failed to allocate matrix of size %d * %d for storing point coordinates\n", 
        xpt_dim, n_point
    );
    memcpy(h2pack->coord, coord, sizeof(DTYPE) * n_point * xpt_dim);
    for (int i = 0; i < n_point; i++) h2pack->coord_idx[i] = i;
    
    // 2. Partition points for H2 tree using linked list 
    int   *coord_idx_tmp = (int*)   malloc(sizeof(int)   * n_point);
    DTYPE *coord_tmp     = (DTYPE*) malloc(sizeof(DTYPE) * n_point * xpt_dim);
    ASSERT_PRINTF(
        coord_tmp != NULL && coord_idx_tmp != NULL,
        "Failed to allocate matrix of size %d * %d for temporarily storing point coordinates\n", 
        pt_dim, n_point
    );
    H2P_tree_node_t root = H2P_bisection_partition_points(
        0, 0, n_point-1, pt_dim, xpt_dim, n_point, 
        max_leaf_size, max_leaf_points, h2pack->root_enbox, 
        h2pack->coord, coord_tmp, h2pack->coord_idx, coord_idx_tmp, part_vars
    );
    free(coord_tmp);
    free(coord_idx_tmp);
    
    // 3. Convert linked list H2 tree partition to arrays
    int n_node    = root->n_node;
    int max_child = 1 << pt_dim;
    int max_level = part_vars->max_level;
    h2pack->n_node        = n_node;
    h2pack->root_idx      = n_node - 1;
    h2pack->n_leaf_node   = part_vars->n_leaf_node;
    h2pack->max_child     = max_child;
    h2pack->max_level     = max_level++;
    size_t int_n_node_msize    = sizeof(int)   * n_node;
    size_t int_max_level_msize = sizeof(int)   * max_level;
    size_t enbox_msize         = sizeof(DTYPE) * n_node * 2 * pt_dim;
    h2pack->parent        = malloc(int_n_node_msize);
    h2pack->children      = malloc(int_n_node_msize * max_child);
    h2pack->pt_cluster    = malloc(int_n_node_msize * 2);
    h2pack->mat_cluster   = malloc(int_n_node_msize * 2);
    h2pack->n_child       = malloc(int_n_node_msize);
    h2pack->node_level    = malloc(int_n_node_msize);
    h2pack->node_height   = malloc(int_n_node_msize);
    h2pack->level_n_node  = malloc(int_max_level_msize);
    h2pack->level_nodes   = malloc(int_max_level_msize * h2pack->n_leaf_node);
    h2pack->height_n_node = malloc(int_max_level_msize);
    h2pack->height_nodes  = malloc(int_max_level_msize * h2pack->n_leaf_node);
    h2pack->enbox         = malloc(enbox_msize);
    ASSERT_PRINTF(h2pack->parent        != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->children      != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->pt_cluster    != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->mat_cluster   != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->n_child       != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->node_level    != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->node_height   != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->level_n_node  != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->level_nodes   != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->height_n_node != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->height_nodes  != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    ASSERT_PRINTF(h2pack->enbox         != NULL, "Failed to allocate hierarchical partitioning tree arrays\n");
    memset(h2pack->level_n_node,  0, int_max_level_msize);
    memset(h2pack->height_n_node, 0, int_max_level_msize);
    H2P_tree_to_array(root, h2pack);
    h2pack->parent[h2pack->root_idx] = -1;  // Root node doesn't have parent
    H2P_tree_node_destroy(root);  // We don't need the linked list H2 tree anymore
    
    // In H2ERI, mat_cluster and krnl_mat_size will be set outside and we don't need xT, yT
    if (h2pack->is_H2ERI == 0)
    {
        for (int i = 0; i < n_node; i++)
        {
            int i20 = i * 2;
            int i21 = i * 2 + 1;
            h2pack->mat_cluster[i20] = h2pack->krnl_dim * h2pack->pt_cluster[i20];
            h2pack->mat_cluster[i21] = h2pack->krnl_dim * (h2pack->pt_cluster[i21] + 1) - 1;
        }
        h2pack->krnl_mat_size = h2pack->krnl_dim * h2pack->n_point;
        h2pack->xT    = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->krnl_mat_size);
        h2pack->yT    = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->krnl_mat_size);
        h2pack->pmt_x = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->krnl_mat_size * h2pack->mm_max_n_vec);
        h2pack->pmt_y = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->krnl_mat_size * h2pack->mm_max_n_vec);
        ASSERT_PRINTF(
            h2pack->xT != NULL && h2pack->yT != NULL && h2pack->pmt_x != NULL && h2pack->pmt_y != NULL,
            "Failed to allocate working arrays of size %d for matvec & matmul\n", 2 * h2pack->krnl_mat_size * (h2pack->mm_max_n_vec+1)
        );
    }
    
    // 4. Calculate reduced (in)admissible pairs
    // h2pack->min_adm_level can be set manually to restrict the minimal admissible level
    // If h2pack->min_adm_level != 0, part_vars->min_adm_level is useless
    h2pack->min_adm_level = 0;
    part_vars->min_adm_level = h2pack->max_level;
    H2P_calc_reduced_adm_pairs(h2pack, ALPHA_H2, h2pack->root_idx, h2pack->root_idx, part_vars);
    h2pack->min_adm_level = part_vars->min_adm_level;
    
    // 5. Copy reduced (in)admissible pairs from H2P_int_vec to h2pack arrays
    h2pack->n_r_inadm_pair = part_vars->r_inadm_pairs->length / 2;
    h2pack->n_r_adm_pair   = part_vars->r_adm_pairs->length   / 2;
    size_t r_inadm_pairs_msize = sizeof(int) * h2pack->n_r_inadm_pair * 2;
    size_t r_adm_pairs_msize   = sizeof(int) * h2pack->n_r_adm_pair   * 2;
    h2pack->r_inadm_pairs = (int*) malloc(r_inadm_pairs_msize);
    h2pack->r_adm_pairs   = (int*) malloc(r_adm_pairs_msize);
    ASSERT_PRINTF(
        h2pack->r_inadm_pairs != NULL && h2pack->r_adm_pairs != NULL,
        "Failed to allocate arrays of sizes %d and %d for storing (in)admissible pairs\n",
        h2pack->n_r_inadm_pair * 2, h2pack->n_r_adm_pair   * 2
    );
    memcpy(h2pack->r_inadm_pairs, part_vars->r_inadm_pairs->data, r_inadm_pairs_msize);
    memcpy(h2pack->r_adm_pairs,   part_vars->r_adm_pairs->data,   r_adm_pairs_msize);

    // 6. Initialize thread-local buffer
    h2pack->tb = (H2P_thread_buf_t*) malloc(sizeof(H2P_thread_buf_t) * h2pack->n_thread);
    ASSERT_PRINTF(h2pack->tb != NULL, "Failed to allocate %d thread buffers\n", h2pack->n_thread);
    for (int i = 0; i < h2pack->n_thread; i++)
        H2P_thread_buf_init(&h2pack->tb[i], h2pack->krnl_mat_size);
    
    // 7. Construct a DAG_task_queue for H2P_build_H2_UJ_proxy 
    int min_adm_level  = h2pack->min_adm_level;
    int *node_level    = h2pack->node_level;
    int *parent        = h2pack->parent;
    int *DAG_src_ptr   = (int*) malloc(sizeof(int) * (n_node + 1));
    int *DAG_dst_idx   = (int*) malloc(sizeof(int) * n_node);
    ASSERT_PRINTF(
        DAG_src_ptr != NULL && DAG_dst_idx != NULL, 
        "Failed to allocate working buffer for DAG task queue construction\n"
    );
    for (int node = 0; node < n_node; node++)
    {
        DAG_src_ptr[node] = node;
        if (node_level[node] < min_adm_level) DAG_dst_idx[node] = node;
        else DAG_dst_idx[node] = parent[node];
    }
    DAG_src_ptr[n_node] = n_node;
    DAG_task_queue_init(n_node, n_node, DAG_src_ptr, DAG_dst_idx, &h2pack->upward_tq);
    free(DAG_src_ptr);
    free(DAG_dst_idx);

    // 8: Optional: calculate reduced (in)admissible pairs for HSS
    if (h2pack->is_HSS == 1) H2P_HSS_calc_adm_inadm_pairs(h2pack);

    H2P_partition_vars_destroy(part_vars);

    et = get_wtime_sec();
    h2pack->timers[_PT_TIMER_IDX] = et - st;
}
