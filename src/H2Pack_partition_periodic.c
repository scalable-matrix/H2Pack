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
#include "H2Pack_partition_periodic.h"
#include "utils.h"

// Calculate reduced (in)admissible pairs of a H2 tree for periodic system
// Input parameters:
//   h2pack     : H2Pack structure with H2 tree partitioning in arrays
//   alpha      : Admissible pair coefficient
//   n0, n1     : Node pair
//   shift      : Size h2pack->pt_dim, Coordinate shift for j node
//   lattice_id : Lattice index corresponding to shift
//   part_vars  : Structure for storing working variables and arrays in point partitioning
// Output parameter:
//   per_partition_vars : H2Pack structure reduced (in)admissible pairs
void H2P_calc_reduced_adm_pairs_per(
    H2Pack_p h2pack, const DTYPE alpha, const int n0, const int n1, 
    const DTYPE *shift, const int lattice_id, H2P_partition_vars_p part_vars
)
{
    int   pt_dim    = h2pack->pt_dim;
    int   max_child = h2pack->max_child;
    int   *children = h2pack->children;
    int   *n_child  = h2pack->n_child;
    DTYPE *enbox    = h2pack->enbox;

    DTYPE shifted_box[8];
    int has_shift = 0;
    for (int i = 0; i < pt_dim; i++)
        if (DABS(shift[i]) > 1e-10) has_shift = 1;
    
    if ((n0 == n1) && (has_shift == 0))
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
            H2P_calc_reduced_adm_pairs_per(h2pack, alpha, child_idx, child_idx, shift, lattice_id, part_vars);
        }
        // (2) Interaction between different children nodes
        for (int i = 0; i < n_child_n0; i++)
        {
            int child_idx_i = child_node[i];
            for (int j = i + 1; j < n_child_n0; j++)
            {
                int child_idx_j = child_node[j];
                H2P_calc_reduced_adm_pairs_per(h2pack, alpha, child_idx_i, child_idx_j, shift, lattice_id, part_vars);
            }
        }
    } else {
        // Interaction between two different nodes
        int n_child_n0 = n_child[n0];
        int n_child_n1 = n_child[n1];
        
        // 1. Admissible pair and the level of both node is larger than 
        //    the minimum level of reduced admissible box pair 
        DTYPE *enbox_n0 = enbox + n0 * pt_dim * 2;
        DTYPE *enbox_n1 = enbox + n1 * pt_dim * 2;
        memcpy(shifted_box, enbox_n1, sizeof(DTYPE) * pt_dim * 2);
        for (int i = 0; i < pt_dim; i++) shifted_box[i] += shift[i];
        if (H2P_check_box_admissible(enbox_n0, shifted_box, pt_dim, alpha))
        {
            H2P_int_vec_push_back(part_vars->r_adm_pairs, n0);
            H2P_int_vec_push_back(part_vars->r_adm_pairs, n1);
            H2P_int_vec_push_back(part_vars->r_adm_pairs, lattice_id);
            if (has_shift == 0)  // Double count here so matvec can discard symmetric property
            {
                H2P_int_vec_push_back(part_vars->r_adm_pairs, n1);
                H2P_int_vec_push_back(part_vars->r_adm_pairs, n0);
                H2P_int_vec_push_back(part_vars->r_adm_pairs, lattice_id);
            }
            return;
        }
        
        // 2. Two inadmissible leaf node
        if ((n_child_n0 == 0) && (n_child_n1 == 0))
        {
            H2P_int_vec_push_back(part_vars->r_inadm_pairs, n0);
            H2P_int_vec_push_back(part_vars->r_inadm_pairs, n1);
            H2P_int_vec_push_back(part_vars->r_inadm_pairs, lattice_id);
            if (has_shift == 0)  // Double count here so matvec can discard symmetric property
            {
                H2P_int_vec_push_back(part_vars->r_inadm_pairs, n1);
                H2P_int_vec_push_back(part_vars->r_inadm_pairs, n0);
                H2P_int_vec_push_back(part_vars->r_inadm_pairs, lattice_id);
            }
            return;
        }
        
        // 3. n0 is leaf node, n1 is non-leaf node: check n0 with n1's children
        if ((n_child_n0 == 0) && (n_child_n1 > 0))
        {
            int *child_n1 = children + n1 * max_child;
            for (int j = 0; j < n_child_n1; j++)
            {
                int n1_child_j = child_n1[j];
                H2P_calc_reduced_adm_pairs_per(h2pack, alpha, n0, n1_child_j, shift, lattice_id, part_vars);
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
                H2P_calc_reduced_adm_pairs_per(h2pack, alpha, n0_child_i, n1, shift, lattice_id, part_vars);
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
                    H2P_calc_reduced_adm_pairs_per(h2pack, alpha, n0_child_i, n1_child_j, shift, lattice_id, part_vars);
                }
            }
        }
    }  // End of "if ((n0 == n1) && (has_shift == 0))"
}

// Partition points for a H2 tree
void H2P_partition_points_periodic(
    H2Pack_p h2pack, const int n_point, const DTYPE *coord, int max_leaf_points, 
    DTYPE max_leaf_size, DTYPE *unit_cell
)
{
    const int pt_dim  = h2pack->pt_dim;
    const int xpt_dim = h2pack->xpt_dim;
    double st, et;
    
    st = get_wtime_sec();

    H2P_partition_vars_p part_vars;
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
        pt_dim, n_point
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
    H2P_tree_node_p root = H2P_bisection_partition_points(
        0, 0, n_point-1, pt_dim, xpt_dim, n_point, 
        max_leaf_size, max_leaf_points, unit_cell, 
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
    part_vars->curr_leaf_idx = 0;
    memset(h2pack->level_n_node,  0, int_max_level_msize);
    memset(h2pack->height_n_node, 0, int_max_level_msize);
    H2P_tree_to_array(root, h2pack);
    h2pack->parent[h2pack->root_idx] = -1;  // Root node doesn't have parent
    H2P_tree_node_destroy(&root);  // We don't need the linked list H2 tree anymore
    
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
    DTYPE *enbox0_width = h2pack->enbox + (h2pack->root_idx * (2 * pt_dim) + pt_dim);
    DTYPE shift[8];
    for (int l = 0; l < h2pack->n_lattice; l++)
    {
        DTYPE *lattice_l = h2pack->per_lattices + l * pt_dim;
        for (int j = 0; j < pt_dim; j++) shift[j] = enbox0_width[j] * lattice_l[j];
        H2P_calc_reduced_adm_pairs_per(h2pack, ALPHA_H2, h2pack->root_idx, h2pack->root_idx, shift, l, part_vars);
    }
    h2pack->min_adm_level = 0;
    
    // 5. Copy reduced (in)admissible pairs from H2P_int_vec to h2pack arrays
    h2pack->n_r_inadm_pair = part_vars->r_inadm_pairs->length / 3;
    h2pack->n_r_adm_pair   = part_vars->r_adm_pairs->length   / 3;
    h2pack->r_inadm_pairs    = (int*)   malloc(sizeof(int)   * h2pack->n_r_inadm_pair * 2);
    h2pack->r_adm_pairs      = (int*)   malloc(sizeof(int)   * h2pack->n_r_adm_pair   * 2);
    h2pack->per_inadm_shifts = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->n_r_inadm_pair * pt_dim);
    h2pack->per_adm_shifts   = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->n_r_adm_pair   * pt_dim);
    ASSERT_PRINTF(
        h2pack->r_inadm_pairs  != NULL && h2pack->r_adm_pairs    != NULL && \
        h2pack->per_adm_shifts != NULL && h2pack->per_adm_shifts != NULL, 
        "Failed to allocate arrays of sizes %d and %d for storing (in)admissible pairs\n",
        h2pack->n_r_inadm_pair * 5, h2pack->n_r_adm_pair * 5
    );
    for (int i = 0; i < h2pack->n_r_inadm_pair; i++)
    {
        h2pack->r_inadm_pairs[2 * i + 0] = part_vars->r_inadm_pairs->data[3 * i + 0];
        h2pack->r_inadm_pairs[2 * i + 1] = part_vars->r_inadm_pairs->data[3 * i + 1];
        int l = part_vars->r_inadm_pairs->data[3 * i + 2];
        DTYPE *lattice_l = h2pack->per_lattices + l * pt_dim;
        DTYPE *per_inadm_shift_i = h2pack->per_inadm_shifts + i * pt_dim;
        for (int j = 0; j < pt_dim; j++) per_inadm_shift_i[j] = enbox0_width[j] * lattice_l[j];
    }
    for (int i = 0; i < h2pack->n_r_adm_pair; i++)
    {
        h2pack->r_adm_pairs[2 * i + 0] = part_vars->r_adm_pairs->data[3 * i + 0];
        h2pack->r_adm_pairs[2 * i + 1] = part_vars->r_adm_pairs->data[3 * i + 1];
        int l = part_vars->r_adm_pairs->data[3 * i + 2];
        DTYPE *lattice_l = h2pack->per_lattices + l * pt_dim;
        DTYPE *per_adm_shift_i = h2pack->per_adm_shifts + i * pt_dim;
        for (int j = 0; j < pt_dim; j++) per_adm_shift_i[j] = enbox0_width[j] * lattice_l[j];
    }

    // 6. Initialize thread-local buffer
    h2pack->tb = (H2P_thread_buf_p*) malloc(sizeof(H2P_thread_buf_p) * h2pack->n_thread);
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
    // Root node does not have a destination 
    DAG_src_ptr[n_node - 1] = n_node - 1;
    DAG_src_ptr[n_node]     = n_node - 1;
    DAG_task_queue_init(n_node, n_node - 1, DAG_src_ptr, DAG_dst_idx, &h2pack->upward_tq);
    free(DAG_src_ptr);
    free(DAG_dst_idx);

    H2P_partition_vars_destroy(&part_vars);

    et = get_wtime_sec();
    h2pack->timers[_PT_TIMER_IDX] = et - st;
}
