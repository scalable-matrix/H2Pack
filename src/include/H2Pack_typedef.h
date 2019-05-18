#ifndef __H2PACK_TYPEDEF_H__
#define __H2PACK_TYPEDEF_H__

#include "H2Pack_config.h"
#include "H2Pack_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Structure of H2 matrix tree flatten representation
struct H2Pack
{
    // H2 matrix tree flatten representation
    int   dim;              // Dimension of point coordinate
    int   QR_stop_type;     // Partial QR stop criteria
    int   QR_stop_rank;     // Partial QR maximum rank
    int   n_point;          // Number of points for the kernel matrix
    int   max_leaf_points;  // Maximum point in a leaf node's box
    int   n_node;           // Number of nodes in this H2 tree
    int   root_idx;         // Index of the root node (== n_node - 1, save it for convenience)
    int   n_leaf_node;      // Number of leaf nodes in this H2 tree
    int   max_child;        // Maximum number of children per node, == 2^dim
    int   max_level;        // Maximum level of this H2 tree, (root = 0, total max_level + 1 levels)
    int   min_adm_level;    // Minimum level of reduced admissible pair
    int   n_r_inadm_pair;   // Number of reduced inadmissible pairs 
    int   n_r_adm_pair;     // Number of reduced admissible pairs 
    int   n_UJ;             // Number of projection matrices & skeleton row sets
    int   n_B;              // Number of generator matrices
    int   n_D;              // Number of dense blocks
    int   *parent;          // Size n_node, parent index of each node
    int   *children;        // Size n_node * max_child, indices of a node's children nodes
    int   *cluster;         // Size n_node * 2, start and end (included) indices of points belong to each node
    int   *n_child;         // Size n_node, number of children nodes of each node
    int   *node_level;      // Size n_node, level of each node
    int   *level_n_node;    // Size max_level, number of nodes in each level
    int   *level_nodes;     // Size max_level * n_leaf_node, indices of nodes on each level
    int   *leaf_nodes;      // Size n_leaf_node, leaf node indices
    int   *r_inadm_pairs;   // Size unknown, Reduced inadmissible pairs 
    int   *r_adm_pairs;     // Size unknown, Reduced admissible pairs 
    DTYPE max_leaf_size;    // Maximum size of a leaf node's box
    DTYPE QR_stop_tol;      // Partial QR stop column norm tolerance
    DTYPE *coord;           // Size n_point * dim, sorted point coordinates
    DTYPE *enbox;           // Size n_node * (2*dim), enclosing box data of each node
    H2P_dense_mat_t *pp;    // Proxy points on each level for generating U and J
    H2P_dense_mat_t *U;     // Projection matrices
    H2P_int_vec_t   *J;     // Skeleton row sets
    H2P_dense_mat_t *B;     // Generator matrices
    H2P_dense_mat_t *D;     // Dense blocks in the original matrix (from leaf node self interaction & inadmissible pairs)
    H2P_dense_mat_t *y0;    // Temporary arrays used in matvec
    H2P_dense_mat_t *y1;    // Temporary arrays used in matvec
    
    
    // Statistic data
    size_t mem_bytes;       // Memory usage in bytes
    double timers[9];       // Partition; construct pp, U, B, D; matvec up, down, B, D
    int    n_matvec;
    int    mat_size[3];     // Total size of U, D, B
};
typedef struct H2Pack* H2Pack_t;

// Initialize a H2Pack structure
// Input parameters:
//   dim           : Dimension of point coordinate
//   QR_stop_rank  : Partial QR stop criteria: QR_RANK, QR_REL_NRM, or QR_ABS_NRM
//   QR_stop_param : Pointer to partial QR stop parameter
// Output parameter:
//   h2pack_ : Initialized H2Pack structure
void H2P_init(
    H2Pack_t *h2pack_, const int dim, 
    const int QR_stop_type, void *QR_stop_param
);

// Destroy a H2Pack structure
// Input parameter:
//   h2pack : H2Pack structure to be destroyed
void H2P_destroy(H2Pack_t h2pack);


#ifdef __cplusplus
}
#endif

#endif
