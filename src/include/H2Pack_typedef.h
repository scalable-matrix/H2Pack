#ifndef __H2PACK_TYPEDEF_H__
#define __H2PACK_TYPEDEF_H__

#include "H2Pack_utils.h"
#include "H2Pack_config.h"
#include "H2Pack_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pointer to function that evaluates a kernel matrix using given sets of points.
// The kernel function must by stmmetric.
// Input parameters:
//   coord0 : Matrix, size dim-by-ld0, coordinates of the 1st point set
//   ld0    : Leading dimension of coord0, should be >= n0
//   n0     : Number of points in coord0 (each column in coord0 is a coordinate)
//   coord1 : Matrix, size dim-by-ld1, coordinates of the 2nd point set
//   ld1    : Leading dimension of coord1, should be >= n1
//   n1     : Number of points in coord1 (each column in coord0 is a coordinate)
//   ldm    : Leading dimension of the kernel matrix
// Output parameter:
//   mat : Obtained kernel matrix, size n0-by-ld1
typedef void (*kernel_eval_fptr) (
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *mat, const int ldm
);

// Pointer to function that performs two kernel matrix matvec using given sets
// of points and given input vectors. The kernel function must be symmetric.
// This function computes:
//   (1) x_out_0 = kernel_matrix(coord0, coord1) * x_in_0,
//   (2) x_out_1 = kernel_matrix(coord1, coord0) * x_in_1,
//   where kernel_matrix(coord0, coord1)^T = kernel_matrix(coord1, coord0).
// Input parameters:
//   coord0 : Matrix, size dim-by-ld0, coordinates of the 1st point set
//   ld0    : Leading dimension of coord0, should be >= n0
//   n0     : Number of points in coord0 (each column in coord0 is a coordinate)
//   coord1 : Matrix, size dim-by-ld1, coordinates of the 2nd point set
//   ld1    : Leading dimension of coord1, should be >= n1
//   n1     : Number of points in coord1 (each column in coord0 is a coordinate)
//   x_in_0 : Vector, size >= n1, will be left multiplied by kernel_matrix(coord0, coord1)
//   x_in_1 : Vector, size >= n0, will be left multiplied by kernel_matrix(coord1, coord0).
// Output parameter:
//   x_out_0 : Vector, size >= n0, x_out_0 += kernel_matrix(coord0, coord1) * x_in_0
//   x_out_1 : Vector, size >= n1, x_out_1 += kernel_matrix(coord1, coord0) * x_in_1
// Performance optimization notes:
//   When calling a kernel_symmv_fptr, H2Pack guarantees that:
//     (1) n{0,1} are multiples of SIMD_LEN;
//     (2) The lengths of x_{in,out}_{0,1} are multiples of (SIMD_LEN * H2Pack->krnl_dim)
//     (3) The addresses of coord{0,1}, x_{in,out}_{0,1} are aligned to (SIMD_LEN * sizeof(DTYPE))
typedef void (*kernel_symmv_fptr) (
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const DTYPE *x_in_0, const DTYPE *x_in_1, 
    DTYPE *x_out_0, DTYPE *x_out_1
);

// Structure of H2 matrix tree flatten representation
struct H2Pack
{
    // H2 matrix tree flatten representation
    int    n_thread;                // Number of threads
    int    pt_dim;                  // Dimension of point coordinate
    int    krnl_dim;                // Dimension of tensor kernel's return
    int    QR_stop_type;            // Partial QR stop criteria
    int    QR_stop_rank;            // Partial QR maximum rank
    int    n_point;                 // Number of points for the kernel matrix
    int    krnl_mat_size;           // Size of the kernel matrix
    int    krnl_matvec_flops;       // FLOPs needed in symmetric kernel matvec
    int    max_leaf_points;         // Maximum point in a leaf node's box
    int    n_node;                  // Number of nodes in this H2 tree
    int    root_idx;                // Index of the root node (== n_node - 1, save it for convenience)
    int    n_leaf_node;             // Number of leaf nodes in this H2 tree
    int    max_child;               // Maximum number of children per node, == 2^dim
    int    max_level;               // Maximum level of this H2 tree, (root = 0, total max_level + 1 levels)
    int    min_adm_level;           // Minimum level of reduced admissible pair
    int    max_adm_height;          // Maximum height of reduced admissible pair
    int    n_r_inadm_pair;          // Number of reduced inadmissible pairs 
    int    n_r_adm_pair;            // Number of reduced admissible pairs 
    int    n_UJ;                    // Number of projection matrices & skeleton row sets, == n_node
    int    n_B;                     // Number of generator matrices
    int    n_D;                     // Number of dense blocks
    int    BD_JIT;                  // If B and D matrices are computed just-in-time in matvec
    int    is_H2ERI;                // If H2Pack is called from H2ERI
    int    *parent;                 // Size n_node, parent index of each node
    int    *children;               // Size n_node * max_child, indices of a node's children nodes
    int    *pt_cluster;             // Size n_node * 2, start and end (included) indices of points belong to each node
    int    *mat_cluster;            // Size n_node * 2, start and end (included) indices of matvec vector elements belong to each node
    int    *n_child;                // Size n_node, number of children nodes of each node
    int    *node_level;             // Size n_node, level of each node
    int    *node_height;            // Size n_node, height of each node
    int    *level_n_node;           // Size max_level+1, number of nodes in each level
    int    *level_nodes;            // Size (max_level+1) * n_leaf_node, indices of nodes on each level
    int    *height_n_node;          // Size max_level+1, number of nodes of each height
    int    *height_nodes;           // Size (max_level+1) * n_leaf_node, indices of nodes of each height
    int    *r_inadm_pairs;          // Size unknown, reduced inadmissible pairs 
    int    *r_adm_pairs;            // Size unknown, reduced admissible pairs 
    int    *node_n_r_adm;           // Size n_node, number of reduced admissible pairs of a node
    int    *coord_idx;              // Size n_point, original index of each point
    int    *B_nrow;                 // Size n_B, numbers of rows of generator matrices
    int    *B_ncol;                 // Size n_B, numbers of columns of generator matrices
    int    *D_nrow;                 // Size n_D, numbers of rows of dense blocks in the original matrix
    int    *D_ncol;                 // Size n_D, numbers of columns of dense blocks in the original matrix
    size_t *B_ptr;                  // Size n_B, offset of each generator matrix's data in B_data
    size_t *D_ptr;                  // Size n_D, offset of each dense block's data in D_data
    DTYPE  max_leaf_size;           // Maximum size of a leaf node's box
    DTYPE  QR_stop_tol;             // Partial QR stop column norm tolerance
    DTYPE  *coord;                  // Size n_point * dim, sorted point coordinates
    DTYPE  *enbox;                  // Size n_node * (2*dim), enclosing box data of each node
    DTYPE  *B_data;                 // Size unknown, data of generator matrices
    DTYPE  *D_data;                 // Size unknown, data of dense blocks in the original matrix
    DTYPE  *xT;                     // Size krnl_mat_size, use for transpose matvec input  "matrix" when krnl_dim > 1
    DTYPE  *yT;                     // Size krnl_mat_size, use for transpose matevc output "matrix" when krnl_dim > 1
    H2P_int_vec_t     B_blk;        // Size BD_NTASK_THREAD * n_thread, B matrices task partitioning
    H2P_int_vec_t     D_blk0;       // Size BD_NTASK_THREAD * n_thread, diagonal blocks in D matrices task partitioning
    H2P_int_vec_t     D_blk1;       // Size BD_NTASK_THREAD * n_thread, inadmissible blocks in D matrices task partitioning
    H2P_int_vec_t     *J;           // Size n_node, skeleton row sets
    H2P_dense_mat_t   *J_coord;     // Size n_node, Coordinate of J points
    H2P_dense_mat_t   *pp;          // Size max_level + 1, proxy points on each level for generating U and J
    H2P_dense_mat_t   *U;           // Size n_node, Projection matrices
    H2P_dense_mat_t   *y0;          // Size n_node, temporary arrays used in matvec
    H2P_dense_mat_t   *y1;          // Size n_node, temporary arrays used in matvec
    H2P_thread_buf_t  *tb;          // Size n_thread, thread-local buffer
    kernel_eval_fptr  krnl_eval;    // Pointer to kernel matrix evaluation function
    kernel_symmv_fptr krnl_symmv;   // Pointer to kernel matrix symmetric matvec function

    // Statistic data
    int    n_matvec;                // Number of performed matvec
    size_t mat_size[8];             // Total size of U, B, D; matvec memory footprint
    double timers[9];               // Partition; construct U, B, D; matvec up, down, B, D, reduce
    double JIT_flops[2];            // JIT B & D flops
};
typedef struct H2Pack* H2Pack_t;

// Initialize a H2Pack structure
// Input parameters:
//   pt_dim        : Dimension of point coordinate
//   krnl_dim      : Dimension of tensor kernel's return
//   QR_stop_rank  : Partial QR stop criteria: QR_RANK, QR_REL_NRM, or QR_ABS_NRM
//   QR_stop_param : Pointer to partial QR stop parameter
// Output parameter:
//   h2pack_ : Initialized H2Pack structure
void H2P_init(
    H2Pack_t *h2pack_, const int pt_dim, const int krnl_dim, 
    const int QR_stop_type, void *QR_stop_param
);

// Destroy a H2Pack structure
// Input parameter:
//   h2pack : H2Pack structure to be destroyed
void H2P_destroy(H2Pack_t h2pack);

// Print statistic info of a H2Pack structure
// Input parameter:
//   h2pack : H2Pack structure to be destroyed
void H2P_print_statistic(H2Pack_t h2pack);

#ifdef __cplusplus
}
#endif

#endif
