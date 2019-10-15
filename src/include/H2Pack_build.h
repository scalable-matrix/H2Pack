#ifndef __H2PACK_BUILD_H__
#define __H2PACK_BUILD_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Generate proxy points for constructing H2 projection and skeleton matrices
// using ID compress for any kernel function. 
// This function is isolated because if the enclosing box for all points are fixed,
// we only need to generate proxy points once and use them repeatedly.
// Input parameter:
//   pt_dim      : Dimension of point coordinate
//   krnl_dim    : Dimension of tensor kernel's return
//   max_level   : Maximum level (included) of a H2 tree, (root level == 0)
//   start_level : Minimum level that needs proxy points
//   max_L       : The size of the root node's enclosing box
//   krnl_eval   : Pointer to kernel matrix evaluation function
// Output parameter:
//   pp_  : Array of proxy points for each level
void H2P_generate_proxy_point_ID(
    const int pt_dim, const int krnl_dim, const int max_level, const int start_level,
    DTYPE max_L, kernel_eval_fptr krnl_eval, H2P_dense_mat_t **pp_
);

// Generate uniformly distributed proxy points on a box surface for constructing
// H2 projection and skeleton matrices for SOME kernel function.
// This function is isolated because if the enclosing box for all points are fixed,
// we only need to generate proxy points once and use them repeatedly.
// Input parameter:
//   pt_dim      : Dimension of point coordinate
//   min_npts    : Minimum number of proxy points on the box surface
//   max_level   : Maximum level (included) of a H2 tree, (root level == 0)
//   start_level : Minimum level that needs proxy points
//   max_L       : The size of the root node's enclosing box
//   krnl_eval   : Pointer to kernel matrix evaluation function
// Output parameter:
//   pp_  : Array of proxy points for each level
void H2P_generate_proxy_point_surface(
    const int pt_dim, const int min_npts, const int max_level, const int start_level,
    DTYPE max_L, kernel_eval_fptr krnl_eval, H2P_dense_mat_t **pp_
);

// Partition work units into multiple blocks s.t. each block has 
// approximately the same amount of work
// Input parameters:
//   n_work     : Number of work units
//   work_sizes : Work size of each work unit
//   total_size : Sum of work_sizes
//   n_block    : Number of blocks to be partitioned, the final result
//                may have fewer blocks
// Output parameter:
//   blk_displs : Indices of each block's first work unit
void H2P_partition_workload(
    const int n_work,  const size_t *work_sizes, const size_t total_size, 
    const int n_block, H2P_int_vec_t blk_displs
);

// Build H2 representation with a kernel function
// Input parameter:
//   h2pack            : H2Pack structure with point partitioning info
//   pp                : Array of proxy points for each level
//   BD_JIT            : 0 or 1, if B and D matrices are computed just-in-time in matvec
//   krnl_eval         : Pointer to kernel matrix evaluation function
//   krnl_symmv        : Pointer to kernel matrix symmetric matvec function
//   krnl_matvec_flops : FLOPs needed in symmetric kernel matvec
// Output parameter:
//   h2pack : H2Pack structure with H2 representation matrices
void H2P_build(
    H2Pack_t h2pack, H2P_dense_mat_t *pp, const int BD_JIT, 
    kernel_eval_fptr krnl_eval, kernel_symmv_fptr krnl_symmv, 
    const int krnl_matvec_flops
);

#ifdef __cplusplus
}
#endif

#endif
