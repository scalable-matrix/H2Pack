#ifndef __H2PACK_GEN_PROXY_POINT_H__
#define __H2PACK_GEN_PROXY_POINT_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Generate proxy points for constructing H2 projection and skeleton matrices
// using ID compress for any kernel function. 
// This function is isolated because if the enclosing box for all points are fixed,
// we only need to generate proxy points once and use them repeatedly.
// Input parameters:
//   pt_dim     : Dimension of point coordinate
//   krnl_dim   : Dimension of kernel's return
//   reltol     : Proxy point selection relative error tolerance
//   max_level  : Maximum level (included) of a H2 tree, (root level == 0)
//   min_level  : Minimum level that needs proxy points
//   max_L      : The size of the root node's enclosing box
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   krnl_param : Pointer to kernel function parameter array
// Output parameter:
//   pp_  : Array of proxy points for each level
void H2P_generate_proxy_point_ID(
    const int pt_dim, const int krnl_dim, const DTYPE reltol, const int max_level, const int min_level,
    DTYPE max_L, const void *krnl_param, kernel_eval_fptr krnl_eval, H2P_dense_mat_t **pp_
);

// Generate uniformly distributed proxy points on a box surface for constructing
// H2 projection and skeleton matrices for SOME kernel function.
// This function is isolated because if the enclosing box for all points are fixed,
// we only need to generate proxy points once and use them repeatedly.
// Input parameters:
//   pt_dim     : Dimension of point coordinate
//   xpt_dim    : Dimension of extended point coordinate (for RPY xpt_dim == pt_dim+1, otherwise set xpt_dim == pt_dim)
//   min_npt    : Minimum number of proxy points on the box surface
//   max_level  : Maximum level (included) of a H2 tree, (root level == 0)
//   min_level  : Minimum level that needs proxy points
//   max_L      : The size of the root node's enclosing box
// Output parameter:
//   pp_  : Array of proxy points for each level
void H2P_generate_proxy_point_surface(
    const int pt_dim, const int xpt_dim, const int min_npt, const int max_level, 
    const int min_level, DTYPE max_L, H2P_dense_mat_t **pp_
);

#ifdef __cplusplus
}
#endif

#endif
