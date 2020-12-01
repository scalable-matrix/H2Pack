#ifndef __H2PACK_GEN_PROXY_POINT_H__
#define __H2PACK_GEN_PROXY_POINT_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

void H2P_generate_proxy_point_ID(
    const int pt_dim, const int krnl_dim, const DTYPE reltol, const int max_level, const int min_level,
    DTYPE max_L, const void *krnl_param, kernel_eval_fptr krnl_eval, H2P_dense_mat_p **pp_
);

// Calculate the enclosing box of a given set of points and adjust it if the proxy point file is provided
// Input parameters:
//   pt_dim  : Point dimension
//   n_point : Number of points
//   coord   : Size pt_dim-by-npt, each column is a point coordinate
//   fname   : Proxy point file name, can be NULL
// Output parameter:
//   enbox_ : Box that encloses all points in this node.
//            enbox[0 : pt_dim-1] are the corner with the smallest
//            x/y/z/... coordinates. enbox[pt_dim : 2*pt_dim-1] are 
//            the sizes of this box.
void H2P_calc_enclosing_box(const int pt_dim, const int n_point, const DTYPE *coord, const char *fname, DTYPE **enbox_);

// Write a set of proxy points to a text file
// Input parameters:
//   fname     : File name
//   pt_dim    : Point dimension
//   reltol    : Proxy point selection relative error tolerance
//   L3_nlayer : Y box exterior boundary size factor
//   minL      : Radius of the minimal proxy point set (pp[0])
//   num_pp    : Number of proxy point sets
//   pp        : Proxy point sets. Radius of pp[i] should == 2 * radius of pp[i-1]
void H2P_write_proxy_point_file(
    const char *fname, const int pt_dim, const DTYPE reltol, const int L3_nlayer, 
    const DTYPE minL, const int num_pp, H2P_dense_mat_p *pp
);

// Generate proxy points for constructing H2 projection and skeleton matrices using 
// ID compress, also try to load proxy points from a file and update this file
// Input parameters:
//   h2pack     : Initialized H2Pack structure
//   krnl_param : Pointer to kernel function parameter array
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   fname      : Proxy point file name, if == NULL or cannot find that file, compute all proxy points
// Output parameter:
//   pp_  : Array of proxy points for each level
void H2P_generate_proxy_point_ID_file(
    H2Pack_p h2pack, const void *krnl_param, kernel_eval_fptr krnl_eval, 
    const char *fname, H2P_dense_mat_p **pp_
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
    const int min_level, DTYPE max_L, H2P_dense_mat_p **pp_
);

#ifdef __cplusplus
}
#endif

#endif
