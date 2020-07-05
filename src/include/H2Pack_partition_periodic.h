#ifndef __H2PACK_PARTITION_PERIODIC_H__
#define __H2PACK_PARTITION_PERIODIC_H__

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hierarchical point partitioning for periodic system H2 construction 
// Input parameters:
//   h2pack          : H2Pack structure initialized using H2P_init()
//   n_point         : Number of points for the kernel matrix
//   coord           : Matrix, size h2pack->pt_dim * n_point, each column is a point coordinate
//   max_leaf_points : Maximum point in a leaf node's box. If <= 0, will use 200 for
//                     2D points and 400 for other dimensions
//   max_leaf_size   : Maximum size of a leaf node's box. If == 0, max_leaf_points
//                     will be the only restriction.
//   unit_cell       : Array, size 2 * h2pack->pt_dim, unit cell of the periodic system, 
//                     == the largest enclosing box for all points
// Output parameter:
//   h2pack : H2Pack structure with point partitioning info
void H2P_partition_points_periodic(
    H2Pack_t h2pack, const int n_point, const DTYPE *coord, int max_leaf_points, 
    DTYPE max_leaf_size, DTYPE *unit_cell
);

#ifdef __cplusplus
}
#endif

#endif
