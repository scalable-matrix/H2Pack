#ifndef __H2PACK_PARTITION_H__
#define __H2PACK_PARTITION_H__

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Partition points for a H2 tree
// Input parameters:
//   n_point         : Number of points for the kernel matrix
//   coord           : Size dim * n_point, each column is the coordinate of a point
//   max_leaf_points : Maximum point in a leaf node's box. If <= 0, will use 200 for
//                     2D points and 400 for other dimensions
//   max_leaf_size   : Maximum size of a leaf node's box. If == 0, max_leaf_points
//                     will be the only restriction.
//   h2pack          : H2Pack structure initialized using H2P_init()
// Output parameters:
//   h2pack : H2Pack structure with point partitioning info
void H2P_partition_points(
    H2Pack_t h2pack, const int n_point, const DTYPE *coord, 
    int max_leaf_points, DTYPE max_leaf_size
);

#ifdef __cplusplus
}
#endif

#endif
