#ifndef __H2PACK_PARTITION_H__
#define __H2PACK_PARTITION_H__

#include "H2Pack_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Partition points for a H2 tree
// Input parameters:
//   n_point         : Number of points for the kernel matrix
//   coord           : Size n_point * dim, coordinates of points
//   max_leaf_points : Maximum point in a leaf node's box
//   max_leaf_size   : Maximum size of a leaf node's box
//   h2pack          : H2Pack structure initialized using H2P_init()
// Output parameters:
//   h2pack  : H2Pack structure with point partitioning info
void H2P_partitionPoints(
    H2Pack_t h2pack, const int n_point, const DTYPE *coord,
    const int max_leaf_points, const DTYPE max_leaf_size
);

#ifdef __cplusplus
}
#endif

#endif
