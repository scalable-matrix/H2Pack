#ifndef __H2PACK_PARTITION_H__
#define __H2PACK_PARTITION_H__

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hierarchical point partitioning for H2 / HSS construction
// Input parameters:
//   h2pack          : H2Pack structure initialized using H2P_init()
//   n_point         : Number of points for the kernel matrix
//   coord           : Matrix, size h2pack->pt_dim * n_point, each column is a point coordinate
//   max_leaf_points : Maximum point in a leaf node's box. If <= 0, will use 200 for
//                     2D points and 400 for other dimensions
//   max_leaf_size   : Maximum size of a leaf node's box. If == 0, max_leaf_points
//                     will be the only restriction.
// Output parameter:
//   h2pack : H2Pack structure with point partitioning info
void H2P_partition_points(
    H2Pack_p h2pack, const int n_point, const DTYPE *coord, 
    int max_leaf_points, DTYPE max_leaf_size
);

// Calculate reduced (in)admissible pairs for HSS
// Input parameter:
//   h2pack : H2Pack structure after calling H2P_partition_points()
// Output parameter:
//   h2pack : H2Pack structure with reduced (in)admissible pairs for HSS
void H2P_HSS_calc_adm_inadm_pairs(H2Pack_p h2pack);

#ifdef __cplusplus
}
#endif

#endif
