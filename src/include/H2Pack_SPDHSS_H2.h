#ifndef __H2PACK_SPDHSS_H2_H__
#define __H2PACK_SPDHSS_H2_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Construct an SPD HSS matrix from a H2 matrix
// Input parameters:
//   max_rank : Maximum rank of the HSS matrix
//   reltol   : Relative tolerance in column-pivoted QR
//   shift    : Diagonal shifting
//   h2mat    : Constructed H2 matrix
// Output parameter:
//   *hssmat_ : The constructed SPD HSS matrix, A_{HSS} ~= A_{H2} + shift * I
void H2P_SPDHSS_H2_build(
    const int max_rank, const DTYPE reltol, const DTYPE shift, 
    H2Pack_t h2mat, H2Pack_t *hssmat_
);

#ifdef __cplusplus
}
#endif

#endif

