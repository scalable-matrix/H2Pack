#ifndef __H2PACK_MATVEC_H__
#define __H2PACK_MATVEC_H__

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// H2 representation multiplies a column vector
// Input parameters:
//   h2pack : H2Pack structure with H2 representation matrices
//   x      : Input dense vector
// Output parameter:
//   y : Output dense vector
void H2P_matvec(H2Pack_p h2pack, const DTYPE *x, DTYPE *y);

// Permute the multiplicand vector from the original point ordering to the 
// sorted point ordering inside H2Pack (forward), or vise versa (backward)
// for the output vector. 
// These two functions will be called automatically in H2P_matvec(), you 
// don't need to manually call them. We just provide the interface here.
// Input parameters:
//   h2pack : H2Pack structure with H2 representation matrices
//   x      : Vector to be permuted
// Output parameter:
//   pmt_x  : Permuted vector
void H2P_permute_vector_forward (H2Pack_p h2pack, const DTYPE *x, DTYPE *pmt_x);
void H2P_permute_vector_backward(H2Pack_p h2pack, const DTYPE *x, DTYPE *pmt_x);

#ifdef __cplusplus
}
#endif

#endif
