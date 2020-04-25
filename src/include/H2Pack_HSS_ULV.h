#ifndef __H2PACK_HSS_ULV_H__
#define __H2PACK_HSS_ULV_H__

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Construct the ULV Cholesky factorization for a HSS matrix
// Input parameters:
//   h2pack : H2Pack structure with constructed HSS representation
//   shift  : Shift coefficient k to make (A + k * I) S.P.D.
// Output parameters:
//   h2pack : H2Pack structure with ULV Cholesky factorization
void H2P_HSS_ULV_Cholesky_factorize(H2Pack_t h2pack, const DTYPE shift);

// Solve the linear system A_{HSS} * x = b using the HSS ULV Cholesky factorization,
// where A_{HSS} = L_{HSS} * L_{HSS}^T.
// Input parameters:
//   h2pack : H2Pack structure with ULV Cholesky factorization
//   op     : Operation type, 1, 2, or 3
//   b      : Size >= h2pack->krnl_mat_size, right-hand side vector
// Output parameter:
//   x : Size >= h2pack->krnl_mat_size, solution vector. 
//       If op == 1, x satisfies L_{HSS}^T * x = b.
//       If op == 2, x satisfies L_{HSS}   * x = b.
//       If op == 3, x satisfies A_{HSS}   * x = b.
void H2P_HSS_ULV_Cholesky_solve(H2Pack_t h2pack, const int op, const DTYPE *b, DTYPE *x);

#ifdef __cplusplus
}
#endif

#endif
