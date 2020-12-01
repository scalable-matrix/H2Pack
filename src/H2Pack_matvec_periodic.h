#ifndef __H2PACK_MATVEC_PERIODIC_H__
#define __H2PACK_MATVEC_PERIODIC_H__

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// H2 representation multiplies a column vector, for periodic system
// Input parameters:
//   h2pack : H2Pack structure with H2 representation matrices
//   x      : Input dense vector
// Output parameter:
//   y : Output dense vector
void H2P_matvec_periodic(H2Pack_p h2pack, const DTYPE *x, DTYPE *y);

#ifdef __cplusplus
}
#endif

#endif
