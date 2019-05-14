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
void H2P_matvec(H2Pack_t h2pack, const DTYPE *x, DTYPE *u);

#ifdef __cplusplus
}
#endif

#endif
