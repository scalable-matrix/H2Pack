#ifndef __H2PACK_BUILD_H__
#define __H2PACK_BUILD_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build H2 representation with a kernel function
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with H2 representation matrices
void H2P_build(H2Pack_t h2pack, kernel_func_ptr kernel);

#ifdef __cplusplus
}
#endif

#endif
