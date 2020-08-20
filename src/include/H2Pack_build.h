#ifndef __H2PACK_BUILD_H__
#define __H2PACK_BUILD_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build H2 representation with a kernel function
// Input parameters:
//   h2pack          : H2Pack structure with point partitioning info
//   pp              : Array of proxy points for each level
//   BD_JIT          : 0 or 1, if B and D matrices are computed just-in-time in matvec
//   krnl_param      : Pointer to kernel function parameter array
//   krnl_eval       : Pointer to kernel matrix evaluation function
//   krnl_bimv       : Pointer to kernel matrix bi-matvec function
//   krnl_bimv_flops : FLOPs needed in kernel bi-matvec
// Output parameter:
//   h2pack : H2Pack structure with H2 representation matrices
void H2P_build(
    H2Pack_t h2pack, H2P_dense_mat_t *pp, const int BD_JIT, void *krnl_param, 
    kernel_eval_fptr krnl_eval, kernel_bimv_fptr krnl_bimv, const int krnl_bimv_flops
);

#ifdef __cplusplus
}
#endif

#endif
