#ifndef __H2PACK_BUILD_PERIODIC_H__
#define __H2PACK_BUILD_PERIODIC_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build H2 representation with a regular kernel function and
// a periodic system kernel (Ewald summation) function
// Input parameters:
//   h2pack        : H2Pack structure with point partitioning info
//   pp            : Array of proxy points for each level
//   BD_JIT        : 0 or 1, if B and D matrices are computed just-in-time in matvec
//   krnl_param    : Pointer to kernel function parameter array
//   krnl_eval     : Pointer to kernel matrix evaluation function
//   pkrnl_param   : Pointer to periodic system kernel (Ewald summation) function parameter array
//   pkrnl_eval    : Pointer to periodic system kernel (Ewald summation) matrix evaluation function
//   krnl_mv       : Pointer to kernel matvec function
//   krnl_mv_flops : FLOPs needed in kernel bi-matvec
// Output parameter:
//   h2pack : H2Pack structure with H2 representation matrices
void H2P_build_periodic(
    H2Pack_t h2pack, H2P_dense_mat_t *pp, const int BD_JIT, 
    void *krnl_param,  kernel_eval_fptr krnl_eval, 
    void *pkrnl_param, kernel_eval_fptr pkrnl_eval, 
    kernel_mv_fptr krnl_mv, const int krnl_mv_flops
);

#ifdef __cplusplus
}
#endif

#endif
