#ifndef __PCG_TESTS_H__
#define __PCG_TESTS_H__

#include "H2Pack.h"

#ifdef __cplusplus
extern "C" {
#endif

// Test preconditioned conjugate gradient solver with different preconditioner
// Input parameters:
//   krnl_mat_size  : Size of the kernel matrix
//   h2mat          : Constructed H2 matrix
//   hssmat         : Constructed SPDHSS matrix
//   shift          : Diagonal shift
//   max_rank       : Maximum approximation rank for LRD and FSAI
//   max_iter       : Maximum number of PCG iterations
//   CG_tol         : Residual vector norm tolerance
void pcg_tests(
    const int krnl_mat_size, H2Pack_t h2mat, H2Pack_t hssmat, 
    const DTYPE shift, const int max_rank, const int max_iter, const DTYPE CG_tol
);

#ifdef __cplusplus
}
#endif

#endif
