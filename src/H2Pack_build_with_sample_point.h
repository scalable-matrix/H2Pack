#ifndef __H2PACK_BUILD_WITH_SAMPLE_POINT_H__
#define __H2PACK_BUILD_WITH_SAMPLE_POINT_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Select sample points for constructing H2 projection and skeleton matrices 
// This algorithm is based on the MATLAB code provided by the author of the paper
// doi/10.1109/IPDPS47924.2020.00082, but the algorithm is not discussed in the paper
// Input parameters:
//   h2pack      : Initialized H2Pack structure
//   krnl_param  : Pointer to kernel function parameter array
//   krnl_eval   : Pointer to kernel matrix evaluation function
//   tau         : Separation threshold, usually is 0.7
// Output parameter:
//   *sample_points_  : Array of sample points for each node
void H2P_select_sample_point(
    H2Pack_p h2pack, const void *krnl_param, kernel_eval_fptr krnl_eval, 
    const DTYPE tau, H2P_dense_mat_p **sample_points_
);

// Build H2 representation with a kernel function and sample points
// Input parameters:
//   h2pack          : H2Pack structure with point partitioning info
//   sample_pt       : Array of sample points for each node
//   BD_JIT          : 0 or 1, if B and D matrices are computed just-in-time in matvec
//   krnl_param      : Pointer to kernel function parameter array
//   krnl_eval       : Pointer to kernel matrix evaluation function
//   krnl_bimv       : Pointer to kernel matrix bi-matvec function
//   krnl_bimv_flops : FLOPs needed in kernel bi-matvec
// Output parameter:
//   h2pack : H2Pack structure with H2 representation matrices
void H2P_build_with_sample_point(
    H2Pack_p h2pack, H2P_dense_mat_p *sample_pt, const int BD_JIT, void *krnl_param, 
    kernel_eval_fptr krnl_eval, kernel_bimv_fptr krnl_bimv, const int krnl_bimv_flops
);

#ifdef __cplusplus
}
#endif

#endif
