#ifndef __H2PACK_ID_COMPRESS_H__
#define __H2PACK_ID_COMPRESS_H__

#include "H2Pack_config.h"
#include "H2Pack_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Interpolative Decomposition (ID) using partial QR over rows of a target 
// matrix. Partial pivoting QR may need to be upgraded to SRRQR later. 
// Given an m*n matrix A, an rank-k ID approximation of A is of form
//         A = U * A(J, :)
// where J is a row index subset of size k, and U is a m*k matrix (if 
// SRRQR is used, entries of U are bounded by a parameter 'f'). A(J,:) 
// and U are usually called the skeleton and projection matrix. 
// Input parameters:
//   A          : Target matrix, stored in column major
//   stop_type  : Partial QR stop criteria: QR_RANK, QR_REL_NRM, or QR_ABS_NRM
//   stop_param : Pointer to partial QR stop parameter
//   nthreads   : Number of threads used in this function
//   QR_buff    : Size 2 * A->nrow, working buffer for partial pivoting QR
//   ID_buff    : Size 4 * A->nrow, working buffer for ID compression
// Output parameters:
//   U_ : Projection matrix, will be initialized in this function. If U_ == NULL,
//        the projection matrix will not be calculated.
//   J  : Row indices of the skeleton A
void H2P_ID_compress(
    H2P_dense_mat_t A, const int stop_type, void *stop_param, H2P_dense_mat_t *U_, 
    H2P_int_vec_t J, const int nthreads, DTYPE *QR_buff, int *ID_buff
);

#ifdef __cplusplus
}
#endif

#endif
