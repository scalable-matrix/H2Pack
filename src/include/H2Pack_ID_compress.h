#ifndef __H2PACK_ID_COMPRESS_H__
#define __H2PACK_ID_COMPRESS_H__

#include "H2Pack_config.h"
#include "H2Pack_dense_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

// Partial pivoting QR decomposition, simplified output version
// The partial pivoting QR decomposition is of form:
//     A * P = Q * [R11, R12; 0, R22]
// where R11 is an upper-triangular matrix, R12 and R22 are dense matrices,
// P is a permutation matrix. 
// Input parameters:
//   A        : Target matrix, stored in column major
//   tol_rank : QR stopping parameter, maximum column rank, 
//   tol_norm : QR stopping parameter, maximum column 2-norm
// Output parameters:
//   A  : Matrix R: [R11, R12; 0, R22]
//   p_ : Matrix A column permutation array, A(:, p) = A * P, memory allocated in this function
//   r_ : Dimension of upper-triangular matrix R11
void H2P_partial_pivot_QR(
    H2P_dense_mat_t A, const int tol_rank, const DTYPE tol_norm, 
    int **p_, int *r
);

#ifdef __cplusplus
}
#endif

#endif
