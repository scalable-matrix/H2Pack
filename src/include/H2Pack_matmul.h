#ifndef __H2PACK_MATMUL_H__
#define __H2PACK_MATMUL_H__

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// H2 representation multiplies a dense general matrix
// Input parameters:
//   h2pack : H2Pack structure with H2 representation matrices
//   layout : CblasRowMajor/CblasColMajor if x & y are stored in row/column-major style
//   n_vec  : Number of column vectors in mat_x
//   mat_x  : Size >= h2pack->krnl_mat_size * ldx if layout == CblasRowMajor, 
//            size >=                 n_vec * ldx if layout == CblasColMajor, 
//            input dense matrix, the leading h2pack->krnl_mat_size-by-n_vec part of 
//            mat_x will be used
//   ldx    : Leading dimension of mat_x, must >= n_vec if layout == CblasRowMajor,
//            must >= h2pack->krnl_mat_size if layout == CblasColMajor
//   ldy    : Leading dimension of mat_y, the same requirement of ldx
// Output parameter:
//   mat_y  : Size is the same as mat_x, output dense matrix, mat_y := A_{H2} * mat_x
void H2P_matmul(
    H2Pack_t h2pack, const CBLAS_LAYOUT layout, const int n_vec, 
    const DTYPE *mat_x, const int ldx, DTYPE *mat_y, const int ldy
);

// Permute rows of the multiplicand matrix from the original point ordering to 
// the sorted point ordering inside H2Pack (forward), or vise versa (backward)
// for the output matrix. 
// These two functions will be called automatically in H2P_matmul(), you 
// don't need to manually call them. We just provide the interface here.
//   h2pack : H2Pack structure with H2 representation matrices
//   layout : CblasRowMajor/CblasColMajor if x & y are stored in row/column-major style
//   n_vec  : Number of column vectors in mat_x
//   mat_x  : Size >= h2pack->krnl_mat_size * ldx if layout == CblasRowMajor, 
//            size >=                 n_vec * ldx if layout == CblasColMajor, 
//            dense matrix to be permuted, the leading h2pack->krnl_mat_size-by-n_vec 
//            part of mat_x will be used
//   ldx    : Leading dimension of mat_x, must >= n_vec if layout == CblasRowMajor,
//            must >= h2pack->krnl_mat_size if layout == CblasColMajor
//   ldp    : Leading dimension of pmt_mat_x, the same requirement of ldx
// Output parameter:
//   pmt_mat_x : Size is the same as mat_x, permuted dense matrix
void H2P_permute_matrix_row_forward(
    H2Pack_t h2pack, const CBLAS_LAYOUT layout, const int n_vec, 
    const DTYPE *mat_x, const int ldx, DTYPE *pmt_mat_x, const int ldp
);
void H2P_permute_matrix_row_backward(
    H2Pack_t h2pack, const CBLAS_LAYOUT layout, const int n_vec, 
    const DTYPE *mat_x, const int ldx, DTYPE *pmt_mat_x, const int ldp
);

#ifdef __cplusplus
}
#endif

#endif
