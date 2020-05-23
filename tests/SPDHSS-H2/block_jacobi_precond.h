#ifndef __BLOCK_JACOBI_PRECOND_H__
#define __BLOCK_JACOBI_PRECOND_H__

#include "H2Pack.h"

struct block_jacobi_precond
{
    int    mat_size;        // Size of the matrix to be preconditioned
    int    n_block;         // Number of blocks to use
    int    *blk_sizes;      // Size n_block, size of each block
    int    *blk_displs;     // Size n_block+1, start row & column of each block
    size_t *blk_inv_ptr;    // Size n_block, offset of the inverse of each block
    DTYPE  *blk_inv;        // Size unknown, inverse of each block
};
typedef struct block_jacobi_precond  block_jacobi_precond_s;
typedef struct block_jacobi_precond* block_jacobi_precond_t;

// Construct a block_jacobi_precond from a H2Pack structure
// Input parameters:
//   h2pack : Constructed H2Pack structure
//   shift  : Diagonal shifting of the target matrix
// Output parameter:
//   *precond_ : Constructed block_jacobi_precond structure
void H2P_build_block_jacobi_precond(H2Pack_t h2pack, const DTYPE shift, block_jacobi_precond_t *precond_);

// Apply block Jacobi preconditioner, x := M_{BJP}^{-1} * b
// Input parameters:
//   precond : Constructed block_jacobi_precond structure
//   b       : Size precond->mat_size, input vector
// Output parameter:
//   x : Size precond->mat_size, output vector
void apply_block_jacobi_precond(block_jacobi_precond_t precond, const DTYPE *b, DTYPE *x);

// Destroy a block_jacobi_precond structure
// Input parameter:
//   precond : A block_jacobi_precond structure to be destroyed
void free_block_jacobi_precond(block_jacobi_precond_t precond);

#endif

