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

    // Statistic info
    int    n_apply;
    double t_apply, t_build;
    double mem_MB;
};
typedef struct block_jacobi_precond  block_jacobi_precond_s;
typedef struct block_jacobi_precond* block_jacobi_precond_p;

#ifdef __cplusplus
extern "C" {
#endif

// Construct a block_jacobi_precond from a H2Pack structure
// Input parameters:
//   h2pack : Constructed H2Pack structure
//   shift  : Diagonal shifting of the target matrix
// Output parameter:
//   *precond_ : Constructed block_jacobi_precond structure
void H2P_build_block_jacobi_precond(H2Pack_p h2pack, const DTYPE shift, block_jacobi_precond_p *precond_);

// Apply block Jacobi preconditioner, x := M_{BJP}^{-1} * b
// Input parameters:
//   precond : Constructed block_jacobi_precond structure
//   b       : Size precond->mat_size, input vector
// Output parameter:
//   x : Size precond->mat_size, output vector
void block_jacobi_precond_apply(block_jacobi_precond_p precond, const DTYPE *b, DTYPE *x);

// Destroy a block_jacobi_precond structure
// Input parameter:
//   precond : A block_jacobi_precond structure to be destroyed
void block_jacobi_precond_destroy(block_jacobi_precond_p *precond_);

// Print statistic info of a block_jacobi_precond structure
// Input parameter:
//   precond : block_jacobi_precond structure whose statistic info to be printed
void block_jacobi_precond_print_stat(block_jacobi_precond_p precond);

#ifdef __cplusplus
}
#endif

#endif

