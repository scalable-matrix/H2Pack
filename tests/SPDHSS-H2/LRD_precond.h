#ifndef __LRD_PRECOND_H__
#define __LRD_PRECOND_H__

#include "H2Pack.h"

struct LRD_precond
{
    int   mat_size; // Size of the matrix to be preconditioned
    int   rank;     // Rank of the low-rank decomposition
    DTYPE shift;    // Diagonal shift
    DTYPE *Ut;      // Size rank * mat_size, LRD matrix
    DTYPE *workbuf; // Size rank, working buffer in apply_LRD_precond
};
typedef struct LRD_precond  LRD_precond_s;
typedef struct LRD_precond* LRD_precond_t;

#ifdef __cplusplus
extern "C" {
#endif

// Construct a LRD_precond from a H2Pack structure using Nystrom method with random sampling
// Input parameters:
//   h2pack : Constructed H2Pack structure
//   rank   : Rank of the low-rank decomposition
//   shift  : Diagonal shifting of the target matrix
// Output parameter:
//   *precond_ : Constructed LRD_precond structure
void H2P_build_LRD_precond(H2Pack_t h2pack, const int rank, const DTYPE shift, LRD_precond_t *precond_);

// Apply LRD preconditioner, x := M_{LRD}^{-1} * b
// Input parameters:
//   precond : Constructed LRD_precond structure
//   b       : Size precond->mat_size, input vector
// Output parameter:
//   x : Size precond->mat_size, output vector
void apply_LRD_precond(LRD_precond_t precond, const DTYPE *b, DTYPE *x);

// Destroy a LRD_precond structure
// Input parameter:
//   precond : A LRD_precond structure to be destroyed
void free_LRD_precond(LRD_precond_t precond);

#ifdef __cplusplus
}
#endif

#endif
