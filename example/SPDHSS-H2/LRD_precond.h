#ifndef __LRD_PRECOND_H__
#define __LRD_PRECOND_H__

#include "H2Pack.h"

struct LRD_precond
{
    int   mat_size; // Size of the matrix to be preconditioned
    int   rank;     // Rank of the low-rank decomposition
    int   *fwd_pmt; // Forward permutation index array for input vector
    int   *bwd_pmt; // Backward permutation index array for output vector
    DTYPE shift;    // Diagonal shift
    DTYPE *Ut;      // Size rank * mat_size, LRD matrix
    DTYPE *pmt_b;   // Size mat_size, storing the input vector after permutation
    DTYPE *pmt_x;   // Size mat_size, storing the output vector before permutation
    DTYPE *workbuf; // Size rank, working buffer in apply_LRD_precond

    // Statistic info
    int    n_apply;
    double t_apply, t_build;
    double mem_MB;
};
typedef struct LRD_precond  LRD_precond_s;
typedef struct LRD_precond* LRD_precond_p;

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
void H2P_build_LRD_precond(H2Pack_p h2pack, const int rank, const DTYPE shift, LRD_precond_p *precond_);

// Apply LRD preconditioner, x := M_{LRD}^{-1} * b
// Input parameters:
//   precond : Constructed LRD_precond structure
//   b       : Size precond->mat_size, input vector
// Output parameter:
//   x : Size precond->mat_size, output vector
void LRD_precond_apply(LRD_precond_p precond, const DTYPE *b, DTYPE *x);

// Destroy a LRD_precond structure
// Input parameter:
//   *precond_ : Pointer to a LRD_precond structure to be destroyed
void LRD_precond_destroy(LRD_precond_p *precond_);

// Print statistic info of a LRD_precond structure
// Input parameter:
//   precond : LRD_precond structure whose statistic info to be printed
void LRD_precond_print_stat(LRD_precond_p precond);

#ifdef __cplusplus
}
#endif

#endif
