#ifndef __FSAI_PRECOND_H__
#define __FSAI_PRECOND_H__

#include "H2Pack.h"
#include "CSRPlus.h"

struct FSAI_precond
{
    int        mat_size;    // Size of the matrix to be preconditioned
    DTYPE      *x0;         // Size mat_size, storing G * b in apply_FSAI_precond()
    CSRP_mat_t G, Gt;       // FSAI constructed matrix and its transpose
};
typedef struct FSAI_precond  FSAI_precond_s;
typedef struct FSAI_precond* FSAI_precond_t;

#ifdef __cplusplus
extern "C" {
#endif

// Construct a FSAI_precond from a H2Pack structure
// Input parameters:
//   h2pack : Constructed H2Pack structure
//   rank   : Number of nearest neighbors 
//   shift  : Diagonal shifting of the target matrix
// Output parameter:
//   *precond_ : Constructed FSAI_precond structure
void H2P_build_FSAI_precond(H2Pack_t h2pack, const int rank, const DTYPE shift, FSAI_precond_t *precond_);

// Apply FSAI preconditioner, x := M_{FSAI}^{-1} * b
// Input parameters:
//   precond : Constructed FSAI_precond structure
//   b       : Size precond->mat_size, input vector
// Output parameter:
//   x : Size precond->mat_size, output vector
void apply_FSAI_precond(FSAI_precond_t precond, const DTYPE *b, DTYPE *x);

// Destroy a FSAI_precond structure
// Input parameter:
//   precond : A FSAI_precond structure to be destroyed
void free_FSAI_precond(FSAI_precond_t precond);

#ifdef __cplusplus
}
#endif

#endif
