#ifndef __FSAI_PRECOND_H__
#define __FSAI_PRECOND_H__

#include "H2Pack.h"
#include "CSRPlus.h"

struct FSAI_precond
{
    int   mat_size;     // Size of the matrix to be preconditioned
    int  *fwd_pmt;      // Forward permutation index array for input vector
    int   *bwd_pmt;     // Backward permutation index array for output vector
    DTYPE *x0;          // Size mat_size, storing G * b in apply_FSAI_precond()
    DTYPE *pmt_b;       // Size mat_size, storing the input vector after permutation
    DTYPE *pmt_x;       // Size mat_size, storing the output vector before permutation
    CSRP_mat_p G, Gt;   // FSAI constructed matrix and its transpose

    // Statistic info
    int    n_apply;
    double t_apply, t_build;
    double mem_MB;
};
typedef struct FSAI_precond  FSAI_precond_s;
typedef struct FSAI_precond* FSAI_precond_p;

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
void H2P_build_FSAI_precond(H2Pack_p h2pack, const int rank, const DTYPE shift, FSAI_precond_p *precond_);

// Apply FSAI preconditioner, x := M_{FSAI}^{-1} * b
// Input parameters:
//   precond : Constructed FSAI_precond structure
//   b       : Size precond->mat_size, input vector
// Output parameter:
//   x : Size precond->mat_size, output vector
void FSAI_precond_apply(FSAI_precond_p precond, const DTYPE *b, DTYPE *x);

// Destroy a FSAI_precond structure
// Input parameter:
//   *precond_ : Pointer to a FSAI_precond structure to be destroyed
void FSAI_precond_destroy(FSAI_precond_p *precond_);

// Print statistic info of a FSAI_precond structure
// Input parameter:
//   precond : FSAI_precond structure whose statistic info to be printed
void FSAI_precond_print_stat(FSAI_precond_p precond);

#ifdef __cplusplus
}
#endif

#endif
