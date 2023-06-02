#ifndef __H2PACK_CONFIG_H__
#define __H2PACK_CONFIG_H__

// Parameters used in H2Pack

#define DOUBLE_SIZE     8
#define FLOAT_SIZE      4

#ifndef DTYPE_SIZE
#define DTYPE_SIZE      DOUBLE_SIZE     // Matrix data type: double or float
#endif

#if DTYPE_SIZE == DOUBLE_SIZE           // Marcos for double data type
#define DTYPE           double          // Data type
#define DTYPE_FMTSTR    "%lf"           // Data type format string
#define DABS            fabs            // Abs function
#define DLOG            log             // Natural logarithm function
#define DLOG2           log2            // Base-2 logarithm function
#define DEXP            exp             // Exponential function
#define DPOW            pow             // Power function
#define DSQRT           sqrt            // Sqrt function
#define DSIN            sin             // Sine function
#define DCOS            cos             // Cosine function
#define DERF            erf             // Erf function
#define DERFC           erfc            // Erfc function
#define DFLOOR          floor           // Floor function
#define DROUND          round           // Rounding function
#define DCEIL           ceil            // Ceiling function
#define DFMOD           fmod            // Floating point remainder function
#define CBLAS_NRM2      cblas_dnrm2     // CBLAS vector 2-norm 
#define CBLAS_DOT       cblas_ddot      // CBLAS vector dot product
#define CBLAS_GEMV      cblas_dgemv     // CBLAS matrix-vector multiplication
#define CBLAS_GEMM      cblas_dgemm     // CBLAS matrix-matrix multiplication
#define CBLAS_SYRK      cblas_dsyrk     // CBLAS symmetric rank-k update
#define CBLAS_TRSM      cblas_dtrsm     // CBLAS triangle solve
#define CBLAS_TRMM      cblas_dtrmm     // CBLAS triangle matrix multiplication
#define LAPACK_GETRF    LAPACKE_dgetrf  // LAPACK LU factorization
#define LAPACK_GETRS    LAPACKE_dgetrs  // LAPACK linear system solve using LU factorization
#define LAPACK_GETRI    LAPACKE_dgetri  // LAPACK LU inverse matrix
#define LAPACK_POTRF    LAPACKE_dpotrf  // LAPACK Cholesky factorization
#define LAPACK_POTRS    LAPACKE_dpotrs  // LAPACK linear system solve using Cholesky factorization
#define LAPACK_POTRI    LAPACKE_dpotri  // LAPACK Cholesky inverse matrix
#define LAPACK_GEQRF    LAPACKE_dgeqrf  // LAPACK QR factorization
#define LAPACK_GEQPF    LAPACKE_dgeqpf  // LAPACK QR factorization with column pivoting
#define LAPACK_ORGQR    LAPACKE_dorgqr  // LAPACK QR Q matrix explicitly construction
#define LAPACK_ORMQR    LAPACKE_dormqr  // LAPACK QR Q matrix multiples another matrix
#define LAPACK_SYEVD    LAPACKE_dsyevd  // LAPACK eigenvalue decomposition
#define LAPACK_GESVD    LAPACKE_dgesvd  // LAPACK singular value decomposition
#define N_DTYPE_64B     8               // 8 double == 64 bytes, for alignment
#define SIMD_LEN        SIMD_LEN_D      // SIMD vector length
#define D_EPS           DBL_EPSILON     // Double precision machine epsilon
#define ASTER_DTYPE_DOUBLE
#endif


#if DTYPE_SIZE == FLOAT_SIZE            // Marcos for float data type
#define DTYPE           float
#define DTYPE_FMTSTR    "%f"
#define DABS            fabsf
#define DLOG            logf
#define DLOG2           log2f
#define DEXP            expf
#define DPOW            powf
#define DSQRT           sqrtf
#define DSIN            sinf
#define DCOS            cosf
#define DERF            erff
#define DERFC           erfcf
#define DFLOOR          floorf
#define DROUND          roundf
#define DFMOD           fmodf
#define DCEIL           ceilf
#define CBLAS_NRM2      cblas_snrm2
#define CBLAS_DOT       cblas_sdot
#define CBLAS_GEMV      cblas_sgemv
#define CBLAS_GEMM      cblas_sgemm
#define CBLAS_SYRK      cblas_ssyrk
#define CBLAS_TRSM      cblas_strsm
#define CBLAS_TRMM      cblas_strmm
#define LAPACK_GETRF    LAPACKE_sgetrf
#define LAPACK_GETRS    LAPACKE_sgetrs
#define LAPACK_GETRI    LAPACKE_sgetri
#define LAPACK_POTRF    LAPACKE_spotrf
#define LAPACK_POTRS    LAPACKE_spotrs
#define LAPACK_POTRI    LAPACKE_spotri
#define LAPACK_GEQRF    LAPACKE_sgeqrf
#define LAPACK_GEQPF    LAPACKE_sgeqpf
#define LAPACK_ORGQR    LAPACKE_sorgqr
#define LAPACK_ORMQR    LAPACKE_sormqr
#define LAPACK_SYEVD    LAPACKE_ssyevd
#define LAPACK_GESVD    LAPACKE_sgesvd
#define N_DTYPE_64B     16
#define SIMD_LEN        SIMD_LEN_S
#define D_EPS           FLT_EPSILON
#define ASTER_DTYPE_FLOAT
#endif

#define QR_RANK         0               // Partial QR stop criteria: maximum rank
#define QR_REL_NRM      1               // Partial QR stop criteria: maximum relative column 2-norm
#define QR_ABS_NRM      2               // Partial QR stop criteria: maximum absolute column 2-norm

#define ALIGN_SIZE      64              // Memory allocation alignment
#define ALPHA_H2        0.999999        // Admissible coefficient for H2,  == 1 here
#define ALPHA_HSS       -0.000001       // Admissible coefficient for HSS, == 0 here

#define BD_NTASK_THREAD 10              // Average number of tasks each thread has in B & D build

#include "linalg_lib_wrapper.h"
#include "ASTER/include/aster.h"

#endif
