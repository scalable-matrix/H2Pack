#ifndef __H2PACK_CONFIG_H__
#define __H2PACK_CONFIG_H__

// Parameters used in H2Pack

#define DOUBLE_SIZE      8
#define FLOAT_SIZE       4

#define DTYPE_SIZE       DOUBLE_SIZE   // Matrix data type: double or float

#if DTYPE_SIZE == DOUBLE_SIZE          // Functions for double data type
#define DTYPE            double        // Data type
#define DABS             fabs          // Abs function
#define DLOG             log           // Logarithm function
#define DEXP             exp           // Exponential function
#define DFLOOR           floor         // Floor function
#define DSQRT            sqrt          // Sqrt function
#define CBLAS_GEMV       cblas_dgemv   // CBLAS matrix-vector multiplication
#define CBLAS_GEMM       cblas_dgemm   // CBLAS matrix-matrix multiplication
#define CBLAS_GER        cblas_dger    // CBLAS matrix rank-1 update
#define CBLAS_TRSM       cblas_dtrsm   // CBLAS triangle solve
#define N_DTYPE_64B      8             // 8 double == 64 bytes, for alignment
#define SIMD_LEN         SIMD_LEN_D
#endif

#if DTYPE_SIZE == FLOAT_SIZE           // Functions for float data type
#define DTYPE            float
#define DABS             fabsf
#define DLOG             logf
#define DEXP             expf
#define DFLOOR           floorf
#define DSQRT            sqrtf    
#define CBLAS_GEMV       cblas_sgemv
#define CBLAS_GEMM       cblas_sgemm
#define CBLAS_GER        cblas_sger
#define CBLAS_TRSM       cblas_strsm
#define N_DTYPE_64B      16
#define SIMD_LEN         SIMD_LEN_S
#endif

#define QR_RANK          0             // Partial QR stop criteria: maximum rank
#define QR_REL_NRM       1             // Partial QR stop criteria: maximum relative column 2-norm
#define QR_ABS_NRM       2             // Partial QR stop criteria: maximum absolute column 2-norm

#define ALIGN_SIZE       64            // Memory allocation alignment
#define ALPHA_H2         0.999999      // Admissible coefficient, == 1 here

#define BD_NTASK_THREAD  10            // Average number of tasks each thread has in B & D build

#endif
