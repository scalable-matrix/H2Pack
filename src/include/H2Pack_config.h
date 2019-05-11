#ifndef __H2PACK_CONFIG_H__
#define __H2PACK_CONFIG_H__

// Parameters used in H2Pack

#define DOUBLE_SIZE  8
#define FLOAT_SIZE   4

#define DTYPE_SIZE   DOUBLE_SIZE  // Matrix data type: double or float

#if DTYPE_SIZE == DOUBLE_SIZE     // Functions for double data type
#define DTYPE       double        // Data type
#define DABS        fabs          // Abs function
#define DFLOOR      floor         // Floor function
#define DSQRT       sqrt          // Sqrt function
#define BLAS_NRM2   cblas_dnrm2   // CBLAS 2-norm function
#define BLAS_SCAL   cblas_dscal   // CBLAS vector scaling function
#define BLAS_GEMV   cblas_dgemv   // CBLAS matrix-vector multiplication
#define BLAS_GER    cblas_dger    // CBLAS matrix rank-1 update
#endif

#if DTYPE_SIZE == FLOAT_SIZE      // Functions for float data type
#define DTYPE       float
#define DABS        fabsf
#define DFLOOR      floorf
#define DSQRT       sqrtf    
#define BLAS_NRM2   cblas_snrm2
#define BLAS_SCAL   cblas_sscal
#define BLAS_GEMV   cblas_sgemv
#define BLAS_GER    cblas_sger
#endif

#define ALIGN_SIZE  64            // Memory allocation alignment
#define ALPHA_H2    0.999999      // Admissible coefficient, == 1 here

#endif
