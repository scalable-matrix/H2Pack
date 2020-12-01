#ifndef __LINALG_LIB_WRAPPER_H__
#define __LINALG_LIB_WRAPPER_H__

// Wrapper for linear algebra library (BLAS, LAPACK)

#if !defined(USE_MKL) && !defined(USE_OPENBLAS)
#define USE_OPENBLAS
#endif

#ifdef USE_MKL
#include <mkl.h>
#define BLAS_SET_NUM_THREADS mkl_set_num_threads
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#include <lapacke.h>
#define BLAS_SET_NUM_THREADS openblas_set_num_threads
#endif

#endif

