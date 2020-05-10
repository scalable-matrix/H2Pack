#ifndef __H2PACK_H__
#define __H2PACK_H__

// H2Pack configurations
#include "H2Pack_config.h"

// H2Pack data structure
#include "H2Pack_typedef.h"

// H2Pack auxiliary data structures
#include "H2Pack_aux_structs.h"

// H2Pack point partitioning
#include "H2Pack_partition.h"

// H2Pack ID compression
#include "H2Pack_ID_compress.h"

// H2Pack build H2/HSS representation
#include "H2Pack_build.h"

// H2Pack H2/HSS fast matrix-vector multiplication
#include "H2Pack_matvec.h"

// H2Pack HSS ULV decomposition and solve
#include "H2Pack_HSS_ULV.h"

// H2Pack SPDHSS H2 build
#include "H2Pack_SPDHSS_H2.h"

// H2Pack optimized kernels
#include "H2Pack_kernels.h"

// Helper functions
#include "utils.h"

// Linear algebra library (BLAS, LAPACK) wrapper header
#include "linalg_lib_wrapper.h"

// x86 intrinsic function wrapper
#include "x86_intrin_wrapper.h"

#endif
