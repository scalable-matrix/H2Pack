#ifndef __H2PACK_H__
#define __H2PACK_H__

// H2Pack configurations
#include "H2Pack_config.h"

// H2Pack data structure
#include "H2Pack_typedef.h"

// H2Pack auxiliary data structures
#include "H2Pack_aux_structs.h"

// H2Pack hierarchical point partitioning
#include "H2Pack_partition.h"

// H2Pack hierarchical point partitioning for periodic system
#include "H2Pack_partition_periodic.h"

// H2Pack interpolative decomposition compression
#include "H2Pack_ID_compress.h"

// H2Pack generate proxy points
#include "H2Pack_gen_proxy_point.h"

// H2Pack build H2/HSS representation
#include "H2Pack_build.h"

// H2Pack build H2 representation for periodic system
#include "H2Pack_build_periodic.h"

// H2Pack H2/HSS fast matrix-vector multiplication
#include "H2Pack_matvec.h"

// H2Pack H2 fast matrix-vector multiplication for periodic system
#include "H2Pack_matvec_periodic.h"

// H2Pack H2/HSS fast matrix-matrix multiplication
#include "H2Pack_matmul.h"

// H2Pack H2 fast matrix-matrix multiplication for periodic system
#include "H2Pack_matmul_periodic.h"

// H2Pack HSS ULV decomposition and solve
#include "H2Pack_HSS_ULV.h"

// H2Pack SPDHSS H2 build
#include "H2Pack_SPDHSS_H2.h"

// H2Pack file IO
#include "H2Pack_file_IO.h"

// Linear algebra library (BLAS, LAPACK) wrapper header
#include "linalg_lib_wrapper.h"

// Vector wrapper function wrapper
#include "vec_wrapper_func.h"

// Helper functions
#include "utils.h"

#endif
