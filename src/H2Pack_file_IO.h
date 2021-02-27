#ifndef __H2PACK_FILE_IO_H__
#define __H2PACK_FILE_IO_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Store constructed H2 representation to files 
// Input parameters:
//   h2pack          : H2Pack structure after calling H2P_build()
//   meta_json_fname : Metadata JSON file name
//   aux_json_fname  : Auxiliary JSON file name, can be NULL
//   binary_fname    : Binary data file name
void H2P_store_to_file(
    H2Pack_p h2pack, const char *meta_json_fname, 
    const char *aux_json_fname, const char *binary_fname
);

// Load a constructed H2 representation from files
// Input parameters:
//   meta_json_fname : Metadata JSON file name
//   aux_json_fname  : Auxiliary JSON file name, can be NULL
//   binary_fname    : Binary data file name
//   BD_JIT          : If H2Pack should use just-in-time matvec mode, 0 or 1
//   krnl_param      : Pointer to the krnl_eval parameter buffer
//   krnl_eval       : Pointer to the kernel matrix evaluation function, can be NULL
//   krnl_bimv       : Pointer to the kernel matrix bi-matvec function, can be NULL
//   krnl_bimv_flops : Number of flops required for each bi-matvec operation, for statistic only
// Output parameter:
//   *h2pack_ : H2Pack structure constructed from given files
// Notes:
//   If only meta_json_fname and binary_fname are valid non-empty values, the constructed
//   H2Pack matrix can only be used to perform H2P_matvec(). Performing other operations
//   may crash the program. 
void H2P_read_from_file(
    H2Pack_p *h2pack_, const char *meta_json_fname, const char *aux_json_fname, 
    const char *binary_fname, const int BD_JIT, void *krnl_param, 
    kernel_eval_fptr krnl_eval, kernel_bimv_fptr krnl_bimv, const int krnl_bimv_flops
);

#ifdef __cplusplus
}
#endif

#endif
