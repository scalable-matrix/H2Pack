#ifndef __H2PACK_FILE_IO_H__
#define __H2PACK_FILE_IO_H__

#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

void H2P_store_to_file(H2Pack_p h2pack, const char *metadata_fname, const char *binary_fname);

void H2P_read_from_file(
    H2Pack_p *h2pack_, const char *metadata_fname, const char *binary_fname, const int BD_JIT, 
    void *krnl_param, kernel_eval_fptr krnl_eval, kernel_bimv_fptr krnl_bimv, const int krnl_bimv_flops
);

#ifdef __cplusplus
}
#endif

#endif
