#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"

// Initialize a H2Pack structure
void H2P_init(
    H2Pack_t *h2pack_, const int dim, 
    const int QR_stop_type, void *QR_stop_param
)
{
    H2Pack_t h2pack = (H2Pack_t) malloc(sizeof(struct H2Pack));
    assert(h2pack != NULL);
    
    h2pack->dim       = dim;
    h2pack->mem_bytes = 0;
    h2pack->max_child = 1 << dim;
    h2pack->n_matvec  = 0;
    memset(h2pack->timers,   0, sizeof(double) * 5);
    memset(h2pack->mat_size, 0, sizeof(int)    * 3);
    
    h2pack->QR_stop_type = QR_stop_type;
    if (QR_stop_type == QR_RANK) 
        memcpy(&h2pack->QR_stop_rank, QR_stop_param, sizeof(int));
    if ((QR_stop_type == QR_REL_NRM) || (QR_stop_type == QR_ABS_NRM))
        memcpy(&h2pack->QR_stop_tol,  QR_stop_param, sizeof(DTYPE));
    
    h2pack->parent        = NULL;
    h2pack->children      = NULL;
    h2pack->cluster       = NULL;
    h2pack->n_child       = NULL;
    h2pack->node_level    = NULL;
    h2pack->level_n_node  = NULL;
    h2pack->level_nodes   = NULL;
    h2pack->leaf_nodes    = NULL;
    h2pack->r_inadm_pairs = NULL;
    h2pack->r_adm_pairs   = NULL;
    h2pack->node_adm_list = NULL;
    h2pack->node_adm_cnt  = NULL;
    h2pack->coord         = NULL;
    h2pack->enbox         = NULL;
    h2pack->U             = NULL;
    h2pack->J             = NULL;
    h2pack->B             = NULL;
    h2pack->D             = NULL;
    
    *h2pack_ = h2pack;
}

// Destroy a H2Pack structure
void H2P_destroy(H2Pack_t h2pack)
{
    free(h2pack->parent);
    free(h2pack->children);
    free(h2pack->cluster);
    free(h2pack->n_child);
    free(h2pack->node_level);
    free(h2pack->level_n_node);
    free(h2pack->level_nodes);
    free(h2pack->leaf_nodes);
    free(h2pack->r_inadm_pairs);
    free(h2pack->r_adm_pairs);
    free(h2pack->node_adm_list);
    free(h2pack->node_adm_cnt);
    free(h2pack->coord);
    free(h2pack->enbox);
    
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        H2P_dense_mat_destroy(h2pack->U[i]);
        H2P_int_vec_destroy(h2pack->J[i]);
    }
    for (int i = 0; i < h2pack->n_B; i++)
        H2P_dense_mat_destroy(h2pack->B[i]);
    for (int i = 0; i < h2pack->n_D; i++)
        H2P_dense_mat_destroy(h2pack->D[i]);
    free(h2pack->U);
    free(h2pack->J);
    free(h2pack->B);
    free(h2pack->D);
}
