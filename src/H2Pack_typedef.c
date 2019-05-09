#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"

// Initialize a H2Pack structure
void H2P_init(H2Pack_t *h2pack_, const int dim, const DTYPE reltol)
{
    H2Pack_t h2pack = (H2Pack_t) malloc(sizeof(struct H2Pack));
    assert(h2pack != NULL);
    
    h2pack->dim       = dim;
    h2pack->reltol    = reltol;
    h2pack->mem_bytes = 0;
    h2pack->max_child = 1 << dim;
    memset(h2pack->timers, 0, sizeof(double) * 4);
    
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
}

