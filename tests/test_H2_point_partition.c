#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "H2Pack.h"

int main()
{
    const int   dim     = 2;
    const int   npts    = 8000;
    const DTYPE rel_tol = 1e-6;
    const int   max_child       = 1 << dim;
    const int   max_leaf_points = 100;
    const DTYPE max_leaf_size   = 0.0;
    FILE *inf, *ouf;
    
    DTYPE *coord = (DTYPE*) malloc(sizeof(DTYPE) * npts * dim);
    inf = fopen("coord.txt", "r");
    for (int i = 0; i < npts * dim; i++) 
        fscanf(inf, "%lf", coord + i);
    fclose(inf);
    
    H2Pack_t h2pack;
    H2P_init(&h2pack, dim, rel_tol);
    
    H2P_partition_points(h2pack, npts, coord, max_leaf_points, max_leaf_size);
    printf("H2Pack partition points done, used time = %e (s)\n", h2pack->timers[0]);
    printf("n_node, n_leaf_node, max_child, max_level = %d %d %d %d\n", h2pack->n_node, h2pack->n_leaf_node, h2pack->max_child, h2pack->max_level);
    
    H2P_calc_admissible_pairs(h2pack);
    printf("H2Pack calc (in)adm pairs done, used time = %e (s)\n", h2pack->timers[1]);
    
    ouf = fopen("admcnt.txt", "w");
    for (int i = 0; i < h2pack->n_node; i++) fprintf(ouf, "%d\n", h2pack->node_adm_cnt[i]);
    fclose(ouf);
    
    ouf = fopen("admnode.txt", "w");
    for (int i = 0; i < h2pack->n_node; i++)
    {
        int *adm_list_i = h2pack->node_adm_list + i * h2pack->n_node;
        for (int j = 0; j < h2pack->node_adm_cnt[i]; j++) fprintf(ouf, "%d, ", adm_list_i[j] + 1);
        for (int j = h2pack->node_adm_cnt[i]; j < h2pack->n_node; j++) fprintf(ouf, "0, ");
        fprintf(ouf, "\n");
    }
    fclose(ouf);
    
    H2P_destroy(h2pack);
    free(coord);
}