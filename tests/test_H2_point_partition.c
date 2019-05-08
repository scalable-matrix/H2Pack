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
    FILE *inf;
    
    DTYPE *coord = (DTYPE*) malloc(sizeof(DTYPE) * npts * dim);
    inf = fopen("coord.txt", "r");
    for (int i = 0; i < npts * dim; i++) 
        fscanf(inf, "%lf", coord + i);
    fclose(inf);
    
    H2Pack_t h2pack;
    H2P_init(&h2pack, dim, rel_tol);
    
    double st = H2P_get_wtime_sec();
    H2P_partition_points(h2pack, npts, coord, max_leaf_points, max_leaf_size);
    double et = H2P_get_wtime_sec();
    printf("H2Pack partition points done, used time = %e (s)\n", et - st);
    printf("n_node, n_leaf_node, max_child, max_level = %d %d %d %d\n", h2pack->n_node, h2pack->n_leaf_node, h2pack->max_child, h2pack->max_level);
    
    FILE *ouf;
    ouf = fopen("coord1.txt", "w");
    for (int i = 0; i < npts; i++)
    {
        DTYPE *coord_i = h2pack->coord + i * dim;
        for (int j = 0; j < dim - 1; j++) fprintf(ouf, "%.15lf, ", coord_i[j]);
        fprintf(ouf, "%.15lf\n", coord_i[dim - 1]);
    }
    fclose(ouf);
    
    ouf = fopen("cluster.txt", "w");
    for (int i = 0; i < h2pack->n_node; i++)
        fprintf(ouf, "%d, %d\n", h2pack->cluster[2*i]+1, h2pack->cluster[2*i+1]+1);
    fclose(ouf);
    
    ouf = fopen("parent.txt", "w");
    for (int i = 0; i < h2pack->n_node; i++)
        fprintf(ouf, "%d\n", h2pack->parent[i]+1);
    fclose(ouf);
    
    H2P_destroy(h2pack);
    free(coord);
}