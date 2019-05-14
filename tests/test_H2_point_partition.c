#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "H2Pack.h"

int main()
{
    const int   dim     = 2;
    const int   npts    = 8000;
    const int   max_child       = 1 << dim;
    const int   max_leaf_points = 100;
    const DTYPE max_leaf_size   = 0.0;
    DTYPE rel_tol = 1e-6;
    FILE *inf, *ouf;
    
    DTYPE *coord = (DTYPE*) malloc(sizeof(DTYPE) * npts * dim);
    inf = fopen("coord.txt", "r");
    for (int i = 0; i < npts * dim; i++) 
        fscanf(inf, "%lf", coord + i);
    fclose(inf);
    
    H2Pack_t h2pack;
    H2P_init(&h2pack, dim, QR_REL_NRM, &rel_tol);
    
    H2P_partition_points(h2pack, npts, coord, max_leaf_points, max_leaf_size);
    printf("H2Pack partition points done, used time = %e (s)\n", h2pack->timers[0]);
    printf(
        "n_node, n_leaf_node, max_child, max_level = %d %d %d %d\n", 
        h2pack->n_node, h2pack->n_leaf_node, h2pack->max_child, h2pack->max_level
    );

    H2P_build(h2pack);
    double storage_k = 0.0;
    storage_k += (double) h2pack->mat_size[0];
    storage_k += (double) h2pack->mat_size[1];
    storage_k += (double) h2pack->mat_size[2];
    storage_k /= (double) npts;
    printf(
        "H2P_build done, build U, B, D time = %.3lf, %.3lf, %.3lf (s)\n",
        h2pack->timers[1], h2pack->timers[2], h2pack->timers[3]
    );
    printf(
        "H2Pack U, B, D size = %d, %d, %d, size(U + B + D) / npts = %.2lf\n", 
        h2pack->mat_size[0], h2pack->mat_size[1], h2pack->mat_size[2], storage_k
    );
    fflush(stdout);
    
    H2P_destroy(h2pack);
    free(coord);
}