#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include <mkl.h>

#include "H2Pack.h"

// Symmetry kernel function: reciprocal
// Input parameters:
//   dim  : Dimension of point coordinate
//   x, y : Coordinate of two points
// Output parameter:
//   <return> : Output of kernel function
DTYPE reciprocal_kernel(const int dim, const DTYPE *x, const DTYPE *y)
{
    DTYPE res = 0.0;
    for (int i = 0; i < dim; i++)
    {
        DTYPE delta = x[i] - y[i];
        res += delta * delta;
    }
    if (res < 1e-20) res = 1.0;
    res = 1.0 / DSQRT(res);
    return res;
}

int main(int argc, char **argv)
{
    int dim  = 2;
    int npts = 10000;
    DTYPE rel_tol = 1e-6;
    
    if (argc < 2)
    {
        printf("Dimension = ");
        scanf("%d", &dim);
    } else {
        dim = atoi(argv[1]);
        printf("Dimension = %d\n", dim);
    }
    
    if (argc < 3)
    {
        printf("N points  = ");
        scanf("%d", &npts);
    } else {
        npts = atoi(argv[2]);
        printf("N points  = %d\n", npts);
    }
    
    if (argc < 4)
    {
        printf("rel_tol   = ");
        scanf("%lf", &rel_tol);
    } else {
        rel_tol = atof(argv[3]);
        printf("rel_tol   = %e\n", rel_tol);
    }
    
    int max_child = 1 << dim;
    int max_leaf_points = 128;
    const DTYPE max_leaf_size = 0.0;
    
    FILE *inf, *ouf;
    double st, et, ut, total_t;
    
    srand48(time(NULL));
    
    DTYPE *coord = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * npts * dim);
    
    if (argc < 5)
    {
        DTYPE k = pow((DTYPE) npts, 1.0 / (DTYPE) dim);
        for (int i = 0; i < npts; i++)
        {
            DTYPE *coord_i = coord + i * dim;
            for (int j = 0; j < dim; j++)
                coord_i[j] = k * (DTYPE) rand() / (DTYPE) RAND_MAX;
        }
        #if 0
        ouf = fopen("coord.txt", "w");
        for (int i = 0; i < npts; i++)
        {
            DTYPE *coord_i = coord + i * dim;
            for (int j = 0; j < dim-1; j++) 
                fprintf(ouf, "% .15lf, ", coord_i[j]);
            fprintf(ouf, "% .15lf\n", coord_i[dim-1]);
        }
        fclose(ouf);
        #endif
    } else {
        inf = fopen(argv[4], "r");
        for (int i = 0; i < npts; i++)
        {
            DTYPE *coord_i = coord + i * dim;
            for (int j = 0; j < dim-1; j++) 
                fscanf(inf, "%lf,", &coord_i[j]);
            fscanf(inf, "%lf\n", &coord_i[dim-1]);
        }
        fclose(inf);
    }
    
    H2Pack_t h2pack;
    H2P_init(&h2pack, dim, QR_REL_NRM, &rel_tol);
    
    H2P_partition_points(h2pack, npts, coord, max_leaf_points, max_leaf_size);

    kernel_func_ptr kernel = reciprocal_kernel;
    H2P_dense_mat_t *pp;
    DTYPE max_L = h2pack->enbox[h2pack->root_idx * 2 * dim + dim];
    st = H2P_get_wtime_sec();
    H2P_generate_proxy_point(dim, h2pack->max_level, 2, max_L, kernel, &pp);
    et = H2P_get_wtime_sec();
    printf("H2Pack generate proxy point used %.3lf (s)\n", et - st);
    H2P_build(h2pack, kernel, pp);
    
    DTYPE *x, *y0, *y1;
    x  = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * npts);
    y0 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * npts);
    y1 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * npts);
    assert(x != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < npts; i++) x[i] = drand48();
    
    st = H2P_get_wtime_sec();
    #pragma omp parallel for
    for (int i = 0; i < npts; i++)
    {
        DTYPE res = 0.0;
        //DTYPE *coord_i = h2pack->coord + i * dim;
        DTYPE coord_i[3], coord_j[3];
        for (int k = 0; k < dim; k++)
            coord_i[k] = h2pack->coord[i + k * npts];
        
        #pragma omp simd
        for (int j = 0; j < npts; j++)
        {
            //DTYPE *coord_j = h2pack->coord + j * dim;
            for (int k = 0; k < dim; k++)
                coord_j[k] = h2pack->coord[j + k * npts];
            
            DTYPE Aij = reciprocal_kernel(dim, coord_i, coord_j);
            res += Aij * x[j];
        }
        y0[i] = res;
    }
    et = H2P_get_wtime_sec();
    printf("Reference result obtained, time = %.4lf (s)\n", et - st);
    
    // Warm up
    H2P_matvec(h2pack, x, y1); 
    h2pack->n_matvec = 0;
    memset(h2pack->timers + 4, 0, sizeof(double) * 5);
    
    for (int i = 0; i < 10; i++) 
        H2P_matvec(h2pack, x, y1);
    
    H2P_print_statistic(h2pack);
    
    DTYPE y0_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < npts; i++)
    {
        DTYPE diff = y1[i] - y0[i];
        y0_norm  += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    y0_norm  = DSQRT(y0_norm);
    err_norm = DSQRT(err_norm);
    printf("||y_{H2} - y||_2 / ||y||_2 = %e\n", err_norm / y0_norm);

    H2P_free_aligned(x);
    H2P_free_aligned(y0);
    H2P_free_aligned(y1);
    H2P_free_aligned(coord);
    H2P_destroy(h2pack);
}
