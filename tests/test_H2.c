#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include <mkl.h>

#include "H2Pack.h"

void reciprocal_kernel_3d(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const int dim, DTYPE *mat, const int ldm
)
{
    const DTYPE *x0 = coord0 + ld0 * 0;
    const DTYPE *y0 = coord0 + ld0 * 1;
    const DTYPE *z0 = coord0 + ld0 * 2;
    const DTYPE *x1 = coord1 + ld1 * 0;
    const DTYPE *y1 = coord1 + ld1 * 1;
    const DTYPE *z1 = coord1 + ld1 * 2;
    for (int i = 0; i < n0; i++)
    {
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        const DTYPE z0_i = z0[i];
        DTYPE *mat_i_row = mat + i * ldm;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE dz = z0_i - z1[j];
            DTYPE r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-20) r2 = 1.0;
            mat_i_row[j] = 1.0 / DSQRT(r2);
        }
    }
}

void reciprocal_kernel_2d(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const int dim, DTYPE *mat, const int ldm
)
{
    const DTYPE *x0 = coord0 + ld0 * 0;
    const DTYPE *y0 = coord0 + ld0 * 1;
    const DTYPE *x1 = coord1 + ld1 * 0;
    const DTYPE *y1 = coord1 + ld1 * 1;
    for (int i = 0; i < n0; i++)
    {
        const DTYPE x0_i = x0[i];
        const DTYPE y0_i = y0[i];
        DTYPE *mat_i_row = mat + i * ldm;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE dy = y0_i - y1[j];
            DTYPE r2 = dx * dx + dy * dy;
            if (r2 < 1e-20) r2 = 1.0;
            mat_i_row[j] = 1.0 / DSQRT(r2);
        }
    }
}

void direct_nbody(
    kernel_func_ptr kernel, const int dim, const int n_point, 
    const DTYPE *coord, const DTYPE *x, DTYPE *y
)
{
    int row_blk_size = 128;
    int col_blk_size = 128;
    int thread_blk_size = row_blk_size * col_blk_size;
    int nthreads = omp_get_max_threads();
    DTYPE *thread_buffs = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * thread_blk_size * nthreads);
    double st = H2P_get_wtime_sec();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int t_srow, t_nrow, t_erow;
        H2P_block_partition(n_point, nthreads, tid, &t_srow, &t_nrow);
        t_erow = t_srow + t_nrow;
        DTYPE *thread_A_blk = thread_buffs + tid * thread_blk_size;
        
        for (int irow = t_srow; irow < t_erow; irow += row_blk_size)
        {
            int blk_nrow = (irow + row_blk_size > t_erow) ? t_erow - irow : row_blk_size;
            for (int icol = 0; icol < n_point; icol += col_blk_size)
            {
                DTYPE beta = (icol > 0) ? 1.0 : 0.0;
                int blk_ncol = (icol + col_blk_size > n_point) ? n_point - icol : col_blk_size;
                kernel(
                    coord + irow, n_point, blk_nrow,
                    coord + icol, n_point, blk_ncol,
                    dim, thread_A_blk, blk_ncol
                );
                CBLAS_GEMV(
                    CblasRowMajor, CblasNoTrans, blk_nrow, blk_ncol,
                    1.0, thread_A_blk, blk_ncol,
                    x + icol, 1, beta, y + irow, 1
                );
            }
        }
    }
    double et = H2P_get_wtime_sec();
    printf("Direct N-body reference result obtained, used time = %.3lf (s)\n", et - st);
    H2P_free_aligned(thread_buffs);
}

int main(int argc, char **argv)
{
    int dim    = 2;
    int npts   = 10000;
    int BD_JIT = 0;
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
    
    if (argc < 5)
    {
        printf("BD_JIT    = ");
        scanf("%d", &BD_JIT);
    } else {
        BD_JIT = atoi(argv[4]);
        printf("BD_JIT    = %d\n", BD_JIT);
    }

    FILE *inf, *ouf;
    double st, et, ut, total_t;
    
    srand48(time(NULL));
    
    DTYPE *coord = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * npts * dim);
    
    if (argc < 6)
    {
        DTYPE k = pow((DTYPE) npts, 1.0 / (DTYPE) dim);
        for (int i = 0; i < npts; i++)
        {
            DTYPE *coord_i = coord + i * dim;
            for (int j = 0; j < dim; j++)
                coord_i[j] = k * drand48();
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
        inf = fopen(argv[5], "r");
        for (int i = 0; i < npts; i++)
        {
            DTYPE *coord_i = coord + i;
            for (int j = 0; j < dim-1; j++) 
                fscanf(inf, "%lf,", &coord_i[j * npts]);
            fscanf(inf, "%lf\n", &coord_i[(dim-1) * npts]);
        }
        fclose(inf);
    }
    
    H2Pack_t h2pack;
    H2P_init(&h2pack, dim, QR_REL_NRM, &rel_tol);
    H2P_partition_points(h2pack, npts, coord, 0);
    
    DTYPE coord_diff_sum = 0.0;
    for (int i = 0; i < npts; i++)
    {
        DTYPE *coord_s_i = h2pack->coord + i;
        DTYPE *coord_i   = coord + h2pack->coord_idx[i];
        for (int j = 0; j < dim; j++)
            coord_diff_sum += DABS(coord_s_i[j * npts] - coord_i[j * npts]);
    }
    printf("Coordinate permutation results %s", coord_diff_sum < 1e-15 ? "are correct\n" : "are wrong\n");

    kernel_func_ptr kernel;
    if (dim == 3) kernel = reciprocal_kernel_3d;
    if (dim == 2) kernel = reciprocal_kernel_2d;
    H2P_dense_mat_t *pp;
    DTYPE max_L = h2pack->enbox[h2pack->root_idx * 2 * dim + dim];
    st = H2P_get_wtime_sec();
    H2P_generate_proxy_point(dim, h2pack->max_level, 2, max_L, kernel, &pp);
    et = H2P_get_wtime_sec();
    printf("H2Pack generate proxy point used %.3lf (s)\n", et - st);
    
    H2P_build(h2pack, kernel, pp, BD_JIT);
    
    int nthreads = omp_get_max_threads();
    DTYPE *x, *y0, *y1, *tb;
    x  = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * npts);
    y0 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * npts);
    y1 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * npts);
    assert(x != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < npts; i++) x[i] = drand48();
    
    // Get reference results
    direct_nbody(kernel, dim, npts, h2pack->coord, x, y0);
    
    // Warm up, reset timers, and test the matvec performance
    H2P_matvec(h2pack, x, y1); 
    h2pack->n_matvec = 0;
    memset(h2pack->timers + 4, 0, sizeof(double) * 5);
    for (int i = 0; i < 10; i++) 
        H2P_matvec(h2pack, x, y1);
    
    H2P_print_statistic(h2pack);
    
    // Verify H2 matvec results
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
