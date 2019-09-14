#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include <mkl.h>

#include "H2Pack.h"

struct H2P_test_params
{
    int   pt_dim;
    int   krnl_dim;
    int   n_point;
    int   krnl_mat_size;
    int   BD_JIT;
    int   kernel_id;
    DTYPE rel_tol;
    DTYPE *coord;
    kernel_eval_fptr   krnl_eval;
    kernel_matvec_fptr krnl_matvec;
};
struct H2P_test_params test_params;

void Coulomb_kernel_3d(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *mat, const int ldm
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

void Coulomb_kernel_3d_matvec(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const DTYPE *x_in_0, const DTYPE *x_in_1, 
    DTYPE *x_out_0, DTYPE *x_out_1
)
{
    const DTYPE *x0 = coord0 + ld0 * 0;
    const DTYPE *y0 = coord0 + ld0 * 1;
    const DTYPE *z0 = coord0 + ld0 * 2;
    const DTYPE *x1 = coord1 + ld1 * 0;
    const DTYPE *y1 = coord1 + ld1 * 1;
    const DTYPE *z1 = coord1 + ld1 * 2;
    if (x_in_1 == NULL)
    {
        for (int i = 0; i < n0; i++)
        {
            const DTYPE x0_i = x0[i];
            const DTYPE y0_i = y0[i];
            const DTYPE z0_i = z0[i];
            DTYPE sum = 0.0;
            #pragma omp simd
            for (int j = 0; j < n1; j++)
            {
                DTYPE dx = x0_i - x1[j];
                DTYPE dy = y0_i - y1[j];
                DTYPE dz = z0_i - z1[j];
                DTYPE r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < 1e-20) r2 = 1.0;
                sum += x_in_0[j] / DSQRT(r2);
            }
            x_out_0[i] += sum;
        }
    } else {
        for (int i = 0; i < n0; i++)
        {
            const DTYPE x0_i = x0[i];
            const DTYPE y0_i = y0[i];
            const DTYPE z0_i = z0[i];
            const DTYPE x_in_1_i = x_in_1[i];
            DTYPE sum = 0.0;
            #pragma omp simd
            for (int j = 0; j < n1; j++)
            {
                DTYPE dx = x0_i - x1[j];
                DTYPE dy = y0_i - y1[j];
                DTYPE dz = z0_i - z1[j];
                DTYPE r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < 1e-20) r2 = 1.0;
                DTYPE inv_d = 1.0 / DSQRT(r2);
                sum += x_in_0[j] * inv_d;
                x_out_1[j] += x_in_1_i * inv_d;
            }
            x_out_0[i] += sum;
        }
    }
}

void Laplace_kernel_3d(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *mat, const int ldm
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
            r2 = DSQRT(r2);
            mat_i_row[j] = DLOG(r2);
        }
    }
}

void Gaussian_kernel_3d(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *mat, const int ldm
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
            mat_i_row[j] = DEXP(-r2);
        }
    }
}

void Coulomb_kernel_2d(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *mat, const int ldm
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

void Coulomb_kernel_2d_matvec(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const DTYPE *x_in_0, const DTYPE *x_in_1, 
    DTYPE *x_out_0, DTYPE *x_out_1
)
{
    const DTYPE *x0 = coord0 + ld0 * 0;
    const DTYPE *y0 = coord0 + ld0 * 1;
    const DTYPE *x1 = coord1 + ld1 * 0;
    const DTYPE *y1 = coord1 + ld1 * 1;
    if (x_in_1 == NULL)
    {
        for (int i = 0; i < n0; i++)
        {
            const DTYPE x0_i = x0[i];
            const DTYPE y0_i = y0[i];
            DTYPE sum = 0.0;
            #pragma omp simd
            for (int j = 0; j < n1; j++)
            {
                DTYPE dx = x0_i - x1[j];
                DTYPE dy = y0_i - y1[j];
                DTYPE r2 = dx * dx + dy * dy;
                if (r2 < 1e-20) r2 = 1.0;
                sum += x_in_0[j] / DSQRT(r2);
            }
            x_out_0[i] += sum;
        }
    } else {
        for (int i = 0; i < n0; i++)
        {
            const DTYPE x0_i = x0[i];
            const DTYPE y0_i = y0[i];
            const DTYPE x_in_1_i = x_in_1[i];
            DTYPE sum = 0.0;
            #pragma omp simd
            for (int j = 0; j < n1; j++)
            {
                DTYPE dx = x0_i - x1[j];
                DTYPE dy = y0_i - y1[j];
                DTYPE r2 = dx * dx + dy * dy;
                if (r2 < 1e-20) r2 = 1.0;
                DTYPE inv_d = 1.0 / DSQRT(r2);
                sum += x_in_0[j] * inv_d;
                x_out_1[j] += x_in_1_i * inv_d;
            }
            x_out_0[i] += sum;
        }
    }
}

void Laplace_kernel_2d(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *mat, const int ldm
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
            r2 = DSQRT(r2);
            mat_i_row[j] = DLOG(r2);
        }
    }
}

void Gaussian_kernel_2d(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    DTYPE *mat, const int ldm
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
            mat_i_row[j] = DEXP(-r2);
        }
    }
}

void parse_params(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Kernel dimension   = ");
        scanf("%d", &test_params.pt_dim);
    } else {
        test_params.pt_dim = atoi(argv[1]);
        printf("Kernel dimension   = %d\n", test_params.pt_dim);
    }
    test_params.krnl_dim = 1;
    
    if (argc < 3)
    {
        printf("Number of points   = ");
        scanf("%d", &test_params.n_point);
    } else {
        test_params.n_point = atoi(argv[2]);
        printf("Number of points   = %d\n", test_params.n_point);
    }
    test_params.krnl_mat_size = test_params.krnl_dim * test_params.n_point;
    
    if (argc < 4)
    {
        printf("QR relative tol    = ");
        scanf("%lf", &test_params.rel_tol);
    } else {
        test_params.rel_tol = atof(argv[3]);
        printf("QR relative tol    = %e\n", test_params.rel_tol);
    }
    
    if (argc < 5)
    {
        printf("Just-In-Time B & D = ");
        scanf("%d", &test_params.BD_JIT);
    } else {
        test_params.BD_JIT = atoi(argv[4]);
        printf("Just-In-Time B & D = %d\n", test_params.BD_JIT);
    }
    
    if (argc < 6)
    {
        printf("Kernel function ID = ");
        scanf("%d", &test_params.kernel_id);
    } else {
        test_params.kernel_id = atoi(argv[5]);
        printf("Kernel function ID = %d\n", test_params.kernel_id);
    }
    switch (test_params.kernel_id)
    {
        case 0: printf("Using Coulomb kernel : k(x, y) = 1 / ||x - y||  \n"); break;
        case 1: printf("Using Laplace kernel : k(x, y) = log(||x - y||) \n"); break;
        case 2: printf("Using Gaussian kernel : k(x, y) = exp(-||x - y||^2) \n"); break;
    }
    
    test_params.coord = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.n_point * test_params.pt_dim);
    assert(test_params.coord != NULL);
    
    // Note: coordinates need to be stored in column-major style, i.e. test_params.coord 
    // is row-major and each column stores the coordinate of a point. 
    if (argc < 7)
    {
        DTYPE k = pow((DTYPE) test_params.n_point, 1.0 / (DTYPE) test_params.pt_dim);
        if (test_params.kernel_id == 2) k = 1.0;
        for (int i = 0; i < test_params.n_point; i++)
        {
            DTYPE *coord_i = test_params.coord + i * test_params.pt_dim;
            for (int j = 0; j < test_params.pt_dim; j++)
                coord_i[j] = k * drand48();
        }
        /*
        FILE *ouf = fopen("coord.txt", "w");
        for (int i = 0; i < test_params.n_point; i++)
        {
            DTYPE *coord_i = test_params.coord + i;
            for (int j = 0; j < test_params.pt_dim-1; j++) 
                fprintf(ouf, "% .15lf, ", coord_i[j * test_params.n_point]]);
            fprintf(ouf, "% .15lf\n", coord_i[(test_params.pt_dim-1) * test_params.n_point]);
        }
        fclose(ouf);
        */
    } else {
        FILE *inf = fopen(argv[6], "r");
        for (int i = 0; i < test_params.n_point; i++)
        {
            DTYPE *coord_i = test_params.coord + i;
            for (int j = 0; j < test_params.pt_dim-1; j++) 
                fscanf(inf, "%lf,", &coord_i[j * test_params.n_point]);
            fscanf(inf, "%lf\n", &coord_i[(test_params.pt_dim-1) * test_params.n_point]);
        }
        fclose(inf);
    }
    
    if (test_params.pt_dim == 3) 
    {
        switch (test_params.kernel_id)
        {
            case 0: 
            { 
                test_params.krnl_eval   = Coulomb_kernel_3d; 
                test_params.krnl_matvec = Coulomb_kernel_3d_matvec; 
                break;
            }
            case 1: 
            {
                test_params.krnl_eval   = Laplace_kernel_3d; 
                test_params.krnl_matvec = NULL; 
                break;
            }
            case 2: 
            {
                test_params.krnl_eval   = Gaussian_kernel_3d; 
                test_params.krnl_matvec = NULL; 
                break;
            }
        }
    }
    if (test_params.pt_dim == 2) 
    {
        switch (test_params.kernel_id)
        {
            case 0: 
            {
                test_params.krnl_eval   = Coulomb_kernel_2d; 
                test_params.krnl_matvec = Coulomb_kernel_2d_matvec; 
                break;
            }
            case 1: 
            {
                test_params.krnl_eval   = Laplace_kernel_2d; 
                test_params.krnl_matvec = NULL; 
                break;
            }
            case 2: 
            {
                test_params.krnl_eval   = Gaussian_kernel_2d; 
                test_params.krnl_matvec = NULL; 
                break;
            }
        }
    }
}

void direct_nbody(
    //kernel_eval_fptr krnl_eval, 
    kernel_matvec_fptr krnl_matvec, const int krnl_dim, const int n_point, 
    const DTYPE *coord, const DTYPE *x, DTYPE *y
)
{
    int nx_blk_size = 64;
    int ny_blk_size = 64;
    int row_blk_size = nx_blk_size * krnl_dim;
    int col_blk_size = ny_blk_size * krnl_dim;
    int thread_blk_size = row_blk_size * col_blk_size;
    int nthreads = omp_get_max_threads();
    DTYPE *thread_buffs = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * thread_blk_size * nthreads);
    double st = H2P_get_wtime_sec();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nx_blk_start, nx_blk_len, nx_blk_end;
        H2P_block_partition(n_point, nthreads, tid, &nx_blk_start, &nx_blk_len);
        nx_blk_end = nx_blk_start + nx_blk_len;
        DTYPE *thread_A_blk = thread_buffs + tid * thread_blk_size;

        memset(y + nx_blk_start * krnl_dim, 0, sizeof(DTYPE) * nx_blk_len * krnl_dim);
        
        for (int ix = nx_blk_start; ix < nx_blk_end; ix += nx_blk_size)
        {
            int blk_nx = (ix + nx_blk_size > nx_blk_end) ? nx_blk_end - ix : nx_blk_size;
            for (int iy = 0; iy < n_point; iy += ny_blk_size)
            {
                DTYPE beta = (iy > 0) ? 1.0 : 0.0;
                int blk_ny = (iy + ny_blk_size > n_point) ? n_point - iy : ny_blk_size;
                krnl_matvec(
                    coord + ix, n_point, blk_nx,
                    coord + iy, n_point, blk_ny,
                    x + iy * krnl_dim, NULL, y + ix * krnl_dim, NULL
                );
                /*
                krnl_eval(
                    coord + ix, n_point, blk_nx,
                    coord + iy, n_point, blk_ny,
                    thread_A_blk, blk_ny * krnl_dim
                );
                CBLAS_GEMV(
                    CblasRowMajor, CblasNoTrans, blk_nx * krnl_dim, blk_ny * krnl_dim,
                    1.0, thread_A_blk, blk_ny * krnl_dim,
                    x + iy * krnl_dim, 1, beta, y + ix * krnl_dim, 1
                );
                */
            }
        }
    }
    double et = H2P_get_wtime_sec();
    printf("Direct N-body reference result obtained, used time = %.3lf (s)\n", et - st);
    H2P_free_aligned(thread_buffs);
}

int main(int argc, char **argv)
{
    srand48(time(NULL));
    
    parse_params(argc, argv);
    
    double st, et, ut, total_t;

    H2Pack_t h2pack;
    
    H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    
    H2P_partition_points(h2pack, test_params.n_point, test_params.coord, 0, 0);
    
    // Check if point index permutation is correct in H2Pack
    DTYPE coord_diff_sum = 0.0;
    for (int i = 0; i < test_params.n_point; i++)
    {
        DTYPE *coord_s_i = h2pack->coord + i;
        DTYPE *coord_i   = test_params.coord + h2pack->coord_idx[i];
        for (int j = 0; j < test_params.pt_dim; j++)
        {
            int idx_j = j * test_params.n_point;
            coord_diff_sum += DABS(coord_s_i[idx_j] - coord_i[idx_j]);
        }
    }
    printf("Point index permutation results %s", coord_diff_sum < 1e-15 ? "are correct\n" : "are wrong\n");

    H2P_dense_mat_t *pp;
    DTYPE max_L = h2pack->enbox[h2pack->root_idx * 2 * test_params.pt_dim + test_params.pt_dim];
    st = H2P_get_wtime_sec();
    H2P_generate_proxy_point_ID(
        test_params.pt_dim, test_params.krnl_dim, h2pack->max_level, 
        2, max_L, test_params.krnl_eval, &pp
    );
    et = H2P_get_wtime_sec();
    printf("H2Pack generate proxy point used %.3lf (s)\n", et - st);
    
    H2P_build(h2pack, test_params.krnl_eval, pp, test_params.BD_JIT, test_params.krnl_matvec);
    
    int nthreads = omp_get_max_threads();
    DTYPE *x, *y0, *y1, *tb;
    x  = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.krnl_mat_size);
    y0 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.krnl_mat_size);
    y1 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.krnl_mat_size);
    assert(x != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < test_params.krnl_mat_size; i++) x[i] = drand48();
    
    // Get reference results
    direct_nbody(
        test_params.krnl_matvec, test_params.krnl_dim, 
        test_params.n_point, h2pack->coord, x, y0
    );
    
    // Warm up, reset timers, and test the matvec performance
    H2P_matvec(h2pack, x, y1); 
    h2pack->n_matvec = 0;
    memset(h2pack->timers + 4, 0, sizeof(double) * 5);
    for (int i = 0; i < 10; i++) 
        H2P_matvec(h2pack, x, y1);
    
    H2P_print_statistic(h2pack);
    
    // Verify H2 matvec results
    DTYPE y0_norm = 0.0, err_norm = 0.0;
    for (int i = 0; i < test_params.krnl_mat_size; i++)
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
    H2P_free_aligned(test_params.coord);
    H2P_destroy(h2pack);
}
