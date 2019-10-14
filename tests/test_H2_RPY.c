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
    int   krnl_matvec_flops;
    DTYPE rel_tol;
    DTYPE *coord;
    kernel_eval_fptr   krnl_eval;
    kernel_matvec_fptr krnl_matvec;
};
struct H2P_test_params test_params;

void parse_params(int argc, char **argv)
{
    test_params.pt_dim = 3;
    test_params.krnl_dim = 3;
    
    printf("Using 3D RPY kernel\n");
    if (argc < 2)
    {
        printf("Number of points   = ");
        scanf("%d", &test_params.n_point);
    } else {
        test_params.n_point = atoi(argv[1]);
        printf("Number of points   = %d\n", test_params.n_point);
    }
    test_params.krnl_mat_size = test_params.n_point * test_params.krnl_dim;
    
    if (argc < 3)
    {
        printf("QR relative tol    = ");
        scanf("%lf", &test_params.rel_tol);
    } else {
        test_params.rel_tol = atof(argv[2]);
        printf("QR relative tol    = %e\n", test_params.rel_tol);
    }
    
    if (argc < 4)
    {
        printf("Just-In-Time B & D = ");
        scanf("%d", &test_params.BD_JIT);
    } else {
        test_params.BD_JIT = atoi(argv[3]);
        printf("Just-In-Time B & D = %d\n", test_params.BD_JIT);
    }
    
    test_params.coord = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.n_point * test_params.pt_dim);
    assert(test_params.coord != NULL);
    
    // Note: coordinates need to be stored in column-major style, i.e. test_params.coord 
    // is row-major and each column stores the coordinate of a point. 
    int need_gen = 1;
    if (argc >= 7)
    {
        DTYPE *tmp = (DTYPE*) malloc(sizeof(DTYPE) * test_params.n_point * test_params.pt_dim);
        if (strstr(argv[6], ".csv") != NULL)
        {
            printf("Reading coordinates from CSV file...");
            FILE *inf = fopen(argv[6], "r");
            for (int i = 0; i < test_params.n_point; i++)
            {
                for (int j = 0; j < test_params.pt_dim-1; j++) 
                    fscanf(inf, "%lf,", &tmp[i * test_params.pt_dim + j]);
                fscanf(inf, "%lf\n", &tmp[i * test_params.pt_dim + test_params.pt_dim-1]);
            }
            fclose(inf);
            printf(" done.\n");
            need_gen = 0;
        }
        if (strstr(argv[6], ".bin") != NULL)
        {
            printf("Reading coordinates from binary file...");
            FILE *inf = fopen(argv[6], "rb");
            fread(tmp, sizeof(DTYPE), test_params.n_point * test_params.pt_dim, inf);
            fclose(inf);
            printf(" done.\n");
            need_gen = 0;
        }
        if (need_gen == 0)
        {
            for (int i = 0; i < test_params.pt_dim; i++)
                for (int j = 0; j < test_params.n_point; j++)
                    test_params.coord[i * test_params.n_point + j] = tmp[j * test_params.pt_dim + i];
        }
        free(tmp);
    }
    if (need_gen == 1)
    {
        printf("Binary/CSV coordinate file not provided. Generating random coordinates in unit box...");
        for (int i = 0; i < test_params.n_point * test_params.pt_dim; i++)
            test_params.coord[i] = drand48();
        printf(" done.\n");
    }
    
    test_params.krnl_eval         = RPY_eval_std;
    test_params.krnl_matvec       = RPY_matvec_std;
    test_params.krnl_matvec_flops = RPY_matvec_flop;
}

void direct_nbody(
    //kernel_eval_fptr krnl_eval, 
    kernel_matvec_fptr krnl_matvec, const int krnl_matvec_flops, 
    const int krnl_dim, const int n_point, const DTYPE *coord, const DTYPE *x, DTYPE *y
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
    double ut = H2P_get_wtime_sec() - st;
    double GFLOPS = (double)(n_point) * (double)(n_point) * (double)(krnl_matvec_flops);
    GFLOPS = GFLOPS / 1000000000.0;
    printf("Direct N-body reference result obtained, %.3lf s, %.2lf GFLOPS\n", ut, GFLOPS / ut);
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
    H2P_generate_proxy_point_surface(
        test_params.pt_dim, 600, h2pack->max_level, 
        2, max_L, test_params.krnl_eval, &pp
    );
    et = H2P_get_wtime_sec();
    //printf("Proxy point generation used %.3lf (s)\n", et - st);
    
    H2P_build(
        h2pack, pp, test_params.BD_JIT, test_params.krnl_eval, 
        test_params.krnl_matvec, test_params.krnl_matvec_flops
    );
    
    int nthreads = omp_get_max_threads();
    DTYPE *x, *y0, *y1, *tb;
    x  = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.krnl_mat_size);
    y0 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.krnl_mat_size);
    y1 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.krnl_mat_size);
    assert(x != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < test_params.krnl_mat_size; i++) x[i] = 1.0; //drand48();
    
    // Get reference results
    direct_nbody(
        test_params.krnl_matvec, test_params.krnl_matvec_flops, 
        test_params.krnl_dim, test_params.n_point, h2pack->coord, x, y0
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
