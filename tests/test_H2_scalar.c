#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include <mkl.h>
//#include <ittnotify.h>

#include "H2Pack.h"

struct H2P_test_params
{
    int   pt_dim;
    int   krnl_dim;
    int   n_point;
    int   krnl_mat_size;
    int   BD_JIT;
    int   kernel_id;
    int   krnl_bimv_flops;
    DTYPE rel_tol;
    DTYPE *coord;
    kernel_eval_fptr krnl_eval;
    kernel_bimv_fptr krnl_bimv;
};
struct H2P_test_params test_params;

void parse_params(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Point dimension    = ");
        scanf("%d", &test_params.pt_dim);
    } else {
        test_params.pt_dim = atoi(argv[1]);
        printf("Point dimension    = %d\n", test_params.pt_dim);
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
        case 0: printf("Using Laplace kernel : k(x, y) = 1 / |x - y|  \n"); break;
        case 1: printf("Using Gaussian kernel : k(x, y) = exp(-|x - y|^2) \n"); break;
        case 2: printf("Using 3/2 Matern kernel : k(x, y) = (1 + k) * exp(-k), where k = sqrt(3) * |x - y| \n"); break;
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
    
    if (test_params.pt_dim == 3) 
    {
        switch (test_params.kernel_id)
        {
            case 0: 
            { 
                test_params.krnl_eval       = Coulomb_3d_eval_intrin_d; 
                test_params.krnl_bimv       = Coulomb_3d_krnl_bimv_intrin_d; 
                test_params.krnl_bimv_flops = Coulomb_3d_krnl_bimv_flop;
                break;
            }
            case 1: 
            {
                test_params.krnl_eval       = Gaussian_3d_eval_intrin_d; 
                test_params.krnl_bimv       = Gaussian_3d_krnl_bimv_intrin_d; 
                test_params.krnl_bimv_flops = Gaussian_3d_krnl_bimv_flop;
                break;
            }
            case 2: 
            {
                test_params.krnl_eval       = Matern_3d_eval_intrin_d; 
                test_params.krnl_bimv       = Matern_3d_krnl_bimv_intrin_d; 
                test_params.krnl_bimv_flops = Matern_3d_krnl_bimv_flop;
                break;
            }
        }
    }
}

void direct_nbody(
    const void *krnl_param, kernel_bimv_fptr krnl_bimv, const int krnl_bimv_flops, const int pt_dim, 
    const int krnl_dim, const int n_point, const DTYPE *coord, const DTYPE *x, DTYPE *y
)
{
    int n_point_ext = (n_point + SIMD_LEN - 1) / SIMD_LEN * SIMD_LEN;
    int n_vec_ext = n_point_ext / SIMD_LEN;
    DTYPE *coord_ext = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * pt_dim * n_point_ext);
    DTYPE *x_ext = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * krnl_dim * n_point_ext);
    DTYPE *y_ext = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * krnl_dim * n_point_ext);
    
    for (int i = 0; i < pt_dim; i++)
    {
        const DTYPE *src = coord + i * n_point;
        DTYPE *dst = coord_ext + i * n_point_ext;
        memcpy(dst, src, sizeof(DTYPE) * n_point);
        for (int j = n_point; j < n_point_ext; j++) dst[j] = 1e100;
    }
    memset(y_ext, 0, sizeof(DTYPE) * krnl_dim * n_point_ext);
    memcpy(x_ext, x, sizeof(DTYPE) * krnl_dim * n_point);
    for (int j = krnl_dim * n_point; j < krnl_dim * n_point_ext; j++) x_ext[j] = 0.0;
    
    int blk_size = 512;
    int nthreads = omp_get_max_threads();
    DTYPE *buff  = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * nthreads * blk_size);
    double st = H2P_get_wtime_sec();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_vec, n_vec;
        H2P_block_partition(n_vec_ext, nthreads, tid, &start_vec, &n_vec);
        int start_point = start_vec * SIMD_LEN;
        int n_point1 = n_vec * SIMD_LEN;
        DTYPE *thread_buff = buff + tid * blk_size;
        
        for (int ix = start_point; ix < start_point + n_point1; ix += blk_size)
        {
            int nx = (ix + blk_size > start_point + n_point1) ? (start_point + n_point1 - ix) : blk_size;
            for (int iy = 0; iy < n_point_ext; iy += blk_size)
            {
                int ny = (iy + blk_size > n_point_ext) ? (n_point_ext - iy) : blk_size;
                DTYPE *x_in  = x_ext + iy * krnl_dim;
                DTYPE *x_out = y_ext + ix * krnl_dim;
                krnl_bimv(
                    coord_ext + ix, n_point_ext, nx,
                    coord_ext + iy, n_point_ext, ny,
                    krnl_param, x_in, x_in, x_out, thread_buff
                );
            }
        }
    }
    double ut = H2P_get_wtime_sec() - st;
    H2P_free_aligned(buff);
    double GFLOPS = (double)(n_point) * (double)(n_point) * (double)(krnl_bimv_flops - 2);
    GFLOPS = GFLOPS / 1000000000.0;
    printf("Direct N-body reference result obtained, %.3lf s, %.2lf GFLOPS\n", ut, GFLOPS / ut);
    
    memcpy(y, y_ext, sizeof(DTYPE) * krnl_dim * n_point);
    H2P_free_aligned(y_ext);
    H2P_free_aligned(x_ext);
    H2P_free_aligned(coord_ext);
}

int main(int argc, char **argv)
{
    //__itt_pause();
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
    void *krnl_param = NULL;  // We don't need kernel parameters yet
    
    st = H2P_get_wtime_sec();
    H2P_generate_proxy_point_ID(
        test_params.pt_dim, test_params.krnl_dim, test_params.rel_tol, h2pack->max_level, 
        2, max_L, krnl_param, test_params.krnl_eval, &pp
    );
    et = H2P_get_wtime_sec();
    printf("H2Pack generate proxy points used %.3lf (s)\n", et - st);
    
    H2P_build(
        h2pack, pp, test_params.BD_JIT, krnl_param, 
        test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
    );
    
    int nthreads = omp_get_max_threads();
    DTYPE *x, *y0, *y1, *tb;
    x  = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.krnl_mat_size);
    y0 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.krnl_mat_size);
    y1 = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * test_params.krnl_mat_size);
    assert(x != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < test_params.krnl_mat_size; i++) x[i] = drand48();
    
    // Get reference results
    direct_nbody(
        krnl_param, test_params.krnl_bimv, test_params.krnl_bimv_flops, test_params.pt_dim, 
        test_params.krnl_dim, test_params.n_point, h2pack->coord, x, y0
    );
    
    // Warm up, reset timers, and test the matvec performance
    H2P_matvec(h2pack, x, y1); 
    h2pack->n_matvec = 0;
    memset(h2pack->timers + 4, 0, sizeof(double) * 5);
    //__itt_resume();
    for (int i = 0; i < 10; i++) 
        H2P_matvec(h2pack, x, y1);
    //__itt_pause();
    
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
