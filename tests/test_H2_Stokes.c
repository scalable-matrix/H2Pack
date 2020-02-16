#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

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
    test_params.pt_dim = 3;
    test_params.krnl_dim = 3;
    
    printf("Using 3D Stokes kernel\n");
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
    
    test_params.coord = (DTYPE*) malloc_aligned(sizeof(DTYPE) * test_params.n_point * test_params.pt_dim, 64);
    assert(test_params.coord != NULL);
    
    // Note: coordinates need to be stored in column-major style, i.e. test_params.coord 
    // is row-major and each column stores the coordinate of a point. 
    int need_gen = 1;
    if (argc >= 5)
    {
        DTYPE *tmp = (DTYPE*) malloc(sizeof(DTYPE) * test_params.n_point * test_params.pt_dim);
        if (strstr(argv[4], ".csv") != NULL)
        {
            printf("Reading coordinates from CSV file...");
            FILE *inf = fopen(argv[4], "r");
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
        if (strstr(argv[4], ".bin") != NULL)
        {
            printf("Reading coordinates from binary file...");
            FILE *inf = fopen(argv[4], "rb");
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
    
    test_params.krnl_eval       = Stokes_eval_std;
    test_params.krnl_bimv       = Stokes_krnl_bimv_intrin_d;
    test_params.krnl_bimv_flops = Stokes_krnl_bimv_flop;
}

void direct_nbody(
    const void *krnl_param, kernel_eval_fptr krnl_eval, const int pt_dim, const int krnl_dim, 
    const DTYPE *src_coord, const int src_coord_ld, const int n_src_pt, const DTYPE *src_val,
    const DTYPE *dst_coord, const int dst_coord_ld, const int n_dst_pt, DTYPE *dst_val
)
{
    const int npt_blk  = 256;
    const int blk_size = npt_blk * krnl_dim;
    const int n_thread = omp_get_max_threads();
    
    memset(dst_val, 0, sizeof(DTYPE) * n_dst_pt * krnl_dim);
    
    DTYPE *krnl_mat_buffs = (DTYPE*) malloc(sizeof(DTYPE) * n_thread * blk_size * blk_size);
    assert(krnl_mat_buffs != NULL);
    
    double st = get_wtime_sec();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        DTYPE *krnl_mat_buff = krnl_mat_buffs + tid * blk_size * blk_size;
        
        int tid_dst_pt_s, tid_dst_pt_n, tid_dst_pt_e;
        calc_block_spos_len(n_dst_pt, n_thread, tid, &tid_dst_pt_s, &tid_dst_pt_n);
        tid_dst_pt_e = tid_dst_pt_s + tid_dst_pt_n;
        
        for (int dst_pt_idx = tid_dst_pt_s; dst_pt_idx < tid_dst_pt_e; dst_pt_idx += npt_blk)
        {
            int dst_pt_blk = (dst_pt_idx + npt_blk > tid_dst_pt_e) ? (tid_dst_pt_e - dst_pt_idx) : npt_blk;
            int krnl_mat_nrow = dst_pt_blk * krnl_dim;
            const DTYPE *dst_coord_ptr = dst_coord + dst_pt_idx;
            DTYPE *dst_val_ptr = dst_val + dst_pt_idx * krnl_dim;
            for (int src_pt_idx = 0; src_pt_idx < n_src_pt; src_pt_idx += npt_blk)
            {
                int src_pt_blk = (src_pt_idx + npt_blk > n_src_pt) ? (n_src_pt - src_pt_idx) : npt_blk;
                int krnl_mat_ncol = src_pt_blk * krnl_dim;
                const DTYPE *src_coord_ptr = src_coord + src_pt_idx;
                const DTYPE *src_val_ptr = src_val + src_pt_idx * krnl_dim;
                
                krnl_eval(
                    dst_coord_ptr, dst_coord_ld, dst_pt_blk,
                    src_coord_ptr, src_coord_ld, src_pt_blk, 
                    krnl_param, krnl_mat_buff, krnl_mat_ncol
                );
                
                CBLAS_GEMV(
                    CblasRowMajor, CblasNoTrans, krnl_mat_nrow, krnl_mat_ncol, 
                    1.0, krnl_mat_buff, krnl_mat_ncol, src_val_ptr, 1, 1.0, dst_val_ptr, 1
                );
            }
        }
    }
    double ut = get_wtime_sec() - st;
    printf("Calculate direct n-body reference results for %d points done\n", n_dst_pt);
    free(krnl_mat_buffs);
}


int main(int argc, char **argv)
{
    srand48(time(NULL));
    
    parse_params(argc, argv);
    
    H2Pack_t h2pack;
    
    H2P_init(&h2pack, test_params.pt_dim, test_params.krnl_dim, QR_REL_NRM, &test_params.rel_tol);
    
    H2P_partition_points(h2pack, test_params.n_point, test_params.coord, 300, 0);
    
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
    int start_level = 2;
    int num_pp = ceil(-log10(test_params.rel_tol)) - 1;
    if (num_pp < 4 ) num_pp = 4;
    if (num_pp > 10) num_pp = 10;
    num_pp = 6 * num_pp * num_pp;
    H2P_generate_proxy_point_surface(
        test_params.pt_dim, num_pp, h2pack->max_level, 
        start_level, max_L, &pp
    );
    
    double eta = 0.5;
    double a   = 0.8;
    double krnl_param[2] = {eta, a};
    H2P_build(
        h2pack, pp, test_params.BD_JIT, krnl_param, 
        test_params.krnl_eval, test_params.krnl_bimv, test_params.krnl_bimv_flops
    );
    
    int n_check_pt = 10000, check_pt_s;
    if (n_check_pt > test_params.n_point)
    {
        n_check_pt = test_params.n_point;
        check_pt_s = 0;
    } else {
        srand(time(NULL));
        check_pt_s = rand() % (test_params.n_point - n_check_pt);
    }
    printf("Calculating direct n-body reference result for points %d -> %d\n", check_pt_s, check_pt_s + n_check_pt - 1);
    
    DTYPE *x, *y0, *y1;
    x  = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_dim * n_check_pt);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * test_params.krnl_mat_size);
    assert(x != NULL && y0 != NULL && y1 != NULL);
    for (int i = 0; i < test_params.krnl_mat_size; i++) x[i] = drand48();
    
    // Get reference results
    direct_nbody(
        krnl_param, test_params.krnl_eval, test_params.pt_dim, test_params.krnl_dim, 
        h2pack->coord,              test_params.n_point, test_params.n_point, x, 
        h2pack->coord + check_pt_s, test_params.n_point, n_check_pt,          y0
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
    for (int i = 0; i < test_params.krnl_dim * n_check_pt; i++)
    {
        DTYPE diff = y1[test_params.krnl_dim * check_pt_s + i] - y0[i];
        y0_norm  += y0[i] * y0[i];
        err_norm += diff * diff;
    }
    y0_norm  = DSQRT(y0_norm);
    err_norm = DSQRT(err_norm);
    printf("For %d validating points: ||y_{H2} - y||_2 / ||y||_2 = %e\n", n_check_pt, err_norm / y0_norm);
    
    free(x);
    free(y0);
    free(y1);
    free_aligned(test_params.coord);
    H2P_destroy(h2pack);
}
