
void test_H2_matmul(H2Pack_p h2pack, const int n_vec)
{
    double st, et;
    int n_thread = omp_get_num_threads();
    int krnl_mat_size = h2pack->krnl_mat_size;
    int mat_size = krnl_mat_size * n_vec;
    DTYPE *x0, *x1, *y0, *y1, *y2;
    x0 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    x1 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    y0 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    y1 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    y2 = (DTYPE*) malloc(sizeof(DTYPE) * mat_size);
    ASSERT_PRINTF(
        x0 != NULL && x1 != NULL && y0 != NULL && y1 != NULL && y2 != NULL,
        "Failed to allocate 5 arrays of size %d for H2 matmul tests\n", mat_size
    );
    for (int i = 0; i < mat_size; i++) 
    {
        //x0[i] = (DTYPE) pseudo_randn();
        x0[i] = (DTYPE) drand48() - 0.5;
        y0[i] = 0.0;
        y1[i] = 0.0;
    }

    // Test multiple matvec
    st = get_wtime_sec();
    for (int i = 0; i < n_vec; i++)
    {
        DTYPE *x_ivec = x0 + i * krnl_mat_size;
        DTYPE *y_ivec = y0 + i * krnl_mat_size;
        H2P_matvec(h2pack, x_ivec, y_ivec);
    }
    et = get_wtime_sec();
    printf("%3d           matvec used %.3lf sec\n", n_vec, et - st);

    DTYPE y0_2norm, err_2norm, relerr;
    
    // Test column-major matmul performance
    st = get_wtime_sec();
    H2P_matmul(h2pack, CblasColMajor, n_vec, x0, krnl_mat_size, y1, krnl_mat_size);
    et = get_wtime_sec();
    printf("One col-major matmul used %.3lf sec\n", et - st);

    // Check H2 column-major matmul results
    DTYPE cm_max_relerr = 0.0;
    DTYPE cm_avg_relerr = 0.0; 
    for (int i = 0; i < n_vec; i++)
    {
        DTYPE *y0_ivec = y0 + i * krnl_mat_size;
        DTYPE *y1_ivec = y1 + i * krnl_mat_size;
        calc_err_2norm(krnl_mat_size, y0_ivec, y1_ivec, &y0_2norm, &err_2norm);
        relerr = err_2norm / y0_2norm;
        if (relerr > cm_max_relerr) cm_max_relerr = relerr;
        cm_avg_relerr += relerr;
    }
    cm_avg_relerr /= (DTYPE) n_vec;
    
    // Test row-major matmul performance
    //double trans_t = 0.0, matmul_t = 0.0, total_t = 0.0;
    //st = get_wtime_sec();
    H2P_transpose_dmat(n_thread, n_vec, krnl_mat_size, x0, krnl_mat_size, x1, n_vec);
    //et = get_wtime_sec();
    //trans_t += et - st;

    st = get_wtime_sec();
    H2P_matmul(h2pack, CblasRowMajor, n_vec, x1, n_vec, y1, n_vec);
    et = get_wtime_sec();
    //matmul_t = et - st;

    //st = get_wtime_sec();
    H2P_transpose_dmat(n_thread, krnl_mat_size, n_vec, y1, n_vec, y2, krnl_mat_size);
    //et = get_wtime_sec();
    //trans_t += et - st;
    //total_t = matmul_t + trans_t;
    printf("One row-major matmul used %.3lf sec\n", et - st);

    // Check H2 row-major matmul results
    DTYPE rm_max_relerr = 0.0;
    DTYPE rm_avg_relerr = 0.0; 
    for (int i = 0; i < n_vec; i++)
    {
        DTYPE *y0_ivec = y0 + i * krnl_mat_size;
        DTYPE *y2_ivec = y2 + i * krnl_mat_size;
        calc_err_2norm(krnl_mat_size, y0_ivec, y2_ivec, &y0_2norm, &err_2norm);
        relerr = err_2norm / y0_2norm;
        if (relerr > rm_max_relerr) rm_max_relerr = relerr;
        rm_avg_relerr += relerr;
    }
    rm_avg_relerr /= (DTYPE) n_vec;

    printf("%d vectors col-major matmul max/avg relerr = %e, %e\n", n_vec, cm_max_relerr, cm_avg_relerr);
    printf("%d vectors row-major matmul max/avg relerr = %e, %e\n", n_vec, rm_max_relerr, rm_avg_relerr);
    
    free(x0);
    free(x1);
    free(y0);
    free(y1);
    free(y2);
}

