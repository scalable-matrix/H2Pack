// Setup H2Pack:
//      0.  clean up possible remainings from the previous run
//      1.  parse the variables to set up test_params
//      2.  H2P_init for h2pack
//      3.  H2P_partition_points
// Input parameters:
//      kernel             : 
//      kernel_dimension   : 
//      point_coord        : 
//      point_dimension    : 
//      rel_tol            : 
//      kernel_param (opt) :
//      JIT_mode     (opt) : 
static PyObject *setup(PyObject *self, PyObject *args, PyObject *keywds);
// static PyObject *h2build(PyObject *self, PyObject *args);
static PyObject *h2matvec(PyObject *self, PyObject *args);
static PyObject *direct_matvec(PyObject *self, PyObject *args);
static PyObject *clean(PyObject *self, PyObject *args);
static PyObject *print_statistic(PyObject *self, PyObject *args);
static PyObject *print_setting(PyObject *self, PyObject *args);
static PyObject *print_kernels(PyObject *self, PyObject *args);

//  
//  Auxiliary functions 
//

#ifndef SIMD_LEN_D
#define SIMD_LEN_D 4
#endif

void direct_nbody(
    const void *krnl_param, kernel_eval_fptr krnl_eval, const int pt_dim, 
    const int krnl_dim, const int n_point, const DTYPE *coord, const DTYPE *x, DTYPE *y
)
{
    int n_point_ext = (n_point + SIMD_LEN - 1) / SIMD_LEN * SIMD_LEN;
    int n_vec_ext = n_point_ext / SIMD_LEN;
    DTYPE *coord_ext = (DTYPE*) malloc_aligned(sizeof(DTYPE) * pt_dim * n_point_ext, 64);
    DTYPE *x_ext = (DTYPE*) malloc_aligned(sizeof(DTYPE) * krnl_dim * n_point_ext, 64);
    DTYPE *y_ext = (DTYPE*) malloc_aligned(sizeof(DTYPE) * krnl_dim * n_point_ext, 64);
    
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
    
    int blk_size = 256;
    int mat_size = blk_size * blk_size * krnl_dim * krnl_dim;
    int nthreads = omp_get_max_threads();
    DTYPE *buff  = (DTYPE*) malloc_aligned(sizeof(DTYPE) * nthreads * mat_size, 64);

    double st = get_wtime_sec();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_vec, n_vec;
        calc_block_spos_len(n_vec_ext, nthreads, tid, &start_vec, &n_vec);
        int start_point = start_vec * SIMD_LEN;
        int n_point1 = n_vec * SIMD_LEN;
        DTYPE *thread_buff = buff + tid * mat_size;
        
        for (int ix = start_point; ix < start_point + n_point1; ix += blk_size)
        {
            int nx = (ix + blk_size > start_point + n_point1) ? (start_point + n_point1 - ix) : blk_size;
            for (int iy = 0; iy < n_point_ext; iy += blk_size)
            {
                int ny = (iy + blk_size > n_point_ext) ? (n_point_ext - iy) : blk_size;
                DTYPE *x_in  = x_ext + iy * krnl_dim;
                DTYPE *x_out = y_ext + ix * krnl_dim;
                krnl_eval(
                    coord_ext + ix, n_point_ext, nx,
                    coord_ext + iy, n_point_ext, ny,
                    krnl_param, thread_buff, blk_size*krnl_dim
                );
                CBLAS_GEMV(
                    CblasRowMajor, CblasNoTrans, nx*krnl_dim, ny*krnl_dim, 
                    1.0, thread_buff, blk_size*krnl_dim, 
                    x_in, 1, 1.0, x_out, 1
                );

            }
        }
    }
    double ut = get_wtime_sec() - st;
    free_aligned(buff);
    printf("Direct N-body reference result obtained, %.3lf s\n", ut);
    
    memcpy(y, y_ext, sizeof(DTYPE) * krnl_dim * n_point);
    free_aligned(y_ext);
    free_aligned(x_ext);
    free_aligned(coord_ext);
}

int not_doublematrix(PyArrayObject *mat) {
    if (mat->descr->type_num != PyArray_DOUBLE || mat->nd != 2)
        return 1; 
    else
        return 0;
}