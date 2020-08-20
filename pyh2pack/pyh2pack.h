const char description_setup[] = 
            "H2Pack function setup(...) involves the following three computation steps:\n\
                1. parse the test information \n\
                2. select the proxy points \n\
                3. construct the H2 matrix representation of the kernel matrix \n\n\
             Input description (keywords : datatype): \n\
                kernel           : string, kernel name (presently support 'Coulomb', 'Matern', 'Gaussian', 'RPY', 'Stokes'); \n\
                kernel_dimension : integer, dimension of the kernel function's output;\n\
                point_coord      : 2d numpy array, point coordinates. Each row or column stores the coordinate of one point;\n\
                point_dimension  : integer, dimension of the space points lying in (support 1D,2D,and 3D);\n\
                rel_tol          : float, accuracy threshold for the H2 matrix representation;\n\
                (optional) JIT_mode       : 1 or 0, flag for running matvec in JIT mode (JIT mode reduces storage cost but has slower matvec);\n\
                (optional) kernel_param   : 1d numpy array, parameters of the kernel function;\n\
                (optional) proxy_surface  : 1 or 0, flag for using proxy surface points (mainly work for potential kernel;\n\
                (optional) max_leaf_points: integer, the maximum number of points in each leaf node.";    
static PyObject *setup(PyObject *self, PyObject *args, PyObject *keywds);


const char description_h2matvec[] = 
            "H2Pack function matvec(x) efficiently multiplies the kernel matrix with ONE vector\n\
             Input description (no need for keywords): \n\
                x: 1d numpy array, the multiplied vector. should be of the same dimension as the matrix.\
            ";
static PyObject *h2matvec(PyObject *self, PyObject *args);


const char description_directmatvec[] = 
            "H2Pack function direct_matvec(x) calculates the kernel matrix-vector multiplication directly by evaluating kernel matrix entries dynamically.\n\
             Input description (no need for keywords): \n\
                x: 1d numpy array, the multiplied vector. should be of the same dimension as the matrix.\
            ";
static PyObject *direct_matvec(PyObject *self, PyObject *args);


const char description_printstat[] = 
            "H2Pack funciton print_statistic() prints out the main information of the constructed H2 matrix representation.";
static PyObject *print_statistic(PyObject *self, PyObject *args);

const char description_printset[] = 
            "H2Pack funciton print_setting() prints out the main information of H2Pack setting.";
static PyObject *print_setting(PyObject *self, PyObject *args);

const char description_printkernel[] = 
            "H2Pack funciton print_kernels() lists all the supported kernel functions and their descriptions.";
static PyObject *print_kernels(PyObject *self, PyObject *args);

static PyObject *clean(PyObject *self, PyObject *args);




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
    int n_thread = omp_get_max_threads();
    DTYPE *buff  = (DTYPE*) malloc_aligned(sizeof(DTYPE) * n_thread * mat_size, 64);

    double st = get_wtime_sec();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_vec, n_vec;
        calc_block_spos_len(n_vec_ext, n_thread, tid, &start_vec, &n_vec);
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