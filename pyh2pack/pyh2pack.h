//  Parameter structure
typedef struct{
    DTYPE *pts_coord;           //  Coordinates of the points. 
    int    pts_dim;             //  Dimension of the space points lie in.
    int    pts_xdim; 
    int    pts_num;             //  Number of points

    int    krnl_dim;            //  Dimension of the output of the kernel function
    char   krnl_name[16];       //  Name of the kernel function. 
    DTYPE *krnl_param;          //  parameters of the kernel function
    char   krnl_param_descr[512];//  Description of the kernel.
    int    krnl_param_len;       //  Number of parameters
    int    krnl_bimv_flops;      //  Flops of bimv operation.
    kernel_eval_fptr krnl_eval; //  Function pointer for kernel_evaluation
    kernel_bimv_fptr krnl_bimv; //  Function pointer for kernel_bimv
  
    DTYPE rel_tol;              //  Relative error threshold for kernel matrix compression.
    int   BD_JIT;               //  Flag of Just-In-Time matvec mode. 
    int   krnl_mat_size;        //  Size of the defined kernel matrix K(coord, coord)
    
    int   flag_proxysurface;    //  Indicate whether to use proxy surface method instead.   
    char  pp_fname[128];      
} H2Pack_Params;

//  MODULE CLASS
typedef struct{
    PyObject_HEAD
    //  H2 matrix pointer
    H2Pack_p h2mat;
    //  H2 matrix parameters
    H2Pack_Params params;
    //  Indicate whether h2mat has been built
    int flag_setup;
} H2Mat;


const char description_setup[] = 
            "H2Pack function setup(...) involves the following three computation steps:\n\
                1. parse the test information \n\
                2. select the proxy points \n\
                3. construct the H2 matrix representation of the kernel matrix \n\n\
             Input description (keywords : datatype): \n\
                kernel        : string, kernel name (presently support 'Coulomb', 'Matern', 'Gaussian', 'RPY', 'Stokes'); \n\
                krnl_dim      : integer, dimension of the kernel function's output;\n\
                pt_coord      : 2d numpy array, point coordinates. Each row or column stores the coordinate of one point;\n\
                pt_dim        : integer, dimension of the space points lying in (support 1D,2D,and 3D);\n\
                rel_tol       : float, accuracy threshold for the H2 matrix representation;\n\
                (optional) JIT_mode       : 1 or 0, flag for running matvec in JIT mode (JIT mode reduces storage cost but has slower matvec);\n\
                (optional) krnl_param     : 1d numpy array, parameters of the kernel function;\n\
                (optional) proxy_surface  : 1 or 0, flag for using proxy surface points (mainly work for potential kernel;\n\
                (optional) max_leaf_points: integer, the maximum number of points in each leaf node.;\n\
                (optional) max_leaf_size  : double, the maximum edge length of each leaf box.;\n\
                (optional) pp_filename    : string, path to a file that either contains precomputed proxy points or will be written to in the setup process;";

static PyObject *setup(H2Mat *self, PyObject *args, PyObject *keywds);


const char description_h2matvec[] = 
            "H2Pack function matvec(x) efficiently multiplies the kernel matrix with ONE vector\n\
             Input description (no need for keywords): \n\
                x: 1d numpy array, the multiplied vector. should be of the same dimension as the matrix.\
            ";
static PyObject *h2matvec(H2Mat *self, PyObject *args);


const char description_directmatvec[] = 
            "H2Pack function direct_matvec(x) calculates the kernel matrix-vector multiplication directly by evaluating kernel matrix entries dynamically.\n\
             Input description (no need for keywords): \n\
                x: 1d numpy array, the multiplied vector. should be of the same dimension as the matrix.\
            ";
static PyObject *direct_matvec(H2Mat *self, PyObject *args);


const char description_printstat[] = 
            "H2Pack funciton print_statistic() prints out the main information of the constructed H2 matrix representation.";
static PyObject *print_statistic(H2Mat *self, PyObject *args);

const char description_printset[] = 
            "H2Pack funciton print_setting() prints out the main information of H2Pack setting.";
static PyObject *print_setting(H2Mat *self, PyObject *args);

const char description_printkernel[] = 
            "H2Pack funciton print_kernels() lists all the supported kernel functions and their descriptions.";
static PyObject *print_kernels(PyObject *self, PyObject *args);

static PyObject *clean(H2Mat *self);




//  
//  Auxiliary functions 
//

#ifndef SIMD_LEN_D
#define SIMD_LEN_D 4
#endif

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
    printf("Direct N-body reference result for %d source * %d target obtained, %.3lf s\n", n_src_pt, n_dst_pt, ut);
    free(krnl_mat_buffs);
}

int not_doublematrix(PyArrayObject *mat) {
    if (mat->descr->type_num != PyArray_DOUBLE || mat->nd != 2)
        return 1; 
    else
        return 0;
}