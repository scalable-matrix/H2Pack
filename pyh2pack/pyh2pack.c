#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include "H2Pack.h"
#include "pyh2pack.h"
#include "pyh2pack_kernel.h"
#include "structmember.h"


//  MODULE CLASS management methods
static void H2Mat_dealloc(H2Mat *self)
{   
    if (self->flag_setup)
        clean(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int H2Mat_init(H2Mat *self, PyObject *args, PyObject *kwds)
{
    if (setup(self, args, kwds))
        return 0;
    else 
        return -1;
}

//  MODULE CLASS methods
static PyMethodDef H2Mat_methods[] = {
    {"setup", (PyCFunction) setup, METH_VARARGS|METH_KEYWORDS, description_setup},
    {"h2matvec", h2matvec, METH_VARARGS, description_h2matvec},
    {"h2matmul", h2matmul, METH_VARARGS, description_h2matmul},
    {"direct_matvec", direct_matvec, METH_VARARGS, description_directmatvec},
    {"print_statistic", print_statistic, METH_VARARGS, description_printstat},
    {"print_setting", print_setting, METH_VARARGS, description_printset},
    {"clean", clean, METH_VARARGS, "Reset pyh2pack and free allocated memories."},
    {NULL, NULL, 0, NULL}
};

//  MODULE CLASS definition
static PyTypeObject H2PackType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyh2pack.H2Mat",
    .tp_doc = "H2 matrix data structure in Pyh2pack",
    .tp_basicsize = sizeof(H2Mat),
    .tp_itemsize = 0,
    .tp_init = (initproc) H2Mat_init,
    .tp_dealloc = (destructor) H2Mat_dealloc,
    .tp_methods = H2Mat_methods,
    .tp_new = PyType_GenericNew
};


//  MODULE methods 
static PyMethodDef PyH2Pack_methods[] = {
    {"print_kernels", print_kernels, METH_VARARGS, description_printkernel},
    {NULL, NULL, 0, NULL}
};

//  MODULE definition
static PyModuleDef PyH2Pack_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pyh2pack",
    .m_doc = "Python Interface for H2Pack: A fast summation method for kernel matrices.",
    .m_size = -1,
    .m_methods= PyH2Pack_methods
};

//  MODULE initialization
PyMODINIT_FUNC PyInit_pyh2pack(void) {
    PyObject *tmp;
    if (PyType_Ready(&H2PackType) < 0)
        return NULL;

    import_array();
    tmp = PyModule_Create(&PyH2Pack_Module);
    if (tmp == NULL)
        return NULL;

    Py_INCREF(&H2PackType);
    if (PyModule_AddObject(tmp, "H2Mat", (PyObject *) &H2PackType) < 0) {
        Py_DECREF(&H2PackType);
        Py_DECREF(tmp);
        return NULL;
    }
    return tmp;
}


/********       Implementation of member functions      ********/
static PyObject *setup(H2Mat *self, PyObject *args, PyObject *keywds) {
    PyArrayObject *coord = NULL, *kernel_param = NULL;
    char *krnl_name_in;                 //  kernel name
    char *pp_fname = NULL;
    double st, et;                      //  timing variable.
    int flag_proxysurface_in = -1;      //  flag for the use of proxy surface points
    int max_leaf_points = 0;
    double max_leaf_size = 0.0;
    
    //  Step 1: check whether h2mat has been setup before. 
    if (self->flag_setup)
    {
        free_aligned(self->params.pts_coord);
        free(self->params.krnl_param);
        H2P_destroy(&self->h2mat);
        self->flag_setup = 0;
    }
    
    //  Step 2: parse input variables
    static char *keywdslist[] = {"kernel", "krnl_dim", "pt_coord", "pt_dim", 
                                "rel_tol", "JIT_mode", "krnl_param",  "proxy_surface", 
                                "pp_filename", "max_leaf_pts", "max_leaf_size", NULL};
    self->params.BD_JIT = 1;
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "siO!id|iO!isid", keywdslist,
                                    &krnl_name_in, &(self->params.krnl_dim), &PyArray_Type, &coord, &(self->params.pts_dim), 
                                    &(self->params.rel_tol), &(self->params.BD_JIT), &PyArray_Type, &kernel_param, &flag_proxysurface_in, 
                                    &pp_fname, &max_leaf_points, &max_leaf_size))
    {
        PyErr_SetString(PyExc_TypeError, 
            "Error in the input arguments for initializing an h2pack structure. Refer to help(pyh2pack.setup) for detailed description of the function inputs"
            );
        return NULL;
    }

    //  Step 2.1: process the information of points
    if (coord == NULL) 
    {
        PyErr_SetString(PyExc_ValueError, "Provided point coordinates are not valid.\n");
        return NULL;
    }
    if (not_doublematrix(coord)) 
    {
        PyErr_SetString(PyExc_ValueError, "Provided point coordinates are not valid.\n");
        return NULL;
    }
    if (self->params.pts_dim != coord->dimensions[0] && self->params.pts_dim != coord->dimensions[1]) 
    {
        PyErr_Format(PyExc_ValueError, "Provided point coordinates are of dimension %ld * %ld, not matching the specified point dimension %ld.\n", 
                        coord->dimensions[0], coord->dimensions[1], self->params.pts_dim);
        return NULL;
    }
    if (self->params.pts_dim != 1 && self->params.pts_dim != 2 && self->params.pts_dim != 3)    
    {
        PyErr_SetString(PyExc_ValueError, "Pyh2pack presently only supports 1D, 2D, and 3D problems. \n");
        return NULL;
    }
    self->params.pts_coord = (DTYPE*) malloc_aligned(sizeof(DTYPE) * coord->dimensions[0] * coord->dimensions[1], 64);
    //  self->params.pts_coord is stored row-wisely and each column corresponds to one point.
    if (self->params.pts_dim == coord->dimensions[0])
    {
        self->params.pts_num = coord->dimensions[1];
        for (int i = 0; i < self->params.pts_dim; i++)
            for (int j = 0; j < self->params.pts_num; j++)
            {
                double *tmp = (double*)PyArray_GETPTR2(coord, i, j);
                self->params.pts_coord[i * self->params.pts_num + j] = *tmp;
            }
    }
    else // when test_params.pt_dim == coord->dimensions[1]
    {
        self->params.pts_num = coord->dimensions[0];
        for (int i = 0; i < self->params.pts_dim; i++)
            for (int j = 0; j < self->params.pts_num; j++)
            {
                double *tmp = (double*)PyArray_GETPTR2(coord, j, i);
                self->params.pts_coord[i * self->params.pts_num + j] = *tmp;
            }
    }
    self->params.krnl_mat_size = self->params.pts_num * self->params.krnl_dim;


    //   Step 2.2: Process the information of kernel function.
    H2P_kernel kernel_info = identify_kernel_info(krnl_name_in, self->params.krnl_dim, self->params.pts_dim);
    if (kernel_info.krnl_name == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Pyh2pack presently does not suppor the specified kernel\n");
        free_aligned(self->params.pts_coord);
        return NULL;
    }

    //   A special treatment of the RPY kernel with 4th dimension being the radius.
    if (strcmp(krnl_name_in, "RPY"))
        self->params.pts_xdim = 4;
    else
        self->params.pts_xdim = self->params.pts_dim;

    snprintf(self->params.krnl_name, sizeof(self->params.krnl_name), "%s", krnl_name_in);
    snprintf(self->params.krnl_param_descr, sizeof(self->params.krnl_param_descr), "%s", kernel_info.param_descr);
    self->params.krnl_eval = kernel_info.krnl_eval;
    self->params.krnl_bimv = kernel_info.krnl_bimv;
    self->params.krnl_bimv_flops = kernel_info.krnl_bimv_flops;
    self->params.krnl_param = (DTYPE*) malloc(sizeof(DTYPE)*kernel_info.krnl_param_len);
    self->params.krnl_param_len = kernel_info.krnl_param_len;
    if (kernel_param == NULL)   //  use default krnl parameters
    {
        for (int i = 0; i < kernel_info.krnl_param_len; i++)
            self->params.krnl_param[i] = kernel_info.krnl_param[i];
    }
    else    //  use specified krnl parameters
    {
        if (kernel_param->nd > 1 || PyArray_SIZE(kernel_param) != kernel_info.krnl_param_len)
        {
            PyErr_Format(PyExc_ValueError, "Wrong kernel parameters. Rule: %s\n", kernel_info.param_descr);
            free_aligned(self->params.pts_coord);
            free(self->params.krnl_param);
            return NULL;
        }
        for (int i = 0; i < kernel_info.krnl_param_len; i++)
            self->params.krnl_param[i] = *((double*)PyArray_GETPTR1(kernel_param, i));
    }

    //  Step 2.3: parameters for partitioning and proxy points.
    max_leaf_points = (max_leaf_points > 0) ? max_leaf_points : 0;
    max_leaf_size = (max_leaf_size > 0) ? max_leaf_size : 0;

    if (flag_proxysurface_in < 0)
        self->params.flag_proxysurface = kernel_info.flag_proxysurface;
    else
        self->params.flag_proxysurface = flag_proxysurface_in;

    if (pp_fname != NULL)
        snprintf(self->params.pp_fname, sizeof(self->params.pp_fname), "%s", pp_fname);
    else
        self->params.pp_fname[0] = '\0';
        
    //  Main Step 1: initialize H2Pack 
    H2P_init(&self->h2mat, self->params.pts_dim, self->params.krnl_dim, QR_REL_NRM, &self->params.rel_tol);

    //  Main Step 2: hierarchical partitioning of points
    H2P_calc_enclosing_box(self->params.pts_dim, self->params.pts_num, self->params.pts_coord, pp_fname, &self->h2mat->root_enbox);
    H2P_partition_points(self->h2mat, self->params.pts_num, self->params.pts_coord, max_leaf_points, max_leaf_size);

    //  Main Step 3: construction of the proxy points
    H2P_dense_mat_p *pp;
    if (self->params.flag_proxysurface == 0)   
    {
        //  Numerical selection of the proxy points
        if (pp_fname == NULL)
        {
            DTYPE max_L = self->h2mat->root_enbox[self->params.pts_dim];
            st = get_wtime_sec();
            H2P_generate_proxy_point_ID(
                self->params.pts_dim, self->params.krnl_dim, self->params.rel_tol, self->h2mat->max_level, 
                self->h2mat->min_adm_level, max_L, self->params.krnl_param, self->params.krnl_eval, &pp
            );
            et = get_wtime_sec();
            PySys_WriteStdout("Step 1: H2Pack generate NUMERICAL proxy points used %.3lf (s)\n", et - st);
        }
        else
        {
            st = get_wtime_sec();
            H2P_generate_proxy_point_ID_file(
                self->h2mat, self->params.krnl_param, self->params.krnl_eval, pp_fname, &pp
            );
            et = get_wtime_sec();
            PySys_WriteStdout("Step 1: H2Pack generate/load NUMERICAL proxy points used %.3lf (s)\n", et - st);
        }
    }
    else    
    {
        //  Proxy surface points
        DTYPE max_L = self->h2mat->root_enbox[self->params.pts_dim];
        int num_pp = ceil(-log10(self->params.rel_tol));
        if (num_pp < 4 ) num_pp = 4;
        if (num_pp > 10) num_pp = 10;
        num_pp = 2 * self->params.pts_dim * num_pp * num_pp;
        st = get_wtime_sec();
        H2P_generate_proxy_point_surface(
            self->params.pts_dim, self->params.pts_xdim, num_pp, self->h2mat->max_level, 
            self->h2mat->min_adm_level, max_L, &pp
        );
        et = get_wtime_sec();
        PySys_WriteStdout("Step 1: H2Pack generate SURFACE proxy points used %.3lf (s)\n", et - st);
    }
    // PySys_WriteStdout("number of proxy points at different layers %d %d\n", pp[2]->ncol, pp[3]->ncol);

    //  Main Step 4: H2 Matrix Construction
    st = get_wtime_sec();
    H2P_build(
        self->h2mat, pp, self->params.BD_JIT, self->params.krnl_param, 
        self->params.krnl_eval, self->params.krnl_bimv, self->params.krnl_bimv_flops
    );
    et = get_wtime_sec();
    PySys_WriteStdout("Step 2: H2Pack constructs the H2 matrix rep. used %.3lf (s)\n", et - st);
 
    //  set "setup" flag and return 
    self->flag_setup = 1;
    Py_RETURN_NONE;
}


static PyObject *h2matvec(H2Mat *self, PyObject *args) {
    if (self->flag_setup == 0)
    {
        PyErr_SetString(PyExc_ValueError, "H2 matrix is not set up yet. Run .setup()");
        return NULL;
    }

    PyArrayObject *x_in, *y_out;
    DTYPE *x, *y;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x_in))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }

    //  Check Input
    if (x_in->descr->type_num != PyArray_DOUBLE || x_in->nd != 1) 
    {
        PyErr_SetString(PyExc_ValueError, "Input of h2matvec should be a one-dimensional float numpy array");
        return NULL; 
    }
    if (x_in->dimensions[0] != self->params.krnl_mat_size)
    {
        PyErr_Format(PyExc_ValueError, "Vector dimension %ld doesn't match kernel matrix dimension %d\n", x_in->dimensions[0], self->params.krnl_mat_size);
        return NULL; 
    }

    //  Copy to the local vector x
    x = (DTYPE*) malloc_aligned(sizeof(DTYPE) * self->params.krnl_mat_size, 64);
    y = (DTYPE*) malloc_aligned(sizeof(DTYPE) * self->params.krnl_mat_size, 64);
    if (x == NULL || y == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Allocate vector storage failed!\n");
        return NULL;
    } 
    for (int i = 0; i < self->params.krnl_mat_size; i++)
        x[i] = *((double*)PyArray_GETPTR1(x_in, i));

    //  Matrix Vector Multiplication
    H2P_matvec(self->h2mat, x, y);

    //  Return the result
    npy_intp dim[] = {self->params.krnl_mat_size};
    y_out = (PyArrayObject *)PyArray_SimpleNew(1, dim, PyArray_DOUBLE);
    for (int i = 0; i < self->params.krnl_mat_size; i++)
        *(double*)PyArray_GETPTR1(y_out, i) = y[i];

    free_aligned(x);
    free_aligned(y);
    return PyArray_Return(y_out);
}


static PyObject *h2matmul(H2Mat *self, PyObject *args) {
    if (self->flag_setup == 0)
    {
        PyErr_SetString(PyExc_ValueError, "H2 matrix is not set up yet. Run .setup()");
        return NULL;
    }

    PyArrayObject *x_in, *y_out;
    DTYPE *x, *y;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x_in))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }

    //  Check Input
    if (x_in->descr->type_num != PyArray_DOUBLE || x_in->nd != 2) 
    {
        PyErr_SetString(PyExc_ValueError, "Input of h2matmul should be a two-dimensional float numpy array");
        return NULL; 
    }
    if (x_in->dimensions[0] != self->params.krnl_mat_size)
    {
        PyErr_Format(PyExc_ValueError, "Vector dimension %ld doesn't match kernel matrix dimension %d\n", x_in->dimensions[0], self->params.krnl_mat_size);
        return NULL; 
    }

    int n_vec = x_in->dimensions[1];
    //  Copy to the local vector x
    x = (DTYPE*) malloc_aligned(sizeof(DTYPE) * self->params.krnl_mat_size * n_vec, 64);
    y = (DTYPE*) malloc_aligned(sizeof(DTYPE) * self->params.krnl_mat_size * n_vec, 64);
    if (x == NULL || y == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Allocate vector storage failed!\n");
        return NULL;
    } 
    for (int i = 0; i < n_vec; i++)
        for (int j = 0; j < self->params.krnl_mat_size; j++)
            x[i * self->params.krnl_mat_size + j] = *((double*)PyArray_GETPTR2(x_in, j, i));


    //  MatMul
    H2P_matmul(self->h2mat, CblasColMajor, n_vec, x, self->params.krnl_mat_size, y, self->params.krnl_mat_size);

    //  Return the result
    npy_intp dim[] = {self->params.krnl_mat_size, n_vec};
    y_out = (PyArrayObject *)PyArray_SimpleNew(2, dim, PyArray_DOUBLE);
    for (int i = 0; i < n_vec; i++)
        for (int j = 0; j < self->params.krnl_mat_size; j++)
            *(double*)PyArray_GETPTR2(y_out, j, i) = y[i * self->params.krnl_mat_size + j];

    free_aligned(x);
    free_aligned(y);
    return PyArray_Return(y_out);
}


static PyObject *direct_matvec(H2Mat *self, PyObject *args) {
    if (self->flag_setup == 0)
    {
        PyErr_SetString(PyExc_ValueError, "The H2 matrix is not set up yet. Run .setup() first.");
        return NULL;
    }

    PyArrayObject *x_in, *y_out;
    DTYPE *x, *y;
    int check_pt_start = 1, check_pt_end = self->params.pts_num;
    if (!PyArg_ParseTuple(args, "O!|ii", &PyArray_Type, &x_in, &check_pt_start, &check_pt_end))
    {
        PyErr_SetString(PyExc_ValueError, "Error in the input argument!");
        return NULL;
    }

    //  Check Input
    if (x_in->descr->type_num != PyArray_DOUBLE || x_in->nd != 1) 
    {
        PyErr_SetString(PyExc_ValueError, "Input of direct_matvec should be a one-dimensional float numpy array");
        return NULL; 
    }
    if (x_in->dimensions[0] != self->params.krnl_mat_size)
    {
        PyErr_Format(PyExc_ValueError, "Vector dimension %ld doesn't match kernel matrix dimension %d\n", x_in->dimensions[0], self->params.krnl_mat_size);
        return NULL; 
    }
    if (check_pt_start < 1 || check_pt_start > self->params.pts_num || 
        check_pt_end < 1 || check_pt_end > self->params.pts_num ||
        check_pt_end < check_pt_start)
    {
        PyErr_Format(PyExc_ValueError, "Range of the target particles is incorrect!\n");
        return NULL; 
    }

    int n_check_pt = check_pt_end - check_pt_start + 1;
    //  Copy to a local vector 
    x = (DTYPE*) malloc_aligned(sizeof(DTYPE) * self->params.krnl_mat_size, 64);
    y = (DTYPE*) malloc_aligned(sizeof(DTYPE) * self->params.krnl_dim * n_check_pt, 64);
    if (x == NULL || y == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Allocate vector storage failed!\n");
        return NULL;
    }
    for (int i = 0; i < self->params.krnl_mat_size; i++)
        x[i] = *((double*)PyArray_GETPTR1(x_in, i));

    //  Matrix Vector Multiplication
    direct_nbody(
        self->params.krnl_param, self->params.krnl_eval, self->params.pts_dim, self->params.krnl_dim, 
        self->params.pts_coord, self->params.pts_num, self->params.pts_num, x,
        self->params.pts_coord + check_pt_start-1, self->params.pts_num, n_check_pt, y
    );

    //  Return the result
    npy_intp dim[] = {self->params.krnl_dim * n_check_pt};
    y_out = (PyArrayObject *)PyArray_SimpleNew(1, dim, PyArray_DOUBLE);
    for (int i = 0; i < self->params.krnl_dim * n_check_pt; i++)
        *(double*)PyArray_GETPTR1(y_out, i) = y[i];

    free_aligned(x);
    free_aligned(y);
    return PyArray_Return(y_out);
}


static PyObject *clean(H2Mat *self) {
    if (self->flag_setup == 1)
    {
        free_aligned(self->params.pts_coord);
        free(self->params.krnl_param);
        H2P_destroy(&(self->h2mat));
        self->flag_setup = 0;
    }
    Py_RETURN_NONE;
}


static PyObject *print_statistic(H2Mat *self, PyObject *args) {
    if (self->flag_setup == 1)
        H2P_print_statistic(self->h2mat);
    else
        PySys_WriteStdout("The H2 matrix representation has not been constructed yet!\n");
    Py_RETURN_NONE;
}

static PyObject *print_setting(H2Mat *self, PyObject *args) {
    if (self->flag_setup == 1)
    {
        printf("==================== Kernel Function Information ====================\n");
        printf("  * Kernel Function Name           : %s\n", self->params.krnl_name);
        printf("  * Dimension of kernel's output   : %d\n", self->params.krnl_dim);
        printf("  * Dimension of input points      : %d\n", self->params.pts_dim);
        printf("  * Number of kernel parameters    : %d\n", self->params.krnl_param_len);
        printf("  * Kernel parameters              : ");
        for (int i = 0; i < self->params.krnl_param_len; i++)
            printf("%f ", self->params.krnl_param[i]);
        printf("\n");
        printf("  * Parameter Description          : %s\n\n", self->params.krnl_param_descr);


        printf("==================== H2Pack SetUp Information ====================\n");
        printf("  * Number of Points               : %d\n", self->params.pts_num);
        printf("  * Compression relative tol       : %e\n", self->params.rel_tol);
        if (self->params.BD_JIT != 0)
            printf("  * JIT_mode                       : on\n");
        else
            printf("  * JIT_mode                       : off\n");
        if (self->params.flag_proxysurface != 0)
            printf("  * Select scheme for proxy points : proxy surface\n");
        else
            printf("  * Select scheme for proxy points : QR\n");
        if (self->params.pp_fname[0] != '\0')
            printf("  * Proxy points file loaded from/written to: %s", self->params.pp_fname);
    }
    else
        PySys_WriteStdout("PyH2pack has not been set up yet!\n");
    Py_RETURN_NONE;
}


static PyObject *print_kernels(PyObject *self, PyObject *args) {
    int num_kernel = sizeof(kernel_list)/sizeof(H2P_kernel);
    printf("==================== Supported kernel functions ====================\n");
    for (int i = 0 ; i < num_kernel; i++)
    {
        const H2P_kernel* ktmp = kernel_list + i;
        printf(" * KERNEL %d: %s\n", i, ktmp->krnl_name);
        printf("    - dimension of kernel output   : %d * %d\n", ktmp->krnl_dim, ktmp->krnl_dim);
        printf("    - dimension of input point     : %d\n", ktmp->pts_dim);
        printf("    - number of kernel parameters  : %d\n", ktmp->krnl_param_len);
        printf("    - default kernel parameters    : ");
        for (int j = 0; j < ktmp->krnl_param_len; j++)
            printf("%f ", ktmp->krnl_param[j]);
        printf("\n");
        printf("    - kernel parameter description : %s\n", ktmp->param_descr);
        if (ktmp->flag_proxysurface != 0)
            printf("    - default proxy point select   : proxy surface\n");
        else
            printf("    - default proxy point select   : QR\n");
    }
    Py_RETURN_NONE;
}




