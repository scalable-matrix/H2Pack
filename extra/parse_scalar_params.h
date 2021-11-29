
struct H2P_test_params
{
    int   pt_dim;
    int   krnl_dim;
    int   n_point;
    int   krnl_mat_size;
    int   BD_JIT;
    int   kernel_id;
    int   krnl_bimv_flops;
    char  pp_fname[128];
    void  *krnl_param;
    DTYPE rel_tol;
    DTYPE *coord;
    kernel_eval_fptr krnl_eval;
    kernel_bimv_fptr krnl_bimv;
};
struct H2P_test_params test_params;

DTYPE Gaussian_krnl_param[1]  = {0.5};
DTYPE Expon_krnl_param[1]     = {0.5};
DTYPE Matern32_krnl_param[1]  = {1.0};
DTYPE Matern52_krnl_param[1]  = {1.0};
DTYPE Quadratic_krnl_param[2] = {1.0, -0.5};

static double pseudo_randn()
{
    double res = 0.0;
    for (int i = 0; i < 12; i++) res += drand48();
    return (res - 6.0) / 12.0;
}

void parse_scalar_params(int argc, char **argv)
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
        scanf(DTYPE_FMTSTR, &test_params.rel_tol);
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

    if (argc < 7)
    {
        printf("Proxy point file   = ");
        scanf("%s", test_params.pp_fname);
    } else {
        strcpy(test_params.pp_fname, argv[6]);
    }
    
    switch (test_params.kernel_id)
    {
        case 0: 
        {
            if (test_params.pt_dim == 3) printf("Using Coulomb kernel : k(x, y) = 1 / |x-y|\n"); 
            if (test_params.pt_dim == 2) printf("Using Laplace kernel : k(x, y) = log(|x-y|)\n"); 
            break;
        }
        case 1: 
        {
            printf("Using Gaussian kernel : k(x, y) = exp(-l * |x-y|^2), ");
            printf("l = %.2lf\n", Gaussian_krnl_param[0]); 
            break;
        }
        case 2: 
        {
            printf("Using Exponential kernel : k(x, y) = exp(-l * |x-y|), ");
            printf("l = %.2lf\n", Expon_krnl_param[0]); 
            break;
        }
        case 3: 
        {
            printf("Using 3/2 Matern kernel : k(x, y) = (1 + l*k) * exp(-l*k), ");
            printf("k = sqrt(3) * |x-y|, l = %.2lf\n", Matern32_krnl_param[0]); 
            break;
        }
        case 4: 
        {
            printf("Using 5/2 Matern kernel : k(x, y) = (1 + l*k + l^2*k^2/3) * exp(-l*k), ");
            printf("k = sqrt(5) * |x-y|, l = %.2lf\n", Matern52_krnl_param[0]); 
            break;
        }
        case 5: 
        {
            printf("Using Quadratic kernel : k(x, y) = (1 + c * |x-y|^2)^a, "); 
            printf("c = %.2lf, a = %.2lf\n", Quadratic_krnl_param[0], Quadratic_krnl_param[1]);
            break;
        }
    }
    
    test_params.coord = (DTYPE*) malloc_aligned(sizeof(DTYPE) * test_params.n_point * test_params.pt_dim, 64);
    assert(test_params.coord != NULL);

    // Note: coordinates need to be stored in column-major style, i.e. test_params.coord 
    // is row-major and each column stores the coordinate of a point. 
    int need_gen = 1;
    if (argc >= 8)
    {
        DTYPE *tmp = (DTYPE*) malloc(sizeof(DTYPE) * test_params.n_point * test_params.pt_dim);
        if (strstr(argv[7], ".csv") != NULL)
        {
            printf("Reading coordinates from CSV file...");
            FILE *inf = fopen(argv[7], "r");
            for (int i = 0; i < test_params.n_point; i++)
            {
                for (int j = 0; j < test_params.pt_dim-1; j++) 
                    fscanf(inf, DTYPE_FMTSTR",", &tmp[i * test_params.pt_dim + j]);
                fscanf(inf, DTYPE_FMTSTR"\n", &tmp[i * test_params.pt_dim + test_params.pt_dim-1]);
            }
            fclose(inf);
            printf(" done.\n");
            need_gen = 0;
        }
        if (strstr(argv[7], ".bin") != NULL)
        {
            printf("Reading coordinates from binary file...");
            FILE *inf = fopen(argv[7], "rb");
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
        DTYPE prefac = DPOW((DTYPE) test_params.n_point, 1.0 / (DTYPE) test_params.pt_dim);
        printf("Binary/CSV coordinate file not provided. Generating random coordinates in unit box...");
        for (int i = 0; i < test_params.n_point * test_params.pt_dim; i++)
        {
            //test_params.coord[i] = (DTYPE) pseudo_randn();
            test_params.coord[i] = (DTYPE) drand48();
            test_params.coord[i] *= prefac;
        }
        printf(" done.\n");
        printf("Random coordinate scaling prefactor = %e\n", prefac);
    }
    
    if (test_params.pt_dim == 3) 
    {
        switch (test_params.kernel_id)
        {
            case 0: 
            { 
                test_params.krnl_eval       = Coulomb_3D_eval_intrin_t; 
                test_params.krnl_bimv       = Coulomb_3D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Coulomb_3D_krnl_bimv_flop;
                test_params.krnl_param      = NULL;
                break;
            }
            case 1: 
            {
                test_params.krnl_eval       = Gaussian_3D_eval_intrin_t; 
                test_params.krnl_bimv       = Gaussian_3D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Gaussian_3D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Gaussian_krnl_param[0];
                break;
            }
            case 2: 
            {
                test_params.krnl_eval       = Expon_3D_eval_intrin_t; 
                test_params.krnl_bimv       = Expon_3D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Expon_3D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Expon_krnl_param[0];
                break;
            }
            case 3: 
            {
                test_params.krnl_eval       = Matern32_3D_eval_intrin_t; 
                test_params.krnl_bimv       = Matern32_3D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Matern32_3D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Matern32_krnl_param[0];
                break;
            }
            case 4: 
            {
                test_params.krnl_eval       = Matern52_3D_eval_intrin_t; 
                test_params.krnl_bimv       = Matern52_3D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Matern52_3D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Matern52_krnl_param[0];
            }
            case 5: 
            {
                test_params.krnl_eval       = Quadratic_3D_eval_intrin_t; 
                test_params.krnl_bimv       = Quadratic_3D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Quadratic_3D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Quadratic_krnl_param[0];
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
                test_params.krnl_eval       = Laplace_2D_eval_intrin_t; 
                test_params.krnl_bimv       = Laplace_2D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Laplace_2D_krnl_bimv_flop;
                test_params.krnl_param      = NULL;
                break;
            }
            case 1: 
            {
                test_params.krnl_eval       = Gaussian_2D_eval_intrin_t; 
                test_params.krnl_bimv       = Gaussian_2D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Gaussian_2D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Gaussian_krnl_param[0];
                break;
            }
            case 2: 
            {
                test_params.krnl_eval       = Expon_2D_eval_intrin_t; 
                test_params.krnl_bimv       = Expon_2D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Expon_2D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Expon_krnl_param[0];
                break;
            }
            case 3: 
            {
                test_params.krnl_eval       = Matern32_2D_eval_intrin_t; 
                test_params.krnl_bimv       = Matern32_2D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Matern32_2D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Matern32_krnl_param[0];
                break;
            }
            case 4: 
            {
                test_params.krnl_eval       = Matern52_2D_eval_intrin_t; 
                test_params.krnl_bimv       = Matern52_2D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Matern52_2D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Matern52_krnl_param[0];
                break;
            }
            case 5: 
            {
                test_params.krnl_eval       = Quadratic_2D_eval_intrin_t; 
                test_params.krnl_bimv       = Quadratic_2D_krnl_bimv_intrin_t; 
                test_params.krnl_bimv_flops = Quadratic_2D_krnl_bimv_flop;
                test_params.krnl_param      = (void*) &Quadratic_krnl_param[0];
                break;
            }
        }
    }
}
