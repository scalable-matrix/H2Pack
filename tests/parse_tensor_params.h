
struct H2P_test_params
{
    int   pt_dim;
    int   xpt_dim;
    int   krnl_dim;
    int   n_point;
    int   krnl_mat_size;
    int   BD_JIT;
    int   kernel_id;
    int   krnl_bimv_flops;
    void  *krnl_param;
    DTYPE rel_tol;
    DTYPE *coord;
    kernel_eval_fptr krnl_eval;
    kernel_bimv_fptr krnl_bimv;
};
struct H2P_test_params test_params;

DTYPE Stokes_krnl_param[2] = {1.0, 0.1};
DTYPE RPY_krnl_param[1]    = {1.0};

static double pseudo_randn()
{
    double res = 0.0;
    for (int i = 0; i < 12; i++) res += drand48();
    return (res - 6.0) / 12.0;
}

void parse_tensor_params(int argc, char **argv)
{
    test_params.pt_dim   = 3;
    test_params.xpt_dim  = 3;
    test_params.krnl_dim = 3;
    
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

    if (argc < 5)
    {
        printf("Kernel function ID = ");
        scanf("%d", &test_params.kernel_id);
    } else {
        test_params.kernel_id = atoi(argv[4]);
        printf("Kernel function ID = %d\n", test_params.kernel_id);
    }
    switch (test_params.kernel_id)
    {
        case 0: 
        {
            printf("Using 3D Stokes kernel, eta = %.2lf, a = %.2lf\n", Stokes_krnl_param[0], Stokes_krnl_param[1]); 
            break;
        }
        case 1: 
        {
            printf("Using 3D RPY kernel, eta = %.2lf\n", RPY_krnl_param[0]);
            break;
        }
    }
    
    if (test_params.kernel_id == 1) test_params.xpt_dim = 4;
    test_params.coord = (DTYPE*) malloc_aligned(sizeof(DTYPE) * test_params.n_point * test_params.xpt_dim, 64);
    assert(test_params.coord != NULL);
    
    // Note: coordinates need to be stored in column-major style, i.e. test_params.coord 
    // is row-major and each column stores the coordinate of a point. 
    int need_gen = 1;
    if (argc >= 6)
    {
        DTYPE *tmp = (DTYPE*) malloc(sizeof(DTYPE) * test_params.n_point * test_params.xpt_dim);
        if (strstr(argv[5], ".csv") != NULL)
        {
            printf("Reading coordinates from CSV file...");
            FILE *inf = fopen(argv[5], "r");
            for (int i = 0; i < test_params.n_point; i++)
            {
                for (int j = 0; j < test_params.xpt_dim-1; j++) 
                    fscanf(inf, "%lf,", &tmp[i * test_params.xpt_dim + j]);
                fscanf(inf, "%lf\n", &tmp[i * test_params.xpt_dim + test_params.xpt_dim-1]);
            }
            fclose(inf);
            printf(" done.\n");
            need_gen = 0;
        }
        if (strstr(argv[5], ".bin") != NULL)
        {
            printf("Reading coordinates from binary file...");
            FILE *inf = fopen(argv[5], "rb");
            fread(tmp, sizeof(DTYPE), test_params.n_point * test_params.xpt_dim, inf);
            fclose(inf);
            printf(" done.\n");
            need_gen = 0;
        }
        if (need_gen == 0)
        {
            for (int i = 0; i < test_params.xpt_dim; i++)
                for (int j = 0; j < test_params.n_point; j++)
                    test_params.coord[i * test_params.n_point + j] = tmp[j * test_params.xpt_dim + i];
        }
        free(tmp);
    }
    if (need_gen == 1)
    {
        DTYPE vol_frac = 0.1;
        DTYPE base = 4.0 / 3.0 * M_PI / vol_frac * (DTYPE) test_params.n_point;
        DTYPE expn = 1.0 / (DTYPE) test_params.pt_dim;
        DTYPE prefac = DPOW(base, expn);
        printf("Binary/CSV coordinate file not provided. Generating random coordinates in unit box...");
        if (test_params.kernel_id == 1)
        {
            DTYPE *x = test_params.coord;
            DTYPE *y = test_params.coord + test_params.n_point;
            DTYPE *z = test_params.coord + test_params.n_point * 2;
            DTYPE *a = test_params.coord + test_params.n_point * 3;
            DTYPE sum_a3 = 0.0;
            for (int i = 0; i < test_params.n_point; i++)
            {
                a[i] = 0.5 + 5.0 * (DTYPE) drand48();
                sum_a3 += a[i] * a[i] * a[i];
            }
            base = 4.0 / 3.0 * M_PI * sum_a3 / vol_frac;
            prefac = DPOW(base, expn);
            for (int i = 0; i < test_params.n_point; i++)
            {
                x[i] = (DTYPE) drand48() * prefac;
                y[i] = (DTYPE) drand48() * prefac;
                z[i] = (DTYPE) drand48() * prefac;
            }
        } else {
            for (int i = 0; i < test_params.n_point * test_params.pt_dim; i++)
            {
                //test_params.coord[i] = (DTYPE) pseudo_randn();
                test_params.coord[i] = (DTYPE) drand48();
                test_params.coord[i] *= prefac;
            }
        }
        printf(" done.\n");
    }
    
    switch (test_params.kernel_id)
    {
        case 0: 
        { 
            test_params.krnl_eval       = Stokes_eval_std; 
            test_params.krnl_bimv       = Stokes_krnl_bimv_intrin_d; 
            test_params.krnl_bimv_flops = Stokes_krnl_bimv_flop;
            test_params.krnl_param      = (void*) &Stokes_krnl_param[0];
            break;
        }
        case 1: 
        {
            test_params.krnl_eval       = RPY_eval_radii; 
            test_params.krnl_bimv       = NULL; 
            test_params.krnl_bimv_flops = RPY_krnl_bimv_flop;
            test_params.krnl_param      = (void*) &RPY_krnl_param[0];
            break;
        }
    }
}

