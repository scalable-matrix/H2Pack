
// Generate proxy points for constructing H2 projection and skeleton matrices
// using ID compress for any kernel function. 
// This function is isolated because if the enclosing box for all points are fixed,
// we only need to generate proxy points once and use them repeatedly.
// Input parameters:
//   pt_dim     : Dimension of point coordinate
//   krnl_dim   : Dimension of kernel's return
//   reltol     : Proxy point selection relative error tolerance
//   max_level  : Maximum level (included) of a H2 tree, (root level == 0)
//   min_level  : Minimum level that needs proxy points
//   max_L      : The size of the root node's enclosing box
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   krnl_param : Pointer to kernel function parameter array
// Output parameter:
//   pp_  : Array of proxy points for each level
void H2P_generate_proxy_point_ID(
    const int pt_dim, const int krnl_dim, const DTYPE reltol, const int max_level, const int min_level,
    DTYPE max_L, const void *krnl_param, kernel_eval_fptr krnl_eval, H2P_dense_mat_p **pp_
)
{
    // 1. Initialize proxy point arrays and parameters
    int n_level = max_level + 1;
    H2P_dense_mat_p *pp = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_level);
    ASSERT_PRINTF(pp != NULL, "Failed to allocate %d arrays for storing proxy points", n_level);
    for (int i = 0; i <= max_level; i++) 
    {
        H2P_dense_mat_init(&pp[i], pt_dim, 0);
        pp[i]->ncol = 0;
    }
    
    GET_ENV_INT_VAR(gen_pp_param.alg,          "H2P_GEN_PP_ALG",       "alg",          2,    0,    2);
    GET_ENV_INT_VAR(gen_pp_param.X0_size,      "H2P_GEN_PP_X0_SIZE",   "X0_size",      2000, 500,  5000);
    GET_ENV_INT_VAR(gen_pp_param.Y0_lsize,     "H2P_GEN_PP_Y0_LSIZE",  "Y0_lsize",     4000, 1000, 20000);
    GET_ENV_INT_VAR(gen_pp_param.L3_nlayer,    "H2P_GEN_PP_L3_NLAYER", "L3_nlayer",    8,    8,    32);
    GET_ENV_INT_VAR(gen_pp_param.max_layer,    "H2P_GEN_PP_MAX_LAYER", "max_layer",    8,    4,    32);
    GET_ENV_INT_VAR(gen_pp_param.print_timers, "H2P_PRINT_TIMERS",     "print_timers", 0,    0,    1);

    double timers[4];
    DTYPE L3_nlayer_ = (DTYPE) gen_pp_param.L3_nlayer;

    // 2. Construct proxy points on each level
    DTYPE pow_2_level = 0.5;
    for (int level = 0; level < min_level; level++) pow_2_level *= 2.0;
    for (int level = min_level; level <= max_level; level++)
    {
        // Level 0 and level 1 nodes are not admissible, do not need proxy points
        if (level < 2)
        {
            pow_2_level *= 2.0;
            WARNING_PRINTF("Level %d: no proxy points are generated\n", level);
            continue;
        }

        // Decide box sizes for domains X and Y
        pow_2_level *= 2.0;
        DTYPE L1   = max_L / pow_2_level;
        DTYPE L2   = (1.0 + 2.0 * ALPHA_H2) * L1;
        DTYPE L3_0 = (1.0 + L3_nlayer_ * ALPHA_H2) * L1;
        DTYPE L3_1 = 2.0 * max_L - L1;
        DTYPE L3   = MIN(L3_0, L3_1);

        int Y0_lsize_ = gen_pp_param.Y0_lsize;
        if (gen_pp_param.alg == 0)  // Only one ring, multiple Y0_lsize_ by the number of rings
        {
            int n_layer = DROUND((L3 - L2) / L1);
            if (n_layer > gen_pp_param.max_layer) n_layer = gen_pp_param.max_layer;
            Y0_lsize_ *= n_layer;
        }
        
        // Reset timers
        timers[_GEN_PP_KRNL_T_IDX] = 0.0;
        timers[_GEN_PP_KRNL_T_IDX] = 0.0;
        timers[_GEN_PP_ID_T_IDX]   = 0.0;
        timers[_GEN_PP_MISC_T_IDX] = 0.0;

        // Generate proxy points
        H2P_generate_proxy_point_nlayer(
            pt_dim, krnl_dim, reltol, 
            krnl_param, krnl_eval, 
            L1, L2, L3, 
            gen_pp_param.alg, gen_pp_param.X0_size, Y0_lsize_, gen_pp_param.max_layer, 
            pp[level], &timers[0]
        );
        
        if (gen_pp_param.print_timers == 1)
        {
            INFO_PRINTF("Level %d: %d proxy points generated\n", level, pp[level]->ncol);
            INFO_PRINTF(
                "    kernel, SpMM, ID, other time = %.3lf, %.3lf, %.3lf, %.3lf sec\n", 
                timers[_GEN_PP_KRNL_T_IDX], timers[_GEN_PP_KRNL_T_IDX], 
                timers[_GEN_PP_ID_T_IDX],   timers[_GEN_PP_MISC_T_IDX]
            );
        }
    }  // End of level loop
    
    *pp_ = pp;
}
