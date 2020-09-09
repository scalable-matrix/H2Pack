
// Build H2 projection matrices using proxy points, level by level
void H2P_build_H2_UJ_proxy(H2Pack_p h2pack)
{
    int    pt_dim         = h2pack->pt_dim;
    int    xpt_dim        = h2pack->xpt_dim;
    int    krnl_dim       = h2pack->krnl_dim;
    int    n_node         = h2pack->n_node;
    int    n_leaf_node    = h2pack->n_leaf_node;
    int    n_point        = h2pack->n_point;
    int    n_thread       = h2pack->n_thread;
    int    max_child      = h2pack->max_child;
    int    max_level      = h2pack->max_level;
    int    min_adm_level  = h2pack->min_adm_level;
    int    stop_type      = h2pack->QR_stop_type;
    int    *children      = h2pack->children;
    int    *n_child       = h2pack->n_child;
    int    *node_level    = h2pack->node_level;
    int    *node_height   = h2pack->node_height;
    int    *level_n_node  = h2pack->level_n_node;
    int    *level_nodes   = h2pack->level_nodes;
    int    *leaf_nodes    = h2pack->height_nodes;
    int    *pt_cluster    = h2pack->pt_cluster;
    DTYPE  *coord         = h2pack->coord;
    DTYPE  *enbox         = h2pack->enbox;
    size_t *mat_size      = h2pack->mat_size;
    void   *krnl_param    = h2pack->krnl_param;
    H2P_dense_mat_p  *pp  = h2pack->pp;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    void *stop_param = NULL;
    if (stop_type == QR_RANK) 
        stop_param = &h2pack->QR_stop_rank;
    if ((stop_type == QR_REL_NRM) || (stop_type == QR_ABS_NRM))
        stop_param = &h2pack->QR_stop_tol;
    
    // 1. Allocate U and J
    h2pack->n_UJ = n_node;
    h2pack->U       = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    h2pack->J       = (H2P_int_vec_p*)   malloc(sizeof(H2P_int_vec_p)   * n_node);
    h2pack->J_coord = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    ASSERT_PRINTF(h2pack->U       != NULL, "Failed to allocate %d U matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J       != NULL, "Failed to allocate %d J matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J_coord != NULL, "Failed to allocate %d J_coord auxiliary matrices\n", n_node);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]       = NULL;
        h2pack->J[i]       = NULL;
        h2pack->J_coord[i] = NULL;
    }
    H2P_dense_mat_p *U       = h2pack->U;
    H2P_int_vec_p   *J       = h2pack->J;
    H2P_dense_mat_p *J_coord = h2pack->J_coord;

    double *U_timers = (double*) malloc(sizeof(double) * n_thread * 8);
    
    // 2. Initialize the row indices for leaf nodes: all points in that box
    for (int j = 0; j < n_leaf_node; j++)
    {
        int node = leaf_nodes[j];
        int pt_s = pt_cluster[node * 2];
        int pt_e = pt_cluster[node * 2 + 1];
        int node_npt = pt_e - pt_s + 1;
        H2P_int_vec_init(&J[node], node_npt);
        for (int k = 0; k < node_npt; k++)
            J[node]->data[k] = pt_s + k;
        J[node]->length = node_npt;
        H2P_dense_mat_init(&J_coord[node], xpt_dim, node_npt);
        copy_matrix_block(sizeof(DTYPE), xpt_dim, node_npt, coord + pt_s, n_point, J_coord[node]->data, node_npt);
    }

    // 3. Hierarchical construction level by level. min_adm_level is the 
    //    highest level that still has admissible blocks.
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);
        int level = i;

        // (1) Update row indices associated with clusters at level i
        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            thread_buf[tid]->timer = -get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                if (n_child_node == 0) continue;  // J[node] has already been prepared for leaf node
                int *child_nodes = children + node * max_child;
                int J_child_size = 0;
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    J_child_size += J[i_child_node]->length;
                }
                H2P_int_vec_init(&J[node], J_child_size);
                for (int i_child = 0; i_child < n_child_node; i_child++)
                {
                    int i_child_node = child_nodes[i_child];
                    H2P_int_vec_concatenate(J[node], J[i_child_node]);
                }
            }
            thread_buf[tid]->timer += get_wtime_sec();
        }
        
        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            H2P_int_vec_p   inadm_skel_idx   = thread_buf[tid]->idx0;
            H2P_int_vec_p   sub_idx          = thread_buf[tid]->idx0;
            H2P_int_vec_p   ID_buff          = thread_buf[tid]->idx1;
            H2P_dense_mat_p node_skel_coord  = thread_buf[tid]->mat0;
            H2P_dense_mat_p inadm_skel_coord = thread_buf[tid]->mat1;
            H2P_dense_mat_p A_block          = thread_buf[tid]->mat2;
            H2P_dense_mat_p QR_buff          = thread_buf[tid]->mat1;
            
            double st, et, krnl_t = 0.0, QR_t = 0.0, other_t = 0.0;

            thread_buf[tid]->timer -= get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int height = node_height[node];

                // (2) Gather current node's skeleton points (== all children nodes' skeleton points)
                st = get_wtime_sec();
                H2P_dense_mat_resize(node_skel_coord, xpt_dim, J[node]->length);
                if (height == 0)
                {
                    node_skel_coord = J_coord[node];
                } else {
                    int n_child_node = n_child[node];
                    int *child_nodes = children + node * max_child;
                    int J_child_size = 0;
                    for (int i_child = 0; i_child < n_child_node; i_child++)
                    {
                        int i_child_node = child_nodes[i_child];
                        int src_ld = J_coord[i_child_node]->ncol;
                        int dst_ld = node_skel_coord->ncol;
                        DTYPE *src_mat = J_coord[i_child_node]->data;
                        DTYPE *dst_mat = node_skel_coord->data + J_child_size; 
                        copy_matrix_block(sizeof(DTYPE), xpt_dim, src_ld, src_mat, src_ld, dst_mat, dst_ld);
                        J_child_size += J[i_child_node]->length;
                    }
                }  // End of "if (level == 0)"
                et = get_wtime_sec();
                other_t += et - st;
                
                // (3) Shift current node's skeleton points so their center is at the original point
                st = get_wtime_sec();
                DTYPE *node_box = enbox + node * 2 * pt_dim;
                int node_skel_npt = J[node]->length;
                int node_pp_npt   = pp[level]->ncol;
                for (int k = 0; k < pt_dim; k++)
                {
                    DTYPE box_center_k = node_box[k] + 0.5 * node_box[pt_dim + k];
                    DTYPE *node_skel_coord_k = node_skel_coord->data + k * node_skel_npt;
                    #pragma omp simd
                    for (int l = 0; l < node_skel_npt; l++)
                        node_skel_coord_k[l] -= box_center_k;
                }
                et = get_wtime_sec();
                other_t += et - st;

                // (4) Build the kernel matrix block
                st = get_wtime_sec();
                int A_blk_nrow = node_skel_npt * krnl_dim;
                int A_blk_ncol = node_pp_npt   * krnl_dim;
                H2P_dense_mat_resize(A_block, A_blk_nrow, A_blk_ncol);
                krnl_eval(
                    node_skel_coord->data, node_skel_npt, node_skel_npt,
                    pp[level]->data,       node_pp_npt,   node_pp_npt, 
                    krnl_param, A_block->data, A_block->ld
                );
                et = get_wtime_sec();
                krnl_t += et - st;

                // (5) ID compress 
                // Note: A is transposed in ID compress, be careful when calculating the buffer size
                st = get_wtime_sec();
                if (krnl_dim == 1)
                {
                    H2P_dense_mat_resize(QR_buff, A_block->nrow, 1);
                } else {
                    int QR_buff_size = (2 * krnl_dim + 2) * A_block->ncol + (krnl_dim + 1) * A_block->nrow;
                    H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
                }
                H2P_int_vec_set_capacity(ID_buff, 4 * A_block->nrow);
                H2P_ID_compress(
                    A_block, stop_type, stop_param, &U[node], sub_idx, 
                    1, QR_buff->data, ID_buff->data, krnl_dim
                );
                et = get_wtime_sec();
                QR_t += et - st;
                
                // (6) Choose the skeleton points of this node
                st = get_wtime_sec();
                for (int k = 0; k < sub_idx->length; k++)
                    J[node]->data[k] = J[node]->data[sub_idx->data[k]];
                J[node]->length = sub_idx->length;
                H2P_dense_mat_init(&J_coord[node], xpt_dim, sub_idx->length);
                H2P_gather_matrix_columns(
                    coord, n_point, J_coord[node]->data, J[node]->length, 
                    xpt_dim, J[node]->data, J[node]->length
                );
                et = get_wtime_sec();
                other_t += et - st;
            }  // End of j loop
            thread_buf[tid]->timer += get_wtime_sec();

            double *timers = U_timers + tid * 8;
            timers[_U_BUILD_KRNL_T_IDX]  = krnl_t;
            timers[_U_BUILD_QR_T_IDX]    = QR_t;
            timers[_U_BUILD_OTHER_T_IDX] = other_t;
        }  // End of "pragma omp parallel"
        
        if (h2pack->print_timers == 1)
        {
            double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
            for (int i = 0; i < n_thread_i; i++)
            {
                double thread_i_timer = thread_buf[i]->timer;
                avg_t += thread_i_timer;
                max_t = MAX(max_t, thread_i_timer);
                min_t = MIN(min_t, thread_i_timer);
            }
            avg_t /= (double) n_thread_i;
            INFO_PRINTF("Build U: level %d, %d/%d threads, %d nodes\n", i, n_thread_i, n_thread, level_n_node[i]);
            INFO_PRINTF("    min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
            INFO_PRINTF("Build U subroutine time consumption:\n");
            INFO_PRINTF("    tid, kernel evaluation, ID compress, misc, total\n");
            for (int tid = 0; tid < n_thread_i; tid++)
            {
                double *timers = U_timers + 8 * tid;
                INFO_PRINTF(
                    "    %3d, %6.3lf, %6.3lf, %6.3lf, %6.3lf\n",
                    tid, timers[_U_BUILD_KRNL_T_IDX], timers[_U_BUILD_QR_T_IDX], 
                    timers[_U_BUILD_OTHER_T_IDX], thread_buf[tid]->timer
                );
            }
        }  // End of "if (h2pack->print_timers == 1)"
    }  // End of i loop

    // 3. Initialize other not touched U J & add statistic info
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        if (U[i] == NULL)
        {
            H2P_dense_mat_init(&U[i], 1, 1);
            U[i]->nrow = 0;
            U[i]->ncol = 0;
            U[i]->ld   = 0;
        } else {
            mat_size[_U_SIZE_IDX]      += U[i]->nrow * U[i]->ncol;
            mat_size[_MV_FWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[_MV_FWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
            mat_size[_MV_BWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[_MV_BWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
        }
        if (J[i] == NULL) H2P_int_vec_init(&J[i], 1);
        if (J_coord[i] == NULL)
        {
            H2P_dense_mat_init(&J_coord[i], 1, 1);
            J_coord[i]->nrow = 0;
            J_coord[i]->ncol = 0;
            J_coord[i]->ld   = 0;
        }
        //printf("Node %3d: %d skeleton points\n", i, J[i]->length);
    }
    
    free(U_timers);

    for (int i = 0; i < n_thread; i++)
        H2P_thread_buf_reset(thread_buf[i]);
    BLAS_SET_NUM_THREADS(n_thread);
}
