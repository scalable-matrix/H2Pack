
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
    //printf("Calculate direct n-body reference results for %d points done\n", n_dst_pt);
    free(krnl_mat_buffs);
}

