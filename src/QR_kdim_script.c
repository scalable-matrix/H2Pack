// H2P_partial_pivot_QR operated on column blocks for tensor kernel matrix.
// Each column block has kdim columns. In each step, we swap kdim columns 
// and do kdim Householder orthogonalization.
void H2P_partial_pivot_QR_kdim(
    H2P_dense_mat_t A, const int kdim, const int tol_rank, const DTYPE tol_norm, 
    const int rel_norm, int *p, int *r, const int nthreads, DTYPE *QR_buff
)
{
    DTYPE *R = A->data;
    int nrow = A->nrow;
    int ncol = A->ncol;
    int nblk = ncol / kdim;
    int ldR  = A->ld;
    int max_iter = MIN(nrow, ncol) / kdim;
    
    BLAS_SET_NUM_THREADS(nthreads);
    
    DTYPE *blk_norm = QR_buff;
    
    // Initialization of column permutation
    for (int j = 0; j < ncol; j++) p[j] = j;

    // 2-norm of each column
    #pragma omp parallel for if(nthreads > 1) \
    num_threads(nthreads) schedule(static)
    for (int j = 0; j < ncol; j++)
        blk_norm[j] = CBLAS_NRM2(nrow, R + j * ldR);
    
    // 2-norm of each column block (kdim consecutive columns)
    DTYPE norm_p = 0.0;
    int pivot = 0;
    for (int j = 0; j < nblk; j++)
    {
        DTYPE tmp = 0.0;
        for (int k = 0; k < kdim; k++) 
        {
            int idx = kdim * j + k;
            tmp += blk_norm[idx] * blk_norm[idx]; 
        }
        blk_norm[j] = DSQRT(tmp);
        if (blk_norm[j] > norm_p)
        {
            norm_p = blk_norm[j];
            pivot  = j;
        }
    }
    
    // Scale the stopping norm
    int stop_rank   = MIN(max_iter, tol_rank/kdim);
    DTYPE norm_eps  = DSQRT((DTYPE) nrow) * 1e-15;
    DTYPE stop_norm = MAX(norm_eps, tol_norm);
    if (rel_norm) stop_norm *= norm_p;
    
    int rank = -1;
    // Main iteration of Household QR
    for (int i = 0; i < max_iter; i++)
    {   
        // 1. Check the stop criteria
        if ((norm_p < stop_norm) || (i >= stop_rank))
        {
            rank = i * kdim;
            break;
        }
        
        // 2. Swap the column
        if (i != pivot)
        {
            swap_int(p + i * kdim, p + pivot * kdim, kdim);
            swap_DTYPE(blk_norm + i, blk_norm + pivot, 1);
            DTYPE *R_i = R + i * kdim * ldR;
            DTYPE *R_pivot = R + pivot * kdim * ldR;
            swap_DTYPE(R_i, R_pivot, ldR * kdim);       // assume that ldR == nrow?
        }
        
        // Do kdim times of consecutive Householder orthogonalize
        for (int ii = i * kdim; ii < i * kdim + kdim; ii++)
        {
            // 3. Calculate Householder vector
            int h_len    = nrow - ii;
            int h_len_m1 = h_len - 1;
            DTYPE *h_vec = R + ii * ldR + ii;
            DTYPE sign   = (h_vec[0] > 0.0) ? 1.0 : -1.0;
            DTYPE h_norm = CBLAS_NRM2(h_len, h_vec);
            h_vec[0] = h_vec[0] + sign * h_norm;
            DTYPE inv_h_norm = 1.0 / CBLAS_NRM2(h_len, h_vec);
            #pragma omp simd
            for (int j = 0; j < h_len; j++) h_vec[j] *= inv_h_norm;
            
            // 4. & 5. Householder update & column norm update
            DTYPE *R_block = R + (ii + 1) * ldR + ii;
            int R_block_nrow = h_len;
            int R_block_ncol = ncol - ii - 1;
            #pragma omp parallel for if(nthreads > 1) \
            num_threads(nthreads) schedule(guided)
            for (int j = 0; j < R_block_ncol; j++)
            {
                int ji1 = j + ii + 1;
                
                DTYPE *R_block_j = R_block + j * ldR;
                DTYPE h_Rj = 2.0 * CBLAS_DOT(R_block_nrow, h_vec, R_block_j);
                
                // 4. Orthogonalize columns right to the ii-th column
                #pragma omp simd
                for (int k = 0; k < R_block_nrow; k++)
                    R_block_j[k] -= h_Rj * h_vec[k];

                // 5. Update the norm of the block-column ji1-th belongs to.
                int blk_idx = ji1 / kdim;
                if (blk_idx == i)
                    continue
                
                if (blk_norm[blk_idx] < stop_norm)
                {
                    blk_norm[blk_idx] = 0.0;
                    continue;                    
                }

                DTYPE tmp = R_block_j[0] * R_block_j[0];
                tmp = blk_norm[blk_idx] * blk_norm[blk_idx] - tmp;
                if (tmp <= 1e-10)
                {
                    tmp = 0.0;
                    DTYPE tmp1;
                    for (int k = 0; k < kim; k++)
                    {
                        tmp1 = CBLAS_NRM2(h_len_m1, R + (blk_idx*kdim + k) * ldR + ii + 1);
                        tmp += tmp1 * tmp1;
                    }
                    blk_norm[blk_idx] = DSQRT(tmp);
                } else {
                    // Fast update 2-norm when the new column norm is not so small
                    blk_norm[blk_idx] = DSQRT(tmp);
                }
            }
            
            // We don't need h_vec anymore, can overwrite the i-th column of R
            h_vec[0] = -sign * h_norm;
            memset(h_vec + 1, 0, sizeof(DTYPE) * (h_len - 1));
        }
        
        // 6. Find next pivot 
        pivot  = i + 1;
        norm_p = 0.0;
        for (int j = i + 1; j < nblk; j++)
        {
            if (blk_norm[j] > norm_p)
            {
                norm_p = blk_norm[j];
                pivot  = j;
            }
        }
    }
    if (rank == -1) rank = max_iter * kdim;
    
    *r = rank;
}
