#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_matvec.h"
#include "H2Pack_matvec_periodic.h"
#include "H2Pack_utils.h"
#include "utils.h"

// Extend the number of points to a multiple of SIMD_LEN and perform an n-body matvec
// Input parameters:
//   coord0     : Matrix, size dim-by-ld0, coordinates of the 1st point set
//   ld0        : Leading dimension of coord0, should be >= n0
//   n0         : Number of points in coord0 (each column in coord0 is a coordinate)
//   coord1     : Matrix, size dim-by-ld1, coordinates of the 2nd point set
//   ld1        : Leading dimension of coord1, should be >= n1
//   n1         : Number of points in coord1 (each column in coord0 is a coordinate)
//   x_in       : Matrix, size >= krnl_dim * n1, will be left multiplied by kernel_matrix(coord0, coord1)
//   ldi, ldo   : Leading dimensions of x_in and x_out, ldi >= n1, ldo >= n0
//   xpt_dim    : Dimension of extended point coordinate
//   krnl_dim   : Dimension of tensor kernel's return
//   workbuf    : H2P_dense_mat data structure for allocating working buffer
//   krnl_param : Pointer to kernel function parameter array
//   krnl_mv    : Pointer to kernel matrix matvec function
// Output parameter:
//   x_out  : Matrix, size >= krnl_dim * n0, x_out += kernel_matrix(coord0, coord1) * x_in
// Note:
//   For x_{in,out}, they are not stored as the original (n{0,1} * krnl_dim)-by-1 column vector,
//   which can be viewed as n{0,1}-by-krnl_dim matrices. Instead, they are stored as krnl_dim-by-n{0,1}
//   matrices so the krnl_mv can vectorize the load and store. 
void H2P_ext_krnl_mv(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const DTYPE *x_in, const int ldi, DTYPE * __restrict x_out, const int ldo, 
    const int xpt_dim, const int krnl_dim, H2P_dense_mat_p workbuf, 
    const void *krnl_param, kernel_mv_fptr krnl_mv
)
{
    int n0_ext   = (n0 + SIMD_LEN - 1) / SIMD_LEN * SIMD_LEN;
    int n1_ext   = (n1 + SIMD_LEN - 1) / SIMD_LEN * SIMD_LEN;
    int n01_ext  = n0_ext + n1_ext;
    int buf_size = (xpt_dim + krnl_dim) * n01_ext;
    H2P_dense_mat_resize(workbuf, 1, buf_size);
    DTYPE *trg_coord = workbuf->data;
    DTYPE *src_coord = trg_coord + xpt_dim * n0_ext;
    DTYPE *x_in_     = src_coord + xpt_dim * n1_ext;
    DTYPE *x_out_    = x_in_     + n1_ext * krnl_dim;
    
    // Copy coordinates and pad the extend part
    for (int i = 0; i < xpt_dim; i++)
    {
        const DTYPE *c0_src = coord0 + i * ld0;
        const DTYPE *c1_src = coord1 + i * ld1;
        DTYPE *c0_dst = trg_coord + i * n0_ext;
        DTYPE *c1_dst = src_coord + i * n1_ext;
        memcpy(c0_dst, c0_src, sizeof(DTYPE) * n0);
        memcpy(c1_dst, c1_src, sizeof(DTYPE) * n1);
        // Use an extremely large coordinate so the inverse distance of these 
        // extra points to original points are numerically zero
        for (int j = n0; j < n0_ext; j++) c0_dst[j] = 1e100;
        for (int j = n1; j < n1_ext; j++) c1_dst[j] = 1e100;
    }
    
    // Copy input vector and initialize output vector
    // Must set the last n{0,1}_ext - n{0,1} elements in each row to 0,
    // otherwise tensor kernel results might be incorrect
    for (int i = 0; i < krnl_dim; i++)
    {
        const DTYPE *src = x_in + i * ldi;
        DTYPE *dst = x_in_ + i * n1_ext;
        memcpy(dst, src, sizeof(DTYPE) * n1);
        for (int j = n1; j < n1_ext; j++) dst[j] = 0;
    }
    memset(x_out_, 0, sizeof(DTYPE) * n0_ext * krnl_dim);
    
    // Do the n-body bi-matvec
    krnl_mv(
        trg_coord, n0_ext, n0_ext,
        src_coord, n1_ext, n1_ext,
        krnl_param, x_in_, x_out_
    );
    
    // Add results back to original output vectors
    for (int i = 0; i < krnl_dim; i++)
    {
        DTYPE *dst = x_out  + i * ldo;
        DTYPE *src = x_out_ + i * n0_ext;
        #pragma omp simd
        for (int j = 0; j < n0; j++) dst[j] += src[j];
    }
}

// H2 matvec intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
// Need to calculate all B_{ij} matrices before using it
void H2P_matvec_periodic_intmd_mult_JIT(H2Pack_p h2pack, const DTYPE *x, DTYPE *y)
{
    int   pt_dim          = h2pack->pt_dim;
    int   xpt_dim         = h2pack->xpt_dim;
    int   krnl_dim        = h2pack->krnl_dim;
    int   n_point         = h2pack->n_point;
    int   n_node          = h2pack->n_node;
    int   n_thread        = h2pack->n_thread;
    int   *node_level     = h2pack->node_level;
    int   *pt_cluster     = h2pack->pt_cluster;
    int   *mat_cluster    = h2pack->mat_cluster;
    int   *B_nrow         = h2pack->B_nrow;
    int   *B_ncol         = h2pack->B_ncol;
    int   *B_p2i_rowptr   = h2pack->B_p2i_rowptr;
    int   *B_p2i_colidx   = h2pack->B_p2i_colidx;
    int   *B_p2i_val      = h2pack->B_p2i_val;
    DTYPE *coord          = h2pack->coord;
    DTYPE *per_adm_shifts = h2pack->per_adm_shifts;
    void  *krnl_param     = h2pack->krnl_param;
    H2P_dense_mat_p  *y0 = h2pack->y0;
    H2P_dense_mat_p  *J_coord = h2pack->J_coord;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    kernel_mv_fptr   krnl_mv     = h2pack->krnl_mv;
    H2P_thread_buf_p *thread_buf = h2pack->tb;

    H2P_matvec_init_y1(h2pack);
    H2P_dense_mat_p *y1 = h2pack->y1;

    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p Bij      = thread_buf[tid]->mat0;
        H2P_dense_mat_p workbuf  = thread_buf[tid]->mat0;
        H2P_dense_mat_p coord1_s = thread_buf[tid]->mat1;
        DTYPE shift[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        thread_buf[tid]->timer = -get_wtime_sec();

        #pragma omp for schedule(static)
        for (int i = 0; i < n_node; i++)
        {
            if (y1[i]->ld == 0) continue;
            memset(y1[i]->data, 0, sizeof(DTYPE) * y1[i]->ncol);
        }

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int level0 = node_level[node0];
            
            H2P_dense_mat_p y1_0 = y1[node0];
            memset(y1_0->data, 0, sizeof(DTYPE) * y1_0->nrow * y1_0->ncol);

            for (int i = B_p2i_rowptr[node0]; i < B_p2i_rowptr[node0 + 1]; i++)
            {
                int node1    = B_p2i_colidx[i];
                int pair_idx = B_p2i_val[i] - 1;
                int level1   = node_level[node1];
                
                DTYPE *per_adm_shift_i = per_adm_shifts + pair_idx * pt_dim;
                for (int k = 0; k < pt_dim; k++) shift[k] = per_adm_shift_i[k];

                int Bij_nrow  = B_nrow[pair_idx];
                int Bij_ncol  = B_ncol[pair_idx];
                int node0_npt = Bij_nrow / krnl_dim;
                int node1_npt = Bij_ncol / krnl_dim;
                if (krnl_mv == NULL) H2P_dense_mat_resize(Bij, Bij_nrow, Bij_ncol);

                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    H2P_dense_mat_copy(J_coord[node1], coord1_s);
                    H2P_shift_coord(coord1_s, shift, 1.0);
                    if (krnl_mv != NULL)
                    {
                        H2P_ext_krnl_mv(
                            J_coord[node0]->data, J_coord[node0]->ld, J_coord[node0]->ncol,
                            coord1_s->data,       coord1_s->ld,       coord1_s->ncol, 
                            y0[node1]->data, node1_npt, y1[node0]->data, node0_npt, 
                            xpt_dim, krnl_dim, workbuf, krnl_param, krnl_mv
                        );
                    } else {
                        krnl_eval(
                            J_coord[node0]->data, J_coord[node0]->ld, J_coord[node0]->ncol,
                            coord1_s->data,       coord1_s->ld,       coord1_s->ncol,
                            krnl_param, Bij->data, Bij->ld
                        );
                        CBLAS_GEMV(
                            CblasRowMajor, CblasNoTrans, Bij_nrow, Bij_ncol, 
                            1.0, Bij->data, Bij->ld, y0[node1]->data, 1, 1.0, y1[node0]->data, 1
                        );
                    }
                }  // End of "if (level0 == level1)"

                // (2) node1 is a leaf node and its level is higher than node0's level, 
                //     only compressed on node0's side, node1's side don't need the 
                //     downward sweep and can directly accumulate result to output vector
                if (level0 > level1)
                {
                    int pt_s1     = pt_cluster[node1 * 2];
                    int node1_npt = pt_cluster[node1 * 2 + 1] - pt_s1 + 1;
                    int vec_s1    = mat_cluster[node1 * 2];
                    
                    H2P_dense_mat_resize(coord1_s, xpt_dim, node1_npt);
                    copy_matrix(sizeof(DTYPE), xpt_dim, node1_npt, coord + pt_s1, n_point, coord1_s->data, coord1_s->ld, 0);
                    H2P_shift_coord(coord1_s, shift, 1.0);
                    if (krnl_mv != NULL)
                    {
                        const DTYPE *x_spos = x + pt_s1;
                        H2P_ext_krnl_mv(
                            J_coord[node0]->data, J_coord[node0]->ld, J_coord[node0]->ncol,
                            coord1_s->data,       coord1_s->ld,       coord1_s->ncol,
                            x_spos, n_point, y1[node0]->data, node0_npt, 
                            xpt_dim, krnl_dim, workbuf, krnl_param, krnl_mv
                        );
                    } else {
                        const DTYPE *x_spos = x + vec_s1;
                        krnl_eval(
                            J_coord[node0]->data, J_coord[node0]->ld, J_coord[node0]->ncol,
                            coord1_s->data,       coord1_s->ld,       coord1_s->ncol,
                            krnl_param, Bij->data, Bij->ld
                        );
                        CBLAS_GEMV(
                            CblasRowMajor, CblasNoTrans, Bij_nrow, Bij_ncol, 
                            1.0, Bij->data, Bij->ld, x_spos, 1, 1.0, y1[node0]->data, 1
                        );
                    }
                }  // End of "if (level0 > level1)"

                // (3) node0 is a leaf node and its level is higher than node1's level, 
                //     only compressed on node1's side, node0's side don't need the 
                //     downward sweep and can directly accumulate result to output vector
                if (level0 < level1)
                {
                    int pt_s0     = pt_cluster[node0 * 2];
                    int node0_npt = pt_cluster[node0 * 2 + 1] - pt_s0 + 1;
                    int vec_s0    = mat_cluster[node0 * 2];

                    H2P_dense_mat_copy(J_coord[node1], coord1_s);
                    H2P_shift_coord(coord1_s, shift, 1.0);
                    if (krnl_mv != NULL)
                    {
                        DTYPE *y_spos = y + pt_s0;
                        H2P_ext_krnl_mv(
                            coord + pt_s0,  n_point,      node0_npt,
                            coord1_s->data, coord1_s->ld, coord1_s->ncol,
                            y0[node1]->data, node1_npt, y_spos, n_point, 
                            xpt_dim, krnl_dim, workbuf, krnl_param, krnl_mv
                        );
                    } else {
                        DTYPE *y_spos = y + vec_s0;
                        krnl_eval(
                            coord + pt_s0,  n_point,      node0_npt,
                            coord1_s->data, coord1_s->ld, coord1_s->ncol,
                            krnl_param, Bij->data, Bij->ld
                        );
                        CBLAS_GEMV(
                            CblasRowMajor, CblasNoTrans, Bij_nrow, Bij_ncol, 
                            1.0, Bij->data, Bij->ld, y0[node1]->data, 1, 1.0, y_spos, 1
                        );
                    }
                }  // End of "if (level0 < level1)"
            }  // End of node1 loop
        }  // End of node0 loop
        thread_buf[tid]->timer += get_wtime_sec();
    }  // End of "#pragma omp parallel"
    
    if (h2pack->print_timers == 1)
    {
        double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
        for (int i = 0; i < n_thread; i++)
        {
            double thread_i_timer = thread_buf[i]->timer;
            avg_t += thread_i_timer;
            max_t = MAX(max_t, thread_i_timer);
            min_t = MIN(min_t, thread_i_timer);
        }
        avg_t /= (double) n_thread;
        INFO_PRINTF("Matvec intermediate multiplication: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    }
}

// H2 matvec dense multiplication, calculate D_{ij} * x_j
// Need to calculate all D_{ij} matrices before using it
void H2P_matvec_periodic_dense_mult_JIT(H2Pack_p h2pack, const DTYPE *x, DTYPE *y)
{
    int   pt_dim            = h2pack->pt_dim;
    int   xpt_dim           = h2pack->xpt_dim;
    int   krnl_dim          = h2pack->krnl_dim;
    int   n_point           = h2pack->n_point;
    int   n_node            = h2pack->n_node;
    int   n_leaf_node       = h2pack->n_leaf_node;
    int   n_thread          = h2pack->n_thread;
    int   *pt_cluster       = h2pack->pt_cluster;
    int   *mat_cluster      = h2pack->mat_cluster;
    int   *D_nrow           = h2pack->D_nrow;
    int   *D_ncol           = h2pack->D_ncol;
    int   *D_p2i_rowptr     = h2pack->D_p2i_rowptr;
    int   *D_p2i_colidx     = h2pack->D_p2i_colidx;
    int   *D_p2i_val        = h2pack->D_p2i_val;
    DTYPE *coord            = h2pack->coord;
    DTYPE *per_inadm_shifts = h2pack->per_inadm_shifts;
    void  *krnl_param       = h2pack->krnl_param;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;
    kernel_mv_fptr   krnl_mv     = h2pack->krnl_mv;
    H2P_thread_buf_p *thread_buf = h2pack->tb;

    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p Dij      = thread_buf[tid]->mat0;
        H2P_dense_mat_p workbuf  = thread_buf[tid]->mat0;
        H2P_dense_mat_p coord1_s = thread_buf[tid]->mat1;

        DTYPE shift[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        
        thread_buf[tid]->timer = -get_wtime_sec();

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int pt_s0     = pt_cluster[2 * node0];
            int node0_npt = pt_cluster[2 * node0 + 1] - pt_s0 + 1;
            int y_offset  = (krnl_mv == NULL) ? mat_cluster[node0 * 2] : pt_cluster[node0 * 2];
            DTYPE *y_spos = y + y_offset;

            for (int i = D_p2i_rowptr[node0]; i < D_p2i_rowptr[node0 + 1]; i++)
            {
                int node1     = D_p2i_colidx[i];
                int pair_idx  = D_p2i_val[i] - 1;
                int pt_s1     = pt_cluster[2 * node1];
                int node1_npt = pt_cluster[2 * node1 + 1] - pt_s1 + 1;
                int x_offset  = (krnl_mv == NULL) ? mat_cluster[node1 * 2] : pt_cluster[node1 * 2];
                const DTYPE *x_spos = x + x_offset;

                int Dij_nrow = D_nrow[pair_idx];
                int Dij_ncol = D_ncol[pair_idx];
                if (krnl_mv == NULL) H2P_dense_mat_resize(Dij, Dij_nrow, Dij_ncol);

                if (pair_idx < n_leaf_node)
                {
                    // (i, i) pair, no shift
                    for (int k = 0; k < pt_dim; k++) shift[k] = 0.0;
                } else {
                    // The (pair_idx - n_leaf_node)-th inadmissible pair, need shifting
                    DTYPE *per_inadm_shift_i = per_inadm_shifts + (pair_idx - n_leaf_node) * pt_dim;
                    for (int k = 0; k < pt_dim; k++) shift[k] = per_inadm_shift_i[k];
                }

                H2P_dense_mat_resize(coord1_s, xpt_dim, node1_npt);
                copy_matrix(sizeof(DTYPE), xpt_dim, node1_npt, coord + pt_s1, n_point, coord1_s->data, coord1_s->ld, 0);
                H2P_shift_coord(coord1_s, shift, 1.0);
                
                if (krnl_mv != NULL)
                {
                    H2P_ext_krnl_mv(
                        coord + pt_s0,  n_point,      node0_npt,
                        coord1_s->data, coord1_s->ld, coord1_s->ncol,
                        x_spos, n_point, y_spos, n_point, 
                        xpt_dim, krnl_dim, workbuf, krnl_param, krnl_mv
                    );
                } else {
                    krnl_eval(
                        coord + pt_s0,  n_point,      node0_npt,
                        coord1_s->data, coord1_s->ld, coord1_s->ncol,
                        krnl_param, Dij->data, Dij->ld
                    );
                    CBLAS_GEMV(
                        CblasRowMajor, CblasNoTrans, Dij_nrow, Dij_ncol, 
                        1.0, Dij->data, Dij->ld, x_spos, 1, 1.0, y_spos, 1
                    );
                }
            }  // End of i loop
        }  // End of node0 loop
        thread_buf[tid]->timer += get_wtime_sec();
    }  // End of "pragma omp parallel"
    
    if (h2pack->print_timers == 1)
    {
        double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
        for (int i = 0; i < n_thread; i++)
        {
            double thread_i_timer = thread_buf[i]->timer;
            avg_t += thread_i_timer;
            max_t = MAX(max_t, thread_i_timer);
            min_t = MIN(min_t, thread_i_timer);
        }
        avg_t /= (double) n_thread;
        INFO_PRINTF("Matvec dense multiplication: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    }
}

// H2 representation multiplies a column vector
void H2P_matvec_periodic(H2Pack_p h2pack, const DTYPE *x, DTYPE *y)
{
    double st, et;
    int    krnl_mat_size = h2pack->krnl_mat_size;
    int    n_thread      = h2pack->n_thread;
    int    BD_JIT        = h2pack->BD_JIT;
    int    krnl_dim      = h2pack->krnl_dim;
    int    n_point       = h2pack->n_point;
    int    need_trans    = ((h2pack->krnl_mv != NULL) && (BD_JIT == 1) && (krnl_dim > 1));
    DTYPE  *xT           = h2pack->xT;
    DTYPE  *yT           = h2pack->yT;
    DTYPE  *pmt_x        = h2pack->pmt_x;
    DTYPE  *pmt_y        = h2pack->pmt_y;
    double *timers       = h2pack->timers;
    size_t *mat_size     = h2pack->mat_size;
    H2P_thread_buf_p *thread_buf = h2pack->tb;

    DTYPE *x_ = need_trans ? xT : pmt_x;
    DTYPE *y_ = need_trans ? yT : pmt_y;

    if (BD_JIT != 1)
    {
        ERROR_PRINTF("Only support BD_JIT=1 in this function for the moment.\n");
        return;
    }

    // 1. Forward permute the input vector
    st = get_wtime_sec();
    H2P_permute_vector_forward(h2pack, x, pmt_x);
    et = get_wtime_sec();
    timers[MV_VOP_TIMER_IDX] += et - st;
    mat_size[MV_VOP_SIZE_IDX] += 2 * krnl_mat_size;

    // 2. Reset y result to 0 and transpose x if necessary
    st = get_wtime_sec();
    #pragma omp parallel for simd
    for (int i = 0; i < krnl_mat_size; i++)
    {
        pmt_y[i] = 0.0;
        yT[i] = 0.0;
    }
    mat_size[MV_VOP_SIZE_IDX] += 2 * krnl_mat_size;
    if (need_trans) 
    {
        H2P_transpose_dmat(n_thread, n_point, krnl_dim, pmt_x, krnl_dim, xT, n_point);
        mat_size[MV_VOP_SIZE_IDX] += 2 * krnl_mat_size;
    }
    et = get_wtime_sec();
    timers[MV_VOP_TIMER_IDX] += et - st;

    // 3. Forward transformation, calculate U_j^T * x_j
    st = get_wtime_sec();
    H2P_matvec_fwd_transform(h2pack, pmt_x);
    et = get_wtime_sec();
    timers[MV_FWD_TIMER_IDX] += et - st;
    
    // 4. Intermediate multiplication, calculate B_{ij} * (U_j^T * x_j)
    st = get_wtime_sec();
    if (BD_JIT == 1)
    {
        if (need_trans) H2P_transpose_y0_from_krnldim(h2pack);
        H2P_matvec_periodic_intmd_mult_JIT(h2pack, x_, y_);
        if (need_trans) H2P_transpose_y1_to_krnldim(h2pack);
    } else {
        // "Lost butterfly"
        ERROR_PRINTF("Only support BD_JIT=1 in this function for the moment.\n");
        return;
    }
    // Multiply the periodic block for root node
    // y1{root} = y1{root} + O * y0{root};  % y1{root} should be empty
    int root_idx     = h2pack->root_idx;
    int root_J_npt   = h2pack->J[root_idx]->length;
    int per_blk_size = root_J_npt * krnl_dim;
    H2P_dense_mat_p y0_root = h2pack->y0[root_idx];
    H2P_dense_mat_p y1_root = h2pack->y1[root_idx];
    H2P_dense_mat_resize(y1_root, 1, per_blk_size);
    if (need_trans) 
    {
        H2P_dense_mat_p y0_root_tmp = thread_buf[0]->mat0;
        H2P_dense_mat_resize(y0_root_tmp, root_J_npt, krnl_dim);
        H2P_transpose_dmat(1, krnl_dim, root_J_npt, y0_root->data, root_J_npt, y0_root_tmp->data, krnl_dim);
        memcpy(y0_root->data, y0_root_tmp->data, sizeof(DTYPE) * per_blk_size);
    }
    CBLAS_GEMV(
        CblasRowMajor, CblasNoTrans, per_blk_size, per_blk_size, 
        1.0, h2pack->per_blk, per_blk_size, y0_root->data, 1, 0.0, y1_root->data, 1
    );
    et = get_wtime_sec();
    timers[MV_MID_TIMER_IDX] += et - st;

    // 5. Backward transformation, calculate U_i * (B_{ij} * (U_j^T * x_j))
    st = get_wtime_sec();
    H2P_matvec_bwd_transform(h2pack, pmt_x, pmt_y);
    et = get_wtime_sec();
    timers[MV_BWD_TIMER_IDX] += et - st;

    // 6. Dense multiplication, calculate D_i * x_i
    st = get_wtime_sec();
    if (BD_JIT == 1)
    {
        H2P_matvec_periodic_dense_mult_JIT(h2pack, x_, y_);
    } else {
        // "Lost butterfly"
        ERROR_PRINTF("Only support BD_JIT=1 in this function for the moment.\n");
        return;
    }
    et = get_wtime_sec();
    timers[MV_DEN_TIMER_IDX] += et - st;
    
    // 7. Sum yT partial results into y if needed 
    st = get_wtime_sec();
    // We use xT here to hold the transpose of yT
    if (need_trans)
    {
        H2P_transpose_dmat(n_thread, krnl_dim, n_point, yT, n_point, xT, krnl_dim);
        #pragma omp parallel for simd
        for (int i = 0; i < krnl_mat_size; i++) pmt_y[i] += xT[i];
        mat_size[MV_VOP_SIZE_IDX] += 4 * krnl_mat_size;
    }
    et = get_wtime_sec();
    timers[MV_VOP_TIMER_IDX] += et - st;

    // 8. Backward permute the output vector
    st = get_wtime_sec();
    H2P_permute_vector_backward(h2pack, pmt_y, y);
    et = get_wtime_sec();
    timers[MV_VOP_TIMER_IDX] += et - st;
    mat_size[MV_VOP_SIZE_IDX] += 2 * krnl_mat_size;

    h2pack->n_matvec++;
}

