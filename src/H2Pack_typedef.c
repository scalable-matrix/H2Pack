#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"

// Initialize a H2Pack structure
void H2P_init(
    H2Pack_t *h2pack_, const int pt_dim, const int krnl_dim, 
    const int QR_stop_type, void *QR_stop_param
)
{
    H2Pack_t h2pack = (H2Pack_t) malloc(sizeof(struct H2Pack));
    assert(h2pack != NULL);
    
    h2pack->n_thread  = omp_get_max_threads();
    h2pack->pt_dim    = pt_dim;
    h2pack->krnl_dim  = krnl_dim;
    h2pack->max_child = 1 << pt_dim;
    h2pack->n_matvec  = 0;
    h2pack->is_H2ERI  = 0;
    memset(h2pack->mat_size,  0, sizeof(size_t) * 8);
    memset(h2pack->timers,    0, sizeof(double) * 9);
    memset(h2pack->JIT_flops, 0, sizeof(double) * 2);
    
    h2pack->QR_stop_type = QR_stop_type;
    if (QR_stop_type == QR_RANK) 
        memcpy(&h2pack->QR_stop_rank, QR_stop_param, sizeof(int));
    if ((QR_stop_type == QR_REL_NRM) || (QR_stop_type == QR_ABS_NRM))
        memcpy(&h2pack->QR_stop_tol,  QR_stop_param, sizeof(DTYPE));
    
    h2pack->parent        = NULL;
    h2pack->children      = NULL;
    h2pack->pt_cluster    = NULL;
    h2pack->mat_cluster   = NULL;
    h2pack->n_child       = NULL;
    h2pack->node_level    = NULL;
    h2pack->node_height   = NULL;
    h2pack->level_n_node  = NULL;
    h2pack->level_nodes   = NULL;
    h2pack->height_n_node = NULL;
    h2pack->height_nodes  = NULL;
    h2pack->r_inadm_pairs = NULL;
    h2pack->r_adm_pairs   = NULL;
    h2pack->node_n_r_adm  = NULL;
    h2pack->coord_idx     = NULL;
    h2pack->B_nrow        = NULL;
    h2pack->B_ncol        = NULL;
    h2pack->D_nrow        = NULL;
    h2pack->D_ncol        = NULL;
    h2pack->B_ptr         = NULL;
    h2pack->D_ptr         = NULL;
    h2pack->coord         = NULL;
    h2pack->enbox         = NULL;
    h2pack->B_data        = NULL;
    h2pack->D_data        = NULL;
    h2pack->xT            = NULL;
    h2pack->yT            = NULL;
    h2pack->J             = NULL;
    h2pack->J_coord       = NULL;
    h2pack->pp            = NULL;
    h2pack->U             = NULL;
    h2pack->y0            = NULL;
    h2pack->y1            = NULL;
    h2pack->tb            = NULL;
    
    H2P_int_vec_init(&h2pack->B_blk,  h2pack->n_thread * BD_NTASK_THREAD + 5);
    H2P_int_vec_init(&h2pack->D_blk0, h2pack->n_thread * BD_NTASK_THREAD + 5);
    H2P_int_vec_init(&h2pack->D_blk1, h2pack->n_thread * BD_NTASK_THREAD + 5);

    *h2pack_ = h2pack;
}

// Destroy a H2Pack structure
void H2P_destroy(H2Pack_t h2pack)
{
    free(h2pack->parent);
    free(h2pack->children);
    free(h2pack->pt_cluster);
    free(h2pack->mat_cluster);
    free(h2pack->n_child);
    free(h2pack->node_level);
    free(h2pack->node_height);
    free(h2pack->level_n_node);
    free(h2pack->level_nodes);
    free(h2pack->height_n_node);
    free(h2pack->height_nodes);
    free(h2pack->r_inadm_pairs);
    free(h2pack->r_adm_pairs);
    free(h2pack->node_n_r_adm);
    free(h2pack->coord_idx);
    free(h2pack->B_nrow);
    free(h2pack->B_ncol);
    free(h2pack->D_nrow);
    free(h2pack->D_ncol);
    free(h2pack->B_ptr);
    free(h2pack->D_ptr);
    free(h2pack->coord);
    free(h2pack->enbox);
    H2P_free_aligned(h2pack->B_data);
    H2P_free_aligned(h2pack->D_data);
    free(h2pack->xT);
    free(h2pack->yT);
    
    H2P_int_vec_destroy(h2pack->B_blk);
    H2P_int_vec_destroy(h2pack->D_blk0);
    H2P_int_vec_destroy(h2pack->D_blk1);
    
    // If H2Pack is called from H2P-ERI, pp == J == J_coord == NULL
    
    if (h2pack->pp != NULL)
    {
        for (int i = 0; i <= h2pack->max_level; i++)
            H2P_dense_mat_destroy(h2pack->pp[i]);
        free(h2pack->pp);
    }
    
    if (h2pack->J != NULL)
    {
        for (int i = 0; i < h2pack->n_UJ; i++)
            H2P_int_vec_destroy(h2pack->J[i]);
        free(h2pack->J);
    }
    
    if (h2pack->J_coord != NULL)
    {
        for (int i = 0; i < h2pack->n_UJ; i++)
            H2P_dense_mat_destroy(h2pack->J_coord[i]);
        free(h2pack->J_coord);
    }
    
    for (int i = 0; i < h2pack->n_UJ; i++)
        H2P_dense_mat_destroy(h2pack->U[i]);
    free(h2pack->U);
    
    for (int i = 0; i < h2pack->n_node; i++)
    {
        H2P_dense_mat_destroy(h2pack->y0[i]);
        H2P_dense_mat_destroy(h2pack->y1[i]);
    }
    free(h2pack->y0);
    free(h2pack->y1);
    
    for (int i = 0; i < h2pack->n_thread; i++)
        H2P_thread_buf_destroy(h2pack->tb[i]);
    free(h2pack->tb);
}

// Print statistic info of a H2Pack structure
void H2P_print_statistic(H2Pack_t h2pack)
{
    printf("==================== H2Pack H2 tree info ====================\n");
    printf("  * Number of points               : %d\n", h2pack->n_point);
    printf("  * Kernel matrix size (kms)       : %d\n", h2pack->krnl_mat_size);
    printf("  * Maximum points in a leaf node  : %d\n", h2pack->max_leaf_points);
    printf("  * Height of H2 tree              : %d\n", h2pack->max_level+1);
    printf("  * Number of nodes                : %d\n", h2pack->n_node);
    printf("  * Number of nodes on each level  : ");
    for (int i = 0; i < h2pack->max_level; i++) 
        printf("%d, ", h2pack->level_n_node[i]);
    printf("%d\n", h2pack->level_n_node[h2pack->max_level]);
    printf("  * Number of nodes on each height : ");
    for (int i = 0; i < h2pack->max_level; i++) 
        printf("%d, ", h2pack->height_n_node[i]);
    printf("%d\n", h2pack->height_n_node[h2pack->max_level]);
    printf("  * Number of reduced far pairs    : %d\n", h2pack->n_r_adm_pair);
    printf("  * Number of reduced near pairs   : %d\n", h2pack->n_r_inadm_pair);
    
    printf("==================== H2Pack storage info ====================\n");
    double DTYPE_msize = sizeof(DTYPE);
    double U_MB   = (double) h2pack->mat_size[0] * DTYPE_msize / 1048576.0;
    double B_MB   = (double) h2pack->mat_size[1] * DTYPE_msize / 1048576.0;
    double D_MB   = (double) h2pack->mat_size[2] * DTYPE_msize / 1048576.0;
    double uw_MB  = (double) h2pack->mat_size[3] * DTYPE_msize / 1048576.0;
    double mid_MB = (double) h2pack->mat_size[4] * DTYPE_msize / 1048576.0;
    double dw_MB  = (double) h2pack->mat_size[5] * DTYPE_msize / 1048576.0;
    double db_MB  = (double) h2pack->mat_size[6] * DTYPE_msize / 1048576.0;
    double rd_MB  = (double) h2pack->mat_size[7] * DTYPE_msize / 1048576.0;
    double mv_MB  = uw_MB + mid_MB + dw_MB + db_MB;
    double UBD_k  = 0.0;
    UBD_k += (double) h2pack->mat_size[0];
    UBD_k += (double) h2pack->mat_size[1];
    UBD_k += (double) h2pack->mat_size[2];
    UBD_k /= (double) h2pack->krnl_mat_size;
    double y0y1_MB = 0.0, tb_MB = 0.0;
    for (int i = 0; i < h2pack->n_thread; i++)
    {
        H2P_thread_buf_t tbi = h2pack->tb[i];
        double msize0 = (double) tbi->mat0->size     + (double) tbi->mat1->size;
        double msize1 = (double) tbi->idx0->capacity + (double) tbi->idx1->capacity;
        tb_MB += DTYPE_msize * msize0 + (double) sizeof(int) * msize1;
        tb_MB += DTYPE_msize * (double) h2pack->krnl_mat_size;
    }
    for (int i = 0; i < h2pack->n_node; i++)
    {
        H2P_dense_mat_t y0i = h2pack->y0[i];
        H2P_dense_mat_t y1i = h2pack->y1[i];
        y0y1_MB += DTYPE_msize * (double) (y0i->size + y1i->ncol - 1);
        tb_MB   += DTYPE_msize * (double) (y1i->size - y1i->ncol + 1);
    }
    y0y1_MB /= 1048576.0;
    tb_MB   /= 1048576.0;
    printf("  * Just-In-Time B & D build  : %s\n", h2pack->BD_JIT ? "Yes (B & D not allocated)" : "No");
    printf("  * H2 representation U, B, D : %.2lf, %.2lf, %.2lf (MB) \n", U_MB, B_MB, D_MB);
    printf("  * Matvec auxiliary arrays   : %.2lf (MB) \n", y0y1_MB);
    printf("  * Thread-local buffers      : %.2lf (MB) \n", tb_MB);
    //printf("  * sizeof(U + B + D) / kms   : %.3lf \n", UBD_k);
    int max_node_rank = 0;
    double sum_node_rank = 0.0, non_empty_node = 0.0;
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        int rank_i = h2pack->U[i]->ncol;
        if (rank_i > 0)
        {
            sum_node_rank  += (double) rank_i;
            non_empty_node += 1.0;
            max_node_rank   = (rank_i > max_node_rank) ? rank_i : max_node_rank;
        }
    }
    printf("  * Max / Avg compressed rank : %d, %.0lf \n", max_node_rank, sum_node_rank / non_empty_node);
    
    printf("==================== H2Pack timing info =====================\n");
    int n_matvec = h2pack->n_matvec;
    double build_t = 0.0, matvec_t = 0.0;
    double d_n_matvec = (double) h2pack->n_matvec;
    for (int i = 0; i < 4; i++) build_t += h2pack->timers[i];
    for (int i = 4; i < 9; i++) 
    {
        h2pack->timers[i] /= d_n_matvec;
        matvec_t += h2pack->timers[i];
    }
    printf("  * H2 construction time (sec) = %.3lf \n", build_t);
    printf("      |----> Point partition   = %.3lf \n", h2pack->timers[0]);
    printf("      |----> U construction    = %.3lf \n", h2pack->timers[1]);
    printf("      |----> B construction    = %.3lf \n", h2pack->timers[2]);
    printf("      |----> D construction    = %.3lf \n", h2pack->timers[3]);
    printf(
        "  * H2 matvec average time (sec) = %.3lf, %.2lf GB/s\n", 
        matvec_t, mv_MB / matvec_t / 1024.0
    );
    printf(
        "      |----> Upward sweep        = %.3lf, %.2lf GB/s\n", 
        h2pack->timers[4], uw_MB  / h2pack->timers[4] / 1024.0
    );
    if (h2pack->BD_JIT == 0)
    {
        printf(
            "      |----> Intermediate sweep  = %.3lf, %.2lf GB/s\n", 
            h2pack->timers[5], mid_MB / h2pack->timers[5] / 1024.0
        );
    } else {
        double GFLOPS = h2pack->JIT_flops[0] / 1000000000.0;
        printf(
            "      |----> Intermediate sweep  = %.3lf, %.2lf GFLOPS\n", 
            h2pack->timers[5], GFLOPS / h2pack->timers[5]
        );
    }
    printf(
        "      |----> Downward sweep      = %.3lf, %.2lf GB/s\n", 
        h2pack->timers[6], dw_MB  / h2pack->timers[6] / 1024.0
    );
    if (h2pack->BD_JIT == 0)
    {
        printf(
            "      |----> Dense block sweep   = %.3lf, %.2lf GB/s\n", 
            h2pack->timers[7], db_MB  / h2pack->timers[7] / 1024.0
        );
    } else {
        double GFLOPS = h2pack->JIT_flops[1] / 1000000000.0;
        printf(
            "      |----> Dense block sweep   = %.3lf, %.2lf GFLOPS\n", 
            h2pack->timers[7], GFLOPS / h2pack->timers[7]
        );
    }
    printf(
        "      |----> OpenMP reduction    = %.3lf, %.2lf GB/s\n", 
        h2pack->timers[8], rd_MB  / h2pack->timers[8] / 1024.0
    );
    
    printf("=============================================================\n");
}

