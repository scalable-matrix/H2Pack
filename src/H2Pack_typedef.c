#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "utils.h"
#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "DAG_task_queue.h"

// Initialize a H2Pack structure
void H2P_init(
    H2Pack_t *h2pack_, const int pt_dim, const int krnl_dim, 
    const int QR_stop_type, void *QR_stop_param
)
{
    H2Pack_t h2pack = (H2Pack_t) malloc(sizeof(struct H2Pack));
    ASSERT_PRINTF(h2pack != NULL, "Failed to allocate H2Pack structure\n");
    memset(h2pack, 0, sizeof(struct H2Pack));
    
    h2pack->n_thread    = omp_get_max_threads();
    h2pack->pt_dim      = pt_dim;
    h2pack->xpt_dim     = pt_dim;  // By default, we don't have any extended information
    h2pack->krnl_dim    = krnl_dim;
    h2pack->max_child   = 1 << pt_dim;
    h2pack->n_matvec    = 0;
    h2pack->n_ULV_solve = 0;
    h2pack->is_H2ERI    = 0;
    h2pack->is_HSS      = 0;
    memset(h2pack->mat_size,  0, sizeof(size_t) * 11);
    memset(h2pack->timers,    0, sizeof(double) * 11);
    memset(h2pack->JIT_flops, 0, sizeof(double) * 2);
    
    h2pack->max_neighbor = 1;
    for (int i = 0; i < pt_dim; i++) h2pack->max_neighbor *= 3;

    h2pack->QR_stop_type = QR_stop_type;
    if (QR_stop_type == QR_RANK) 
        memcpy(&h2pack->QR_stop_rank, QR_stop_param, sizeof(int));
    if ((QR_stop_type == QR_REL_NRM) || (QR_stop_type == QR_ABS_NRM))
        memcpy(&h2pack->QR_stop_tol,  QR_stop_param, sizeof(DTYPE));
    
    h2pack->n_node              = 0;
    h2pack->parent              = NULL;
    h2pack->children            = NULL;
    h2pack->pt_cluster          = NULL;
    h2pack->mat_cluster         = NULL;
    h2pack->n_child             = NULL;
    h2pack->node_level          = NULL;
    h2pack->node_height         = NULL;
    h2pack->level_n_node        = NULL;
    h2pack->level_nodes         = NULL;
    h2pack->height_n_node       = NULL;
    h2pack->height_nodes        = NULL;
    h2pack->r_inadm_pairs       = NULL;
    h2pack->r_adm_pairs         = NULL;
    h2pack->HSS_r_inadm_pairs   = NULL;
    h2pack->HSS_r_adm_pairs     = NULL;
    h2pack->node_inadm_lists    = NULL;
    h2pack->node_n_r_inadm      = NULL;
    h2pack->node_n_r_adm        = NULL;
    h2pack->coord_idx           = NULL;
    h2pack->B_p2i_rowptr        = NULL;
    h2pack->B_p2i_colidx        = NULL;
    h2pack->B_p2i_val           = NULL;
    h2pack->D_p2i_rowptr        = NULL;
    h2pack->D_p2i_colidx        = NULL;
    h2pack->D_p2i_val           = NULL;
    h2pack->ULV_Ls              = NULL;
    h2pack->ULV_p               = NULL;
    h2pack->B_nrow              = NULL;
    h2pack->B_ncol              = NULL;
    h2pack->D_nrow              = NULL;
    h2pack->D_ncol              = NULL;
    h2pack->B_ptr               = NULL;
    h2pack->D_ptr               = NULL;
    h2pack->coord               = NULL;
    h2pack->enbox               = NULL;
    h2pack->per_lattices        = NULL;
    h2pack->per_adm_shifts      = NULL;
    h2pack->per_inadm_shifts    = NULL;
    h2pack->B_data              = NULL;
    h2pack->D_data              = NULL;
    h2pack->per_blk             = NULL;
    h2pack->xT                  = NULL;
    h2pack->yT                  = NULL;
    h2pack->J                   = NULL;
    h2pack->ULV_idx             = NULL;
    h2pack->J_coord             = NULL;
    h2pack->pp                  = NULL;
    h2pack->U                   = NULL;
    h2pack->y0                  = NULL;
    h2pack->y1                  = NULL;
    h2pack->ULV_Q               = NULL;
    h2pack->ULV_L               = NULL;
    h2pack->tb                  = NULL;
    h2pack->upward_tq           = NULL;
    
    H2P_int_vec_init(&h2pack->B_blk,  h2pack->n_thread * BD_NTASK_THREAD + 5);
    H2P_int_vec_init(&h2pack->D_blk0, h2pack->n_thread * BD_NTASK_THREAD + 5);
    H2P_int_vec_init(&h2pack->D_blk1, h2pack->n_thread * BD_NTASK_THREAD + 5);

    *h2pack_ = h2pack;
}

// Run H2Pack in HSS mode (by default, H2Pack runs in H2 mode)
void H2P_run_HSS(H2Pack_t h2pack) 
{
    if (h2pack == NULL) return;
    h2pack->is_HSS = 1;
    h2pack->is_RPY_Ewald = 0;
}

// Run the RPY kernel in H2Pack
void H2P_run_RPY(H2Pack_t h2pack) 
{
    if (h2pack == NULL) return;
    //if (h2pack->is_HSS == 1) WARNING_PRINTF("Running RPY kernel in HSS mode will be very slow, please consider using H2 mode\n");
    h2pack->is_RPY  = 1;
    h2pack->xpt_dim = h2pack->pt_dim + 1;
    h2pack->is_RPY_Ewald = 0;
}

const DTYPE per_lattices_2d[9 * 2] = {
    -1, -1,
    -1,  0,
    -1,  1,
     0, -1,
     0,  0, 
     0,  1,
     1, -1,
     1,  0, 
     1,  1
};

const DTYPE per_lattices_3d[27 * 3] = {
    -1, -1, -1, 
    -1, -1,  0, 
    -1, -1,  1, 
    -1,  0, -1,
    -1,  0,  0, 
    -1,  0,  1,
    -1,  1, -1,
    -1,  1,  0, 
    -1,  1,  1,
     0, -1, -1, 
     0, -1,  0, 
     0, -1,  1, 
     0,  0, -1,
     0,  0,  0, 
     0,  0,  1,
     0,  1, -1,
     0,  1,  0, 
     0,  1,  1,
     1, -1, -1, 
     1, -1,  0, 
     1, -1,  1, 
     1,  0, -1,
     1,  0,  0, 
     1,  0,  1,
     1,  1, -1,
     1,  1,  0, 
     1,  1,  1
};

// Run the RPY Ewald summation kernel in H2Pack
void H2P_run_RPY_Ewald(H2Pack_t h2pack)
{
    if (h2pack == NULL) return;
    if (h2pack->is_HSS == 1 || h2pack->is_RPY == 1)
    {
        ERROR_PRINTF("RPY_Ewald kernel conflicts with HSS mode / RPY kernel\n");
        return;
    }
    h2pack->is_RPY_Ewald = 1;
    h2pack->is_HSS  = 0;
    h2pack->is_RPY  = 0;
    h2pack->xpt_dim = h2pack->pt_dim + 1;

    if (h2pack->pt_dim == 2)
    {
        h2pack->n_lattice = 9;
        h2pack->per_lattices = (DTYPE*) malloc(sizeof(DTYPE) * 9 * 2);
        memcpy(h2pack->per_lattices, per_lattices_2d, sizeof(DTYPE) * 9 * 2);
    }
    if (h2pack->pt_dim == 3)
    {
        h2pack->n_lattice = 27;
        h2pack->per_lattices = (DTYPE*) malloc(sizeof(DTYPE) * 27 * 3);
        memcpy(h2pack->per_lattices, per_lattices_3d, sizeof(DTYPE) * 27 * 3);
    }
}

// Destroy a H2Pack structure
void H2P_destroy(H2Pack_t h2pack)
{
    if (h2pack == NULL) return;
    
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
    free(h2pack->HSS_r_inadm_pairs);
    free(h2pack->HSS_r_adm_pairs);
    free(h2pack->node_inadm_lists);
    free(h2pack->node_n_r_inadm);
    free(h2pack->node_n_r_adm);
    free(h2pack->coord_idx);
    free(h2pack->B_p2i_rowptr);
    free(h2pack->B_p2i_colidx);
    free(h2pack->B_p2i_val);
    free(h2pack->D_p2i_rowptr);
    free(h2pack->D_p2i_colidx);
    free(h2pack->D_p2i_val);
    free(h2pack->ULV_Ls);
    free(h2pack->B_nrow);
    free(h2pack->B_ncol);
    free(h2pack->D_nrow);
    free(h2pack->D_ncol);
    free(h2pack->B_ptr);
    free(h2pack->D_ptr);
    free(h2pack->coord);
    free(h2pack->enbox);
    free(h2pack->per_lattices);
    free(h2pack->per_adm_shifts);
    free(h2pack->per_inadm_shifts);
    free_aligned(h2pack->B_data);
    free_aligned(h2pack->D_data);
    free_aligned(h2pack->per_blk);
    free(h2pack->xT);
    free(h2pack->yT);
    DAG_task_queue_free(h2pack->upward_tq);
    
    if (h2pack->B_blk  != NULL) H2P_int_vec_destroy(h2pack->B_blk);
    if (h2pack->D_blk0 != NULL) H2P_int_vec_destroy(h2pack->D_blk0);
    if (h2pack->D_blk1 != NULL) H2P_int_vec_destroy(h2pack->D_blk1);
    
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
    
    if (h2pack->ULV_idx != NULL)
    {
        for (int i = 0; i < h2pack->n_node; i++)
            H2P_int_vec_destroy(h2pack->ULV_idx[i]);
        free(h2pack->ULV_idx);
    }

    if (h2pack->ULV_p != NULL)
    {
        for (int i = 0; i < h2pack->n_node; i++)
            H2P_int_vec_destroy(h2pack->ULV_p[i]);
        free(h2pack->ULV_p);
    }

    if (h2pack->J_coord != NULL)
    {
        for (int i = 0; i < h2pack->n_UJ; i++)
            H2P_dense_mat_destroy(h2pack->J_coord[i]);
        free(h2pack->J_coord);
    }
    
    // If we don't run H2P_build, h2pack->U == NULL
    if (h2pack->U != NULL)
    {
        for (int i = 0; i < h2pack->n_UJ; i++)
            H2P_dense_mat_destroy(h2pack->U[i]);
        free(h2pack->U);
    }

    if (h2pack->ULV_Q != NULL)
    {
        for (int i = 0; i < h2pack->n_node; i++)
        {
            H2P_dense_mat_destroy(h2pack->ULV_Q[i]);
            H2P_dense_mat_destroy(h2pack->ULV_L[i]);
        }
        free(h2pack->ULV_Q);
        free(h2pack->ULV_L);
    }
    
    // If we don't run H2P_matvec, h2pack->y0 == h2pack->y1 == NULL
    if (h2pack->y0 != NULL && h2pack->y1 != NULL)
    {
        for (int i = 0; i < h2pack->n_node; i++)
        {
            H2P_dense_mat_destroy(h2pack->y0[i]);
            H2P_dense_mat_destroy(h2pack->y1[i]);
        }
        free(h2pack->y0);
        free(h2pack->y1);
    }
    
    if (h2pack->tb != NULL)
    {
        for (int i = 0; i < h2pack->n_thread; i++)
            H2P_thread_buf_destroy(h2pack->tb[i]);
        free(h2pack->tb);
    }
}

// Print statistic info of a H2Pack structure
void H2P_print_statistic(H2Pack_t h2pack)
{
    if (h2pack == NULL) return;
    if (h2pack->n_node == 0)
    {
        printf("H2Pack has nothing to report yet.\n");
        return;
    }
    
    printf("==================== H2Pack H2 tree info ====================\n");
    printf("  * Number of points               : %d\n", h2pack->n_point);
    printf("  * Kernel matrix size             : %d\n", h2pack->krnl_mat_size);
    printf("  * Maximum points in a leaf node  : %d\n", h2pack->max_leaf_points);
    printf("  * Maximum leaf node box size     : %e\n", h2pack->max_leaf_size);
    printf("  * Number of levels (root at 0)   : %d\n", h2pack->max_level+1);
    printf("  * Number of nodes                : %d\n", h2pack->n_node);
    printf("  * Number of nodes on each level  : ");
    for (int i = 0; i < h2pack->max_level; i++) 
        printf("%d, ", h2pack->level_n_node[i]);
    printf("%d\n", h2pack->level_n_node[h2pack->max_level]);
    printf("  * Number of nodes on each height : ");
    for (int i = 0; i < h2pack->max_level; i++) 
        printf("%d, ", h2pack->height_n_node[i]);
    printf("%d\n", h2pack->height_n_node[h2pack->max_level]);
    printf("  * H2Pack running mode            : ");
    int is_H2 = 1;
    if (h2pack->is_HSS       == 1) { printf("HSS\n");               is_H2 = 0; }
    if (h2pack->is_RPY       == 1) { printf("RPY kernel\n");        is_H2 = 0; }
    if (h2pack->is_RPY_Ewald == 1) { printf("RPY Ewald kernel\n");  is_H2 = 0; }
    if (h2pack->is_H2ERI     == 1) { printf("H2-ERI\n");            is_H2 = 0; }
    if (is_H2 == 1) printf("H2\n");
    printf("  * Minimum admissible pair level  : %d\n", (h2pack->is_HSS == 0) ? h2pack->min_adm_level : h2pack->HSS_min_adm_level);
    printf("  * Number of reduced adm. pairs   : %d\n", h2pack->n_r_adm_pair);
    printf("  * Number of reduced inadm. pairs : %d\n", h2pack->n_r_inadm_pair);
    
    if (h2pack->U == NULL) 
    {
        printf("H2Pack H2 matrix has not been constructed yet.\n");
        return;
    }
    
    printf("==================== H2Pack storage info ====================\n");
    size_t *mat_size = h2pack->mat_size;
    double DTYPE_MB = (double) sizeof(DTYPE) / 1048576.0;
    double int_MB   = (double) sizeof(int)   / 1048576.0;
    double U_MB   = (double) mat_size[_U_SIZE_IDX]      * DTYPE_MB;
    double B_MB   = (double) mat_size[_B_SIZE_IDX]      * DTYPE_MB;
    double D_MB   = (double) mat_size[_D_SIZE_IDX]      * DTYPE_MB;
    double fw_MB  = (double) mat_size[_MV_FW_SIZE_IDX]  * DTYPE_MB;
    double mid_MB = (double) mat_size[_MV_MID_SIZE_IDX] * DTYPE_MB;
    double bw_MB  = (double) mat_size[_MV_BW_SIZE_IDX]  * DTYPE_MB;
    double den_MB = (double) mat_size[_MV_DEN_SIZE_IDX] * DTYPE_MB;
    double rdc_MB = (double) mat_size[_MV_RDC_SIZE_IDX] * DTYPE_MB;
    double mv_MB  = fw_MB + mid_MB + bw_MB + den_MB;
    double UBD_k  = 0.0;
    UBD_k += (double) mat_size[_U_SIZE_IDX];
    UBD_k += (double) mat_size[_B_SIZE_IDX];
    UBD_k += (double) mat_size[_D_SIZE_IDX];
    UBD_k /= (double) h2pack->krnl_mat_size;
    double matvec_MB = 0.0;
    for (int i = 0; i < h2pack->n_thread; i++)
    {
        H2P_thread_buf_t tbi = h2pack->tb[i];
        double msize0 = (double) tbi->mat0->size     + (double) tbi->mat1->size;
        double msize1 = (double) tbi->idx0->capacity + (double) tbi->idx1->capacity;
        matvec_MB += DTYPE_MB * msize0 + int_MB * msize1;
        matvec_MB += DTYPE_MB * (double) h2pack->krnl_mat_size;
    }
    if (h2pack->y0 != NULL && h2pack->y1 != NULL)
    {
        for (int i = 0; i < h2pack->n_node; i++)
        {
            H2P_dense_mat_t y0i = h2pack->y0[i];
            H2P_dense_mat_t y1i = h2pack->y1[i];
            matvec_MB += DTYPE_MB * (y0i->size + y1i->size);
        }
    }
    printf("  * Just-In-Time B & D build      : %s\n", h2pack->BD_JIT ? "Yes (B & D not allocated)" : "No");
    printf("  * H2 representation U, B, D     : %.2lf, %.2lf, %.2lf (MB) \n", U_MB, B_MB, D_MB);
    printf("  * Matvec auxiliary arrays       : %.2lf (MB) \n", matvec_MB);
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
    printf("  * Max / Avg compressed rank     : %d, %.0lf \n", max_node_rank, sum_node_rank / non_empty_node);

    if (h2pack->is_HSS == 1)
    {
        double ULV_Q_MB = (double) mat_size[_ULV_Q_SIZE_IDX] * DTYPE_MB;
        double ULV_L_MB = (double) mat_size[_ULV_L_SIZE_IDX] * DTYPE_MB;
        double ULV_I_MB = (double) mat_size[_ULV_I_SIZE_IDX] * DTYPE_MB;
        printf("  * HSS ULV factorization Q, L, I : %.2lf, %.2lf, %.2lf (MB) \n", ULV_Q_MB, ULV_L_MB, ULV_I_MB);
    }
    
    printf("==================== H2Pack timing info =====================\n");
    double *timers = h2pack->timers;
    double build_t = 0.0, matvec_t = 0.0;
    double d_n_matvec = (double) h2pack->n_matvec;
    build_t += timers[_PT_TIMER_IDX];
    build_t += timers[_U_BUILD_TIMER_IDX];
    build_t += timers[_B_BUILD_TIMER_IDX];
    build_t += timers[_D_BUILD_TIMER_IDX];
    printf("  * H2 construction time (sec)   = %.3lf \n", build_t);
    printf("      |----> Point partition     = %.3lf \n", timers[_PT_TIMER_IDX]);
    printf("      |----> U construction      = %.3lf \n", timers[_U_BUILD_TIMER_IDX]);
    printf("      |----> B construction      = %.3lf \n", timers[_B_BUILD_TIMER_IDX]);
    printf("      |----> D construction      = %.3lf \n", timers[_D_BUILD_TIMER_IDX]);

    if (h2pack->n_matvec == 0)
    {
        printf("H2Pack does not have matvec timings results yet.\n");
    } else {
        double fw_t  = timers[_MV_FW_TIMER_IDX]  / d_n_matvec;
        double mid_t = timers[_MV_MID_TIMER_IDX] / d_n_matvec;
        double bw_t  = timers[_MV_BW_TIMER_IDX]  / d_n_matvec;
        double den_t = timers[_MV_DEN_TIMER_IDX] / d_n_matvec;
        double rdc_t = timers[_MV_RDC_TIMER_IDX] / d_n_matvec;
        matvec_t = fw_t + mid_t + bw_t + den_t + rdc_t;
        printf(
            "  * H2 matvec average time (sec) = %.3lf, %.2lf GB/s\n", 
            matvec_t, mv_MB / matvec_t / 1024.0
        );
        printf(
            "      |----> Forward transformation      = %.3lf, %.2lf GB/s\n", 
            fw_t, fw_MB / fw_t / 1024.0
        );
        if (h2pack->BD_JIT == 0)
        {
            printf(
                "      |----> Intermediate multiplication = %.3lf, %.2lf GB/s\n", 
                mid_t, mid_MB / mid_t / 1024.0
            );
        } else {
            double GFLOPS = h2pack->JIT_flops[0] / 1000000000.0;
            printf(
                "      |----> Intermediate multiplication = %.3lf, %.2lf GFLOPS\n", 
                mid_t, GFLOPS / mid_t
            );
        }
        printf(
            "      |----> Backward transformation     = %.3lf, %.2lf GB/s\n", 
            bw_t, bw_MB / bw_t / 1024.0
        );
        if (h2pack->BD_JIT == 0)
        {
            printf(
                "      |----> Dense multiplication        = %.3lf, %.2lf GB/s\n", 
                den_t, den_MB / den_t / 1024.0
            );
        } else {
            double GFLOPS = h2pack->JIT_flops[1] / 1000000000.0;
            printf(
                "      |----> Dense multiplication        = %.3lf, %.2lf GFLOPS\n", 
                den_t, GFLOPS / den_t
            );
        }
        printf(
            "      |----> OpenMP reduction            = %.3lf, %.2lf GB/s\n", 
            rdc_t, rdc_MB / rdc_t / 1024.0
        );
    }

    if (h2pack->is_HSS == 1)
    {
        double ULV_solve_t = timers[_ULV_SLV_TIMER_IDX] / (double) h2pack->n_ULV_solve;
        printf("  * HSS factorization time (sec) = %.3lf\n", timers[_ULV_FCT_TIMER_IDX]);
        printf("  * HSS solve average time (sec) = %.3lf\n", ULV_solve_t);
    }
    
    printf("=============================================================\n");
}

