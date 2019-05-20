#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"

// Initialize a H2Pack structure
void H2P_init(
    H2Pack_t *h2pack_, const int dim, 
    const int QR_stop_type, void *QR_stop_param
)
{
    H2Pack_t h2pack = (H2Pack_t) malloc(sizeof(struct H2Pack));
    assert(h2pack != NULL);
    
    h2pack->n_thread  = omp_get_max_threads();
    h2pack->dim       = dim;
    h2pack->mem_bytes = 0;
    h2pack->max_child = 1 << dim;
    h2pack->n_matvec  = 0;
    memset(h2pack->timers,   0, sizeof(double) * 8);
    memset(h2pack->mat_size, 0, sizeof(int)    * 3);
    
    h2pack->QR_stop_type = QR_stop_type;
    if (QR_stop_type == QR_RANK) 
        memcpy(&h2pack->QR_stop_rank, QR_stop_param, sizeof(int));
    if ((QR_stop_type == QR_REL_NRM) || (QR_stop_type == QR_ABS_NRM))
        memcpy(&h2pack->QR_stop_tol,  QR_stop_param, sizeof(DTYPE));
    
    h2pack->parent        = NULL;
    h2pack->children      = NULL;
    h2pack->cluster       = NULL;
    h2pack->n_child       = NULL;
    h2pack->node_level    = NULL;
    h2pack->node_height   = NULL;
    h2pack->level_n_node  = NULL;
    h2pack->level_nodes   = NULL;
    h2pack->height_n_node = NULL;
    h2pack->height_nodes  = NULL;
    h2pack->r_inadm_pairs = NULL;
    h2pack->r_adm_pairs   = NULL;
    h2pack->coord         = NULL;
    h2pack->enbox         = NULL;
    h2pack->U             = NULL;
    h2pack->J             = NULL;
    h2pack->B             = NULL;
    h2pack->D             = NULL;
    h2pack->y0            = NULL;
    h2pack->y1            = NULL;
    h2pack->tb            = NULL;
    
    *h2pack_ = h2pack;
}

// Destroy a H2Pack structure
void H2P_destroy(H2Pack_t h2pack)
{
    free(h2pack->parent);
    free(h2pack->children);
    free(h2pack->cluster);
    free(h2pack->n_child);
    free(h2pack->node_level);
    free(h2pack->node_height);
    free(h2pack->level_n_node);
    free(h2pack->level_nodes);
    free(h2pack->height_n_node);
    free(h2pack->height_nodes);
    free(h2pack->r_inadm_pairs);
    free(h2pack->r_adm_pairs);
    free(h2pack->coord);
    free(h2pack->enbox);
    
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        H2P_dense_mat_destroy(h2pack->U[i]);
        H2P_int_vec_destroy(h2pack->J[i]);
    }
    for (int i = 0; i < h2pack->n_B; i++)
        H2P_dense_mat_destroy(h2pack->B[i]);
    for (int i = 0; i < h2pack->n_D; i++)
        H2P_dense_mat_destroy(h2pack->D[i]);
    free(h2pack->U);
    free(h2pack->J);
    free(h2pack->B);
    free(h2pack->D);
    
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
    printf("  * Total size of projection matrices   (U) : %d\n", h2pack->mat_size[0]);
    printf("  * Total size of generator matrices    (B) : %d\n", h2pack->mat_size[1]);
    printf("  * Total size of original dense blocks (D) : %d\n", h2pack->mat_size[2]);
    double storage_k = 0.0;
    storage_k += (double) h2pack->mat_size[0];
    storage_k += (double) h2pack->mat_size[1];
    storage_k += (double) h2pack->mat_size[2];
    storage_k /= (double) h2pack->n_point;
    storage_k /= (double) h2pack->n_point;
    printf("  * Compression ratio                       : %.3lf \n", 1.0 / storage_k);
    
    printf("==================== H2Pack timing info =====================\n");
    double build_t = 0.0, matvec_t = 0.0;
    int n_matvec = h2pack->n_matvec;
    double d_n_matvec = (double) h2pack->n_matvec;
    for (int i = 0; i < 4; i++) build_t  += h2pack->timers[i];
    for (int i = 4; i < 8; i++) matvec_t += h2pack->timers[i];
    printf("  * H2 construction time = %.4lf (s)\n", build_t);
    printf("      |----> Point partition = %.4lf (s)\n", h2pack->timers[0]);
    printf("      |----> U construction  = %.4lf (s)\n", h2pack->timers[1]);
    printf("      |----> B construction  = %.4lf (s)\n", h2pack->timers[2]);
    printf("      |----> D construction  = %.4lf (s)\n", h2pack->timers[3]);
    printf(
        "  * H2 matvec total (average) time   = %.4lf (%.4lf) (s)\n", 
        matvec_t, matvec_t / d_n_matvec
    );
    printf(
        "      |----> Upward sweep time       = %.4lf (%.4lf) (s)\n", 
        h2pack->timers[4], h2pack->timers[4] / d_n_matvec
    );
    printf(
        "      |----> Intermediate sweep time = %.4lf (%.4lf) (s)\n", 
        h2pack->timers[5], h2pack->timers[5] / d_n_matvec
    );
    printf(
        "      |----> Downward sweep time     = %.4lf (%.4lf) (s)\n", 
        h2pack->timers[6], h2pack->timers[6] / d_n_matvec
    );
    printf(
        "      |----> Dense block sweep time  = %.4lf (%.4lf) (s)\n", 
        h2pack->timers[7], h2pack->timers[7] / d_n_matvec
    );
    
    printf("=============================================================\n");
}

