#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_utils.h"
#include "H2Pack_typedef.h"
#include "H2Pack_gen_proxy_point.h"
#include "H2Pack_file_IO.h"

void H2P_store_to_file(H2Pack_p h2pack, const char *metadata_fname, const char *binary_fname)
{
    if (h2pack->is_H2ERI || h2pack->is_RPY_Ewald || (h2pack->xpt_dim > h2pack->pt_dim))
    {
        ERROR_PRINTF("Cannot store H2 matrix in H2ERI/RPY-Ewald mode to file\n");
        return;
    }

    FILE *metadata_file = fopen(metadata_fname, "w");
    FILE *binary_file   = fopen(binary_fname,   "wb");

    H2P_dense_mat_p tmpM;
    H2P_dense_mat_init(&tmpM, 4096, 4096);

    // 1. Metadata: H2 / HSS common part
    fprintf(metadata_file, "%d\n", h2pack->pt_dim);
    fprintf(metadata_file, "%d\n", h2pack->krnl_dim);
    fprintf(metadata_file, "%d\n", h2pack->n_point);
    fprintf(metadata_file, "%d\n", h2pack->krnl_mat_size);
    fprintf(metadata_file, "%d\n", 1);  // Kernel matrix is symmetric
    fprintf(metadata_file, "%d\n", h2pack->max_leaf_points);
    fprintf(metadata_file, "%d\n", h2pack->n_node);
    fprintf(metadata_file, "%d\n", h2pack->root_idx);
    fprintf(metadata_file, "%d\n", h2pack->max_level + 1);
    int output_min_adm_level, output_n_r_inadm_pair, output_n_r_adm_pair;
    int *output_r_inadm_pairs, *output_r_adm_pairs;
    if (h2pack->is_HSS)
    {
        fprintf(metadata_file, "%d\n", 1);
        output_min_adm_level  = h2pack->HSS_min_adm_level;
        output_n_r_inadm_pair = h2pack->HSS_n_r_inadm_pair;
        output_n_r_adm_pair   = h2pack->HSS_n_r_adm_pair;
        output_r_inadm_pairs  = h2pack->HSS_r_inadm_pairs;
        output_r_adm_pairs    = h2pack->HSS_r_adm_pairs;
    } else {
        fprintf(metadata_file, "%d\n", 0);
        output_min_adm_level  = h2pack->min_adm_level;
        output_n_r_inadm_pair = h2pack->n_r_inadm_pair;
        output_n_r_adm_pair   = h2pack->n_r_adm_pair;
        output_r_inadm_pairs  = h2pack->r_inadm_pairs;
        output_r_adm_pairs    = h2pack->r_adm_pairs;
    }
    fprintf(metadata_file, "%d\n", output_min_adm_level);
    fprintf(metadata_file, "%d\n", output_n_r_inadm_pair);
    fprintf(metadata_file, "%d\n", output_n_r_adm_pair);
    fprintf(metadata_file, "%e\n", h2pack->QR_stop_tol);
    for (int i = 0; i < h2pack->n_node; i++)
    {
        fprintf(metadata_file, "%6d ", i);
        fprintf(metadata_file, "%2d ", h2pack->node_level[i]);
        fprintf(metadata_file, "%8d %8d ", h2pack->pt_cluster[2 * i], h2pack->pt_cluster[2 * i + 1]);
        fprintf(metadata_file, "%2d ", h2pack->n_child[i]);
        int *node_i_childs = h2pack->children + i * h2pack->max_child;
        for (int j = 0; j < h2pack->n_child[i]; j++) fprintf(metadata_file, "%8d ", node_i_childs[j]);
        for (int j = h2pack->n_child[i]; j < h2pack->max_child; j++) fprintf(metadata_file, "-1 ");
        fprintf(metadata_file, "\n");
    }
    
    // 2. Binary data: input point coordinates and permutation array
    DTYPE *coord0 = (DTYPE *) malloc(sizeof(DTYPE) * h2pack->n_point * h2pack->pt_dim);
    for (int i = 0; i < h2pack->n_point; i++)
    {
        int idx0 = h2pack->coord_idx[i];
        for (int j = 0; j < h2pack->pt_dim; j++)
            coord0[idx0 * h2pack->pt_dim + j] = h2pack->coord[j * h2pack->n_point + i];
    }
    fwrite(coord0, sizeof(DTYPE), h2pack->n_point * h2pack->pt_dim, binary_file);
    fwrite(h2pack->coord_idx, sizeof(int), h2pack->n_point, binary_file);
    free(coord0);
    
    // 3. Metadata & binary data: U matrices
    for (int i = 0; i < h2pack->n_node; i++)
    {
        int U_nrow = h2pack->U[i]->nrow;
        int U_ncol = h2pack->U[i]->ncol;
        fprintf(metadata_file, "%6d %5d %5d\n", i, U_nrow, U_ncol);
        fwrite(h2pack->U[i]->data, sizeof(DTYPE), U_nrow * U_ncol, binary_file);
    }

    // 4. Metadata & binary data: B matrices
    for (int i = 0; i < output_n_r_adm_pair; i++)
    {
        int node0 = output_r_adm_pairs[2 * i];
        int node1 = output_r_adm_pairs[2 * i + 1];
        H2P_get_Bij_block(h2pack, node0, node1, tmpM);
        fprintf(metadata_file, "%6d %6d %5d %5d\n", node0, node1, tmpM->nrow, tmpM->ncol);
        fwrite(tmpM->data, sizeof(DTYPE), tmpM->nrow * tmpM->ncol, binary_file);
    }

    // 5. Metadata & binary data: D matrices
    int *leaf_nodes = h2pack->height_nodes;
    for (int i = 0; i < h2pack->n_leaf_node; i++)
    {
        int node = leaf_nodes[i];  // i-th leaf node
        H2P_get_Dij_block(h2pack, node, node, tmpM);
        fprintf(metadata_file, "%6d %6d %5d %5d\n", node, node, tmpM->nrow, tmpM->ncol);
        fwrite(tmpM->data, sizeof(DTYPE), tmpM->nrow * tmpM->ncol, binary_file);
    }
    for (int i = 0; i < output_n_r_inadm_pair; i++)
    {
        int node0 = output_r_inadm_pairs[2 * i];
        int node1 = output_r_inadm_pairs[2 * i + 1];
        H2P_get_Dij_block(h2pack, node0, node1, tmpM);
        fprintf(metadata_file, "%6d %6d %5d %5d\n", node0, node1, tmpM->nrow, tmpM->ncol);
        fwrite(tmpM->data, sizeof(DTYPE), tmpM->nrow * tmpM->ncol, binary_file);
    }

    // 6. Metadata: H2Pack dedicated part: skeleton point indices
    fprintf(metadata_file, "1\n");  // Has H2Pack skeleton point indices
    for (int i = 0; i < h2pack->n_node; i++)
    {
        H2P_int_vec_p Ji = h2pack->J[i];
        fprintf(metadata_file, "%6d %6d ", i, Ji->length);
        for (int j = 0; j < Ji->length; j++)
            fprintf(metadata_file, "%d ", Ji->data[j]);
        fprintf(metadata_file, "\n");
    }
    
    H2P_dense_mat_destroy(&tmpM);
    fclose(metadata_file);
    fclose(binary_file);
}

void H2P_read_from_file(
    H2Pack_p *h2pack_, const char *metadata_fname, const char *binary_fname, const int BD_JIT, 
    void *krnl_param, kernel_eval_fptr krnl_eval, kernel_bimv_fptr krnl_bimv, const int krnl_bimv_flops
)
{
    H2Pack_p h2pack;

    FILE *metadata_file = fopen(metadata_fname, "r");
    FILE *binary_file   = fopen(binary_fname,   "rb");
    ASSERT_PRINTF(metadata_file != NULL, "Cannot open metadata file %s\n",    metadata_fname);
    ASSERT_PRINTF(binary_file   != NULL, "Cannot open binary data file %s\n", binary_fname);

    // 1. Metadata: H2 / HSS common part
    int pt_dim, krnl_dim, is_symm;
    DTYPE reltol = 1e-6;
    fscanf(metadata_file, "%d", &pt_dim);
    fscanf(metadata_file, "%d", &krnl_dim);
    H2P_init(&h2pack, pt_dim, krnl_dim, QR_REL_NRM, &reltol);
    fscanf(metadata_file, "%d", &h2pack->n_point);
    fscanf(metadata_file, "%d", &h2pack->krnl_mat_size);
    fscanf(metadata_file, "%d", &is_symm);
    if (is_symm != 1)
    {
        H2P_destroy(&h2pack);
        ASSERT_PRINTF(is_symm == 1, "H2Pack only support symmetric kernel matrix\n");
    }
    fscanf(metadata_file, "%d", &h2pack->max_leaf_points);
    fscanf(metadata_file, "%d", &h2pack->n_node);
    fscanf(metadata_file, "%d", &h2pack->root_idx);
    fscanf(metadata_file, "%d", &h2pack->max_level);
    h2pack->max_level--;
    fscanf(metadata_file, "%d", &h2pack->is_HSS);
    int input_n_r_adm_pair, input_n_r_inadm_pair;
    int *input_r_adm_pairs, *input_r_inadm_pairs;
    if (h2pack->is_HSS)
    {
        fscanf(metadata_file, "%d", &h2pack->HSS_min_adm_level);
        fscanf(metadata_file, "%d", &h2pack->HSS_n_r_inadm_pair);
        fscanf(metadata_file, "%d", &h2pack->HSS_n_r_adm_pair);
        h2pack->HSS_r_inadm_pairs = (int *) malloc(sizeof(int) * h2pack->HSS_n_r_inadm_pair * 2);
        h2pack->HSS_r_adm_pairs   = (int *) malloc(sizeof(int) * h2pack->HSS_n_r_adm_pair * 2);
        input_n_r_inadm_pair = h2pack->HSS_n_r_inadm_pair;
        input_n_r_adm_pair   = h2pack->HSS_n_r_adm_pair;
        input_r_inadm_pairs  = h2pack->HSS_r_inadm_pairs;
        input_r_adm_pairs    = h2pack->HSS_r_adm_pairs;
    } else {
        fscanf(metadata_file, "%d", &h2pack->min_adm_level);
        fscanf(metadata_file, "%d", &h2pack->n_r_inadm_pair);
        fscanf(metadata_file, "%d", &h2pack->n_r_adm_pair);
        h2pack->r_inadm_pairs = (int *) malloc(sizeof(int) * h2pack->n_r_inadm_pair * 2);
        h2pack->r_adm_pairs   = (int *) malloc(sizeof(int) * h2pack->n_r_adm_pair * 2);
        input_n_r_inadm_pair = h2pack->n_r_inadm_pair;
        input_n_r_adm_pair   = h2pack->n_r_adm_pair;
        input_r_inadm_pairs  = h2pack->r_inadm_pairs;
        input_r_adm_pairs    = h2pack->r_adm_pairs;
    }
    fscanf(metadata_file, DTYPE_FMTSTR, &reltol);
    memcpy(&h2pack->QR_stop_tol, &reltol, sizeof(DTYPE));
    h2pack->xpt_dim = pt_dim;
    h2pack->krnl_mat_size = h2pack->n_point * krnl_dim;
    h2pack->krnl_param = krnl_param;
    h2pack->krnl_eval  = krnl_eval;
    h2pack->krnl_bimv  = krnl_bimv;
    h2pack->krnl_bimv_flops = krnl_bimv_flops;

    int n_node  = h2pack->n_node;
    int n_point = h2pack->n_point;
    h2pack->node_level  = (int*) malloc(sizeof(int) * n_node);
    h2pack->pt_cluster  = (int*) malloc(sizeof(int) * n_node * 2);
    h2pack->mat_cluster = (int*) malloc(sizeof(int) * n_node * 2);
    h2pack->n_child     = (int*) malloc(sizeof(int) * n_node);
    h2pack->children    = (int*) malloc(sizeof(int) * n_node * h2pack->max_child);
    h2pack->parent      = (int*) malloc(sizeof(int) * n_node);
    int n_leaf_node = 0;
    for (int i = 0; i < n_node; i++)
    {
        int node, level, tmp;
        fscanf(metadata_file, "%d", &node);
        fscanf(metadata_file, "%d", &level);
        fscanf(metadata_file, "%d", &h2pack->pt_cluster[2 * node]);
        fscanf(metadata_file, "%d", &h2pack->pt_cluster[2 * node + 1]);
        fscanf(metadata_file, "%d", &h2pack->n_child[node]);
        h2pack->mat_cluster[2 * node]     = h2pack->krnl_dim * h2pack->pt_cluster[2 * node];
        h2pack->mat_cluster[2 * node + 1] = h2pack->krnl_dim * (h2pack->pt_cluster[2 * node + 1] + 1) - 1;
        if (h2pack->n_child[node] == 0) n_leaf_node++;
        h2pack->node_level[node] = level;
        int *node_i_childs = h2pack->children + i * h2pack->max_child;
        for (int j = 0; j < h2pack->n_child[node]; j++)
        {
            fscanf(metadata_file, "%d", &node_i_childs[j]);
            h2pack->parent[node_i_childs[j]] = node;
        }
        for (int j = h2pack->n_child[node]; j < h2pack->max_child; j++)
        {
            fscanf(metadata_file, "%d", &tmp);
            node_i_childs[j] = -1;
        }
    }
    h2pack->level_n_node  = (int*) malloc(sizeof(int) * n_node);
    h2pack->height_n_node = (int*) malloc(sizeof(int) * n_node);
    h2pack->level_nodes   = (int*) malloc(sizeof(int) * n_leaf_node * (h2pack->max_level + 1));
    h2pack->height_nodes  = (int*) malloc(sizeof(int) * n_leaf_node * (h2pack->max_level + 1));
    memset(h2pack->level_n_node,  0, sizeof(int) * n_node);
    memset(h2pack->height_n_node, 0, sizeof(int) * n_node);
    h2pack->n_leaf_node = n_leaf_node;
    h2pack->height_nodes[0] = n_leaf_node;
    for (int node = 0; node < n_node; node++)
    {
        int level = h2pack->node_level[node];
        int level_idx = h2pack->level_n_node[level];
        h2pack->level_nodes[level * n_leaf_node + level_idx] = node;
        h2pack->level_n_node[level]++;
    }

    // 2. Binary data: input point coordinates and permutation array
    DTYPE *coord0     = (DTYPE*) malloc(sizeof(DTYPE) * n_point * pt_dim);
    h2pack->coord     = (DTYPE*) malloc(sizeof(DTYPE) * n_point * pt_dim);
    h2pack->coord0    = (DTYPE*) malloc(sizeof(DTYPE) * n_point * pt_dim);
    h2pack->coord_idx = (int*)   malloc(sizeof(int)   * n_point);
    fread(coord0, sizeof(DTYPE), n_point * pt_dim, binary_file);
    fread(h2pack->coord_idx, sizeof(int), n_point, binary_file);
    for (int i = 0; i < n_point; i++)
    {
        int idx0 = h2pack->coord_idx[i];
        for (int j = 0; j < pt_dim; j++)
        {
            h2pack->coord[j * n_point + i] = coord0[idx0 * pt_dim + j];
            h2pack->coord0[j * n_point + idx0] = coord0[idx0 * pt_dim + j];
        }
    }
    free(coord0);

    // 3. Metadata & binary data: U matrices
    h2pack->n_UJ = n_node;
    h2pack->U       = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    h2pack->J       = (H2P_int_vec_p*)   malloc(sizeof(H2P_int_vec_p)   * n_node);
    h2pack->J_coord = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]       = NULL;
        h2pack->J[i]       = NULL;
        h2pack->J_coord[i] = NULL;
    }
    H2P_dense_mat_p *U       = h2pack->U;
    H2P_int_vec_p   *J       = h2pack->J;
    H2P_dense_mat_p *J_coord = h2pack->J_coord;
    for (int i = 0; i < n_node; i++)
    {
        int node, U_nrow, U_ncol;
        fscanf(metadata_file, "%d", &node);
        fscanf(metadata_file, "%d", &U_nrow);
        fscanf(metadata_file, "%d", &U_ncol);
        H2P_dense_mat_init(&U[node], U_nrow, U_ncol);
        U[node]->nrow = U_nrow;
        U[node]->ncol = U_ncol;
        fread(U[node]->data, sizeof(DTYPE), U_nrow * U_ncol, binary_file);
    }

    // 4. Metadata: B matrices
    h2pack->n_B = input_n_r_adm_pair;
    h2pack->B_nrow = (int*)    malloc(sizeof(int)    * input_n_r_adm_pair);
    h2pack->B_ncol = (int*)    malloc(sizeof(int)    * input_n_r_adm_pair);
    h2pack->B_ptr  = (size_t*) malloc(sizeof(size_t) * (input_n_r_adm_pair + 1));
    h2pack->B_ptr[0] = 0;
    size_t B_total_size = 0;
    int    *B_nrow = h2pack->B_nrow;
    int    *B_ncol = h2pack->B_ncol;
    size_t *B_ptr  = h2pack->B_ptr;
    for (int i = 0; i < input_n_r_adm_pair; i++)
    {
        fscanf(metadata_file, "%d", &input_r_adm_pairs[2 * i]);
        fscanf(metadata_file, "%d", &input_r_adm_pairs[2 * i + 1]);
        fscanf(metadata_file, "%d", &B_nrow[i]);
        fscanf(metadata_file, "%d", &B_ncol[i]);
        size_t Bi_size = (size_t) (B_nrow[i] * B_ncol[i]);
        B_ptr[i + 1] = Bi_size;
        B_total_size += Bi_size;
    }
    size_t *mat_size = h2pack->mat_size;

    // 5. Metadata: D matrices
    h2pack->n_D = n_leaf_node + input_n_r_inadm_pair;
    h2pack->D_nrow = (int*)    malloc(sizeof(int)    * h2pack->n_D);
    h2pack->D_ncol = (int*)    malloc(sizeof(int)    * h2pack->n_D);
    h2pack->D_ptr  = (size_t*) malloc(sizeof(size_t) * (h2pack->n_D + 1));
    int    *D_nrow = h2pack->D_nrow;
    int    *D_ncol = h2pack->D_ncol;
    size_t *D_ptr  = h2pack->D_ptr;
    D_ptr[0] = 0;
    size_t D0_total_size = 0;
    int *leaf_nodes = h2pack->height_nodes;
    for (int i = 0; i < n_leaf_node; i++)
    {
        fscanf(metadata_file, "%d", &leaf_nodes[i]);
        fscanf(metadata_file, "%d", &leaf_nodes[i]);
        fscanf(metadata_file, "%d", &D_nrow[i]);
        fscanf(metadata_file, "%d", &D_ncol[i]);
        size_t Di_size = (size_t) (D_nrow[i] * D_ncol[i]);
        D_ptr[i + 1] = Di_size;
        D0_total_size += Di_size;
    }
    size_t D1_total_size = 0;
    for (int i = 0; i < input_n_r_inadm_pair; i++)
    {
        int ii = i + n_leaf_node;
        fscanf(metadata_file, "%d", &input_r_inadm_pairs[2 * i]);
        fscanf(metadata_file, "%d", &input_r_inadm_pairs[2 * i + 1]);
        fscanf(metadata_file, "%d", &D_nrow[ii]);
        fscanf(metadata_file, "%d", &D_ncol[ii]);
        size_t Di_size = (size_t) (D_nrow[ii] * D_ncol[ii]);
        D_ptr[ii + 1] = Di_size;
        D1_total_size += Di_size;
    }
    size_t D_total_size = D0_total_size + D1_total_size;

    // 6. Metadata: H2Pack dedicated part: skeleton point indices
    int has_skel;
    fscanf(metadata_file, "%d", &has_skel);
    if (has_skel)
    {
        for (int i = 0; i < n_node; i++)
        {
            int node, length;
            fscanf(metadata_file, "%d", &node);
            fscanf(metadata_file, "%d", &length);
            if (length > 0)
            {
                H2P_int_vec_init(&J[node], length);
                for (int j = 0; j < length; j++)
                    fscanf(metadata_file, "%d", &J[node]->data[j]);
                J[node]->length = length;
                H2P_dense_mat_init(&J_coord[node], pt_dim, length);
                H2P_gather_matrix_columns(
                    h2pack->coord, n_point, J_coord[node]->data, J_coord[node]->ld, 
                    pt_dim, J[node]->data, J[node]->length
                );
            } else {
                H2P_int_vec_init(&J[node], 1);
                J[node]->length = 0;
                H2P_dense_mat_init(&J_coord[node], pt_dim, 1);
                J_coord[node]->nrow = 0;
                J_coord[node]->ncol = 0;
                J_coord[node]->ld   = 0;
            }
        }
        h2pack->BD_JIT = BD_JIT;
    } else {
        h2pack->BD_JIT = 0;
    }

    // 7. Binary data: B matrices
    if (h2pack->BD_JIT == 0)
    {
        h2pack->B_data = (DTYPE*) malloc_aligned(sizeof(DTYPE) * B_total_size, 64);
        ASSERT_PRINTF(h2pack->B_data != NULL, "Failed to allocate space for storing all %zu B matrices elements\n", B_total_size);
        DTYPE *B_data = h2pack->B_data;
        size_t B_offset = 0;
        for (int i = 0; i < input_n_r_adm_pair; i++)
        {
            size_t Bi_size = B_nrow[i] * B_ncol[i];
            fread(B_data + B_offset, sizeof(DTYPE), Bi_size, binary_file);
            B_offset += Bi_size;
        }
    }

    // 8. Binary data: D matrices
    if (h2pack->BD_JIT == 0)
    {
        h2pack->D_data = (DTYPE*) malloc_aligned(sizeof(DTYPE) * D_total_size, 64);
        ASSERT_PRINTF(h2pack->D_data != NULL, "Failed to allocate space for storing all %zu D matrices elements\n", D_total_size);
        DTYPE *D_data = h2pack->D_data;
        size_t D_offset = 0;
        for (int i = 0; i < n_leaf_node; i++)
        {
            size_t Di_size = D_nrow[i] * D_ncol[i];
            fread(D_data + D_offset, sizeof(DTYPE), Di_size, binary_file);
            D_offset += Di_size;
        }
        for (int i = 0; i < input_n_r_inadm_pair; i++)
        {
            int ii = i + n_leaf_node;
            size_t Di_size = D_nrow[ii] * D_ncol[ii];
            fread(D_data + D_offset, sizeof(DTYPE), Di_size, binary_file);
            D_offset += Di_size;
        }
    }

    // 9. Post-processing of U matrices
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        if (U[i] == NULL)
        {
            H2P_dense_mat_init(&U[i], 1, 1);
            U[i]->nrow = 0;
            U[i]->ncol = 0;
            U[i]->ld   = 0;
        } else {
            mat_size[U_SIZE_IDX]      += U[i]->nrow * U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
        }
    }

    // 10. Post-processing of B matrices
    int *B_pair_i = (int*) malloc(sizeof(int) * input_n_r_adm_pair * 2);
    int *B_pair_j = (int*) malloc(sizeof(int) * input_n_r_adm_pair * 2);
    int *B_pair_v = (int*) malloc(sizeof(int) * input_n_r_adm_pair * 2);
    ASSERT_PRINTF(
        B_pair_i != NULL && B_pair_j != NULL && B_pair_v != NULL,
        "Failed to allocate working buffer for B matrices indexing\n"
    );
    double *JIT_flops = h2pack->JIT_flops;
    int *node_level = h2pack->node_level;
    int *pt_cluster = h2pack->pt_cluster;
    h2pack->node_n_r_adm = (int*) malloc(sizeof(int) * n_node);
    ASSERT_PRINTF(
        h2pack->node_n_r_adm != NULL, 
        "Failed to allocate array of size %d for counting node admissible pairs\n", n_node
    );
    int *node_n_r_adm = h2pack->node_n_r_adm;
    memset(node_n_r_adm, 0, sizeof(int) * n_node);
    int B_pair_cnt = 0;
    for (int i = 0; i < input_n_r_adm_pair; i++)
    {
        int node0 = input_r_adm_pairs[2 * i];
        int node1 = input_r_adm_pairs[2 * i + 1];
        B_pair_i[B_pair_cnt] = node0;
        B_pair_j[B_pair_cnt] = node1;
        B_pair_v[B_pair_cnt] = i + 1;
        B_pair_cnt++;
        B_pair_i[B_pair_cnt] = node1;
        B_pair_j[B_pair_cnt] = node0;
        B_pair_v[B_pair_cnt] = -(i + 1);
        B_pair_cnt++;
        node_n_r_adm[node0]++;
        node_n_r_adm[node1]++;
        mat_size[MV_MID_SIZE_IDX] +=  B_nrow[i] * B_ncol[i];
        mat_size[MV_MID_SIZE_IDX] += (B_nrow[i] + B_ncol[i]);
        mat_size[MV_MID_SIZE_IDX] += (B_nrow[i] + B_ncol[i]);
        if (h2pack->BD_JIT)
        {
            int level0 = node_level[node0];
            int level1 = node_level[node1];
            int node0_npt = 0, node1_npt = 0;
            if (level0 == level1)
            {
                node0_npt = J[node0]->length;
                node1_npt = J[node1]->length;
            }
            if (level0 > level1)
            {
                int pt_s1 = pt_cluster[2 * node1];
                int pt_e1 = pt_cluster[2 * node1 + 1];
                node0_npt = J[node0]->length;
                node1_npt = pt_e1 - pt_s1 + 1;
            }
            if (level0 < level1)
            {
                int pt_s0 = pt_cluster[2 * node0];
                int pt_e0 = pt_cluster[2 * node0 + 1];
                node0_npt = pt_e0 - pt_s0 + 1;
                node1_npt = J[node1]->length;
            }
            JIT_flops[JIT_B_FLOPS_IDX] += (double)(krnl_bimv_flops) * (double)(node0_npt * node1_npt);
        }
    }
    
    int BD_ntask_thread = (h2pack->BD_JIT == 1) ? BD_NTASK_THREAD : 1;
    int n_B_blk = h2pack->n_thread * BD_ntask_thread;
    H2P_partition_workload(input_n_r_adm_pair, B_ptr + 1, B_total_size, n_B_blk, h2pack->B_blk);
    for (int i = 1; i <= input_n_r_adm_pair; i++) B_ptr[i] += B_ptr[i - 1];
    mat_size[B_SIZE_IDX] = B_total_size;

    h2pack->B_p2i_rowptr = (int*) malloc(sizeof(int) * (n_node + 1));
    h2pack->B_p2i_colidx = (int*) malloc(sizeof(int) * input_n_r_adm_pair * 2);
    h2pack->B_p2i_val    = (int*) malloc(sizeof(int) * input_n_r_adm_pair * 2);
    ASSERT_PRINTF(h2pack->B_p2i_rowptr != NULL, "Failed to allocate arrays for B matrices indexing\n");
    ASSERT_PRINTF(h2pack->B_p2i_colidx != NULL, "Failed to allocate arrays for B matrices indexing\n");
    ASSERT_PRINTF(h2pack->B_p2i_val    != NULL, "Failed to allocate arrays for B matrices indexing\n");
    H2P_int_COO_to_CSR(
        n_node, B_pair_cnt, B_pair_i, B_pair_j, B_pair_v, 
        h2pack->B_p2i_rowptr, h2pack->B_p2i_colidx, h2pack->B_p2i_val
    );
    free(B_pair_i);
    free(B_pair_j);
    free(B_pair_v);

    // 11. Post-processing of D matrices
    int D_pair_cnt = 0;
    int n_Dij_pair = n_leaf_node + 2 * input_n_r_inadm_pair;
    int *D_pair_i  = (int*) malloc(sizeof(int) * n_Dij_pair);
    int *D_pair_j  = (int*) malloc(sizeof(int) * n_Dij_pair);
    int *D_pair_v  = (int*) malloc(sizeof(int) * n_Dij_pair);
    ASSERT_PRINTF(
        D_pair_i != NULL && D_pair_j != NULL && D_pair_v != NULL,
        "Failed to allocate working buffer for D matrices indexing\n"
    );

    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int pt_s = pt_cluster[2 * node];
        int pt_e = pt_cluster[2 * node + 1];
        int node_npt = pt_e - pt_s + 1;
        D_pair_i[D_pair_cnt] = node;
        D_pair_j[D_pair_cnt] = node;
        D_pair_v[D_pair_cnt] = i + 1;
        D_pair_cnt++;
        mat_size[MV_DEN_SIZE_IDX] += D_nrow[i] * D_ncol[i];
        mat_size[MV_DEN_SIZE_IDX] += D_nrow[i] + D_ncol[i];
        if (h2pack->BD_JIT) JIT_flops[JIT_D_FLOPS_IDX] += (double)(krnl_bimv_flops) * (double)(node_npt * node_npt);
    }
    for (int i = 0; i < input_n_r_inadm_pair; i++)
    {
        int ii = i + n_leaf_node;
        int node0 = input_r_inadm_pairs[2 * i];
        int node1 = input_r_inadm_pairs[2 * i + 1];
        int pt_s0 = pt_cluster[2 * node0];
        int pt_s1 = pt_cluster[2 * node1];
        int pt_e0 = pt_cluster[2 * node0 + 1];
        int pt_e1 = pt_cluster[2 * node1 + 1];
        int node0_npt = pt_e0 - pt_s0 + 1;
        int node1_npt = pt_e1 - pt_s1 + 1;
        D_pair_i[D_pair_cnt] = node0;
        D_pair_j[D_pair_cnt] = node1;
        D_pair_v[D_pair_cnt] = ii + 1;
        D_pair_cnt++;
        D_pair_i[D_pair_cnt] = node1;
        D_pair_j[D_pair_cnt] = node0;
        D_pair_v[D_pair_cnt] = -(ii + 1);
        D_pair_cnt++;
        mat_size[MV_DEN_SIZE_IDX] +=  D_nrow[ii] * D_ncol[ii];
        mat_size[MV_DEN_SIZE_IDX] += (D_nrow[ii] + D_ncol[ii]);
        mat_size[MV_DEN_SIZE_IDX] += (D_nrow[ii] + D_ncol[ii]);
        if (h2pack->BD_JIT) JIT_flops[JIT_D_FLOPS_IDX] += (double)(krnl_bimv_flops) * (double)(node0_npt * node1_npt);
    }

    int D_n_blk = h2pack->n_thread * BD_ntask_thread;
    H2P_partition_workload(n_leaf_node,          D_ptr + 1,               D0_total_size, D_n_blk, h2pack->D_blk0);
    H2P_partition_workload(input_n_r_inadm_pair, D_ptr + n_leaf_node + 1, D1_total_size, D_n_blk, h2pack->D_blk1);
    for (int i = 1; i <= n_leaf_node + input_n_r_inadm_pair; i++) D_ptr[i] += D_ptr[i - 1];
    mat_size[D_SIZE_IDX] = D_total_size;

    h2pack->D_p2i_rowptr = (int*) malloc(sizeof(int) * (n_node + 1));
    h2pack->D_p2i_colidx = (int*) malloc(sizeof(int) * n_Dij_pair);
    h2pack->D_p2i_val    = (int*) malloc(sizeof(int) * n_Dij_pair);
    ASSERT_PRINTF(h2pack->D_p2i_rowptr != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(h2pack->D_p2i_colidx != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(h2pack->D_p2i_val    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    H2P_int_COO_to_CSR(
        n_node, D_pair_cnt, D_pair_i, D_pair_j, D_pair_v, 
        h2pack->D_p2i_rowptr, h2pack->D_p2i_colidx, h2pack->D_p2i_val
    );
    free(D_pair_i);
    free(D_pair_j);
    free(D_pair_v);

    // 12. Set up permutation arrays
    h2pack->xT    = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->krnl_mat_size);
    h2pack->yT    = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->krnl_mat_size);
    h2pack->pmt_x = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->krnl_mat_size * h2pack->mm_max_n_vec);
    h2pack->pmt_y = (DTYPE*) malloc(sizeof(DTYPE) * h2pack->krnl_mat_size * h2pack->mm_max_n_vec);
    ASSERT_PRINTF(
        h2pack->xT != NULL && h2pack->yT != NULL && h2pack->pmt_x != NULL && h2pack->pmt_y != NULL,
        "Failed to allocate working arrays of size %d for matvec & matmul\n", 2 * h2pack->krnl_mat_size * (h2pack->mm_max_n_vec+1)
    );
    int *coord_idx = h2pack->coord_idx;
    int *fwd_pmt_idx = (int*) malloc(sizeof(int) * n_point * krnl_dim);
    int *bwd_pmt_idx = (int*) malloc(sizeof(int) * n_point * krnl_dim);
    for (int i = 0; i < n_point; i++)
    {
        for (int j = 0; j < krnl_dim; j++)
        {
            fwd_pmt_idx[i * krnl_dim + j] = coord_idx[i] * krnl_dim + j;
            bwd_pmt_idx[coord_idx[i] * krnl_dim + j] = i * krnl_dim + j;
        }
    }
    h2pack->fwd_pmt_idx = fwd_pmt_idx;
    h2pack->bwd_pmt_idx = bwd_pmt_idx;

    // 13. Misc
    H2P_calc_enclosing_box(pt_dim, n_point, h2pack->coord, NULL, &h2pack->enbox);
    H2P_calc_node_inadm_lists(h2pack);
    h2pack->tb = (H2P_thread_buf_p*) malloc(sizeof(H2P_thread_buf_p) * h2pack->n_thread);
    ASSERT_PRINTF(h2pack->tb != NULL, "Failed to allocate %d thread buffers\n", h2pack->n_thread);
    for (int i = 0; i < h2pack->n_thread; i++)
        H2P_thread_buf_init(&h2pack->tb[i], h2pack->krnl_mat_size);

    // Finally done...
    fclose(metadata_file);
    fclose(binary_file);
    *h2pack_ = h2pack;
}