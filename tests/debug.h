
#include "H2Pack.h"
#include "H2Pack_utils.h"
#include "utils.h"

void dump_HSS(H2Pack_p h2pack)
{
    // Assumption: MATLAB code is using the same point set
    // and has the same r_adm_pairs
    FILE *ouf0 = fopen("add_C_HSS_mat_metadata.m", "w");
    FILE *ouf1 = fopen("C_HSS_mat.bin", "wb");

    H2P_dense_mat_p tmpM;
    H2P_dense_mat_init(&tmpM, 1024, 1024);

    fprintf(ouf0, "C_U_sizes = [\n");
    for (int i = 0; i < h2pack->n_node; i++)
    {
        fprintf(ouf0, "%d %d;\n", h2pack->U[i]->nrow, h2pack->U[i]->ncol);
        fwrite(h2pack->U[i]->data, sizeof(DTYPE), h2pack->U[i]->nrow * h2pack->U[i]->ncol, ouf1);
    }
    fprintf(ouf0, "];\n");

    fprintf(ouf0, "C_B_sizes = [\n");
    for (int i = 0; i < h2pack->HSS_n_r_adm_pair; i++)
    {
        int node0 = h2pack->HSS_r_adm_pairs[2 * i];
        int node1 = h2pack->HSS_r_adm_pairs[2 * i + 1];
        H2P_get_Bij_block(h2pack, node0, node1, tmpM);
        fprintf(ouf0, "%d %d;\n", h2pack->B_nrow[i], h2pack->B_ncol[i]);
        fwrite(tmpM->data, sizeof(DTYPE), tmpM->nrow * tmpM->ncol, ouf1);
    }
    fprintf(ouf0, "];\n");

    fprintf(ouf0, "C_D_sizes = [\n");
    for (int i = 0; i < h2pack->n_leaf_node; i++)
    {
        int node = h2pack->height_nodes[i];  // i-th leaf node
        fprintf(ouf0, "%d %d;\n", h2pack->D_nrow[i], h2pack->D_ncol[i]);
        H2P_get_Dij_block(h2pack, node, node, tmpM);
        fwrite(tmpM->data, sizeof(DTYPE), tmpM->nrow * tmpM->ncol, ouf1);
    }
    fprintf(ouf0, "];\n");

    H2P_dense_mat_destroy(tmpM);

    fclose(ouf0);
    fclose(ouf1);
}
