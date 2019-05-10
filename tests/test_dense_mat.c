#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "H2Pack_dense_mat.h"

int main()
{
    int m, n;
    scanf("%d %d", &m, &n);
    H2P_dense_mat_t mat0, mat1;
    H2P_dense_mat_init(&mat0, m, n);
    H2P_dense_mat_init(&mat1, m, n);
    for (int i = 0; i < m * n; i++) 
    {
        mat0->data[i] = (DTYPE) (100 + i);
        mat1->data[i] = 0.0;
    }
    printf("Initial mat 0:\n");
    H2P_dense_mat_print(mat0);
    printf("Initial mat 1:\n");
    H2P_dense_mat_print(mat1);
    
    H2P_dense_mat_copy_block(mat0, mat1, 1, 2, 0, 1, m - 1, n - 2);
    printf("Copy block from mat0 -> mat1:\n");
    H2P_dense_mat_print(mat1);
    
    H2P_dense_mat_transpose(mat1);
    printf("Transposed mat1:\n");
    H2P_dense_mat_print(mat1);
    
    int *p = (int*) malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++) p[i] = (i + 3) % m;
    H2P_dense_mat_permute_rows(mat0, p);
    printf("Permuted mat0:\n");
    H2P_dense_mat_print(mat0);
    
    free(p);
    H2P_dense_mat_destroy(mat0);
    H2P_dense_mat_destroy(mat1);
    free(mat0);
    free(mat1);
    return 0;
}