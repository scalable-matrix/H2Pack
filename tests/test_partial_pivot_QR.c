#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>

#include "H2Pack_utils.h"
#include "H2Pack_dense_mat.h"
#include "H2Pack_ID_compress.h"

int main()
{
    int nrow, ncol;
    scanf("%d %d", &nrow, &ncol);
    H2P_dense_mat_t A;
    H2P_dense_mat_init(&A, nrow, ncol);
    
    A->ld = nrow;  // Treat A as a column major style matrix
    DTYPE delta_y = 1.0 / (DTYPE)(nrow - 1);
    DTYPE delta_x = 1.0 / (DTYPE)(ncol - 1);
    for (int icol = 0; icol < ncol; icol++)
    {
        DTYPE *A_icol = A->data + icol * nrow;
        DTYPE x_icol = 2.0 + delta_x * icol;
        for (int irow = 0; irow < nrow; irow++)
        {
            DTYPE y_irow = delta_y * irow;
            A_icol[irow] = 1.0 / (1.0 + x_icol - y_irow);
        }
    }
    FILE *ouf = fopen("AT.csv", "w");
    for (int icol = 0; icol < ncol; icol++)
    {
        DTYPE *A_icol = A->data + icol * nrow;
        for (int irow = 0; irow < nrow - 1; irow++) fprintf(ouf, "%.15lf, ", A_icol[irow]);
        fprintf(ouf, "%.15lf\n", A_icol[nrow - 1]);
    }
    fclose(ouf);

    int *p, r, tol_rank;
    DTYPE tol_norm;
    printf("Stop param:");
    scanf("%d", &tol_rank);
    scanf("%lf", &tol_norm);
    mkl_set_num_threads(1);
    double st = H2P_get_wtime_sec();
    H2P_partial_pivot_QR(A, tol_rank, tol_norm, &p, &r);
    double et = H2P_get_wtime_sec();
    printf("Partial pivot QR rank = %d, used time = %.4lf (s)\n", r, et - st);
    ouf = fopen("RT.csv", "w");
    for (int icol = 0; icol < ncol; icol++)
    {
        DTYPE *A_icol = A->data + icol * nrow;
        for (int irow = 0; irow < nrow - 1; irow++) fprintf(ouf, "%.15lf, ", A_icol[irow]);
        fprintf(ouf, "%.15lf\n", A_icol[nrow - 1]);
    }
    fclose(ouf);
    
    H2P_dense_mat_destroy(A);
    free(p);
    return 0;
}