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
    H2P_dense_mat_t A, U;
    H2P_dense_mat_init(&A, nrow, ncol);
    
    DTYPE delta_y = 1.0 / (DTYPE)(nrow - 1);
    DTYPE delta_x = 1.0 / (DTYPE)(ncol - 1);
    for (int irow = 0; irow < nrow; irow++)
    {
        DTYPE *A_irow = A->data + irow * ncol;
        DTYPE y_irow = delta_y * irow;
        for (int icol = 0; icol < ncol; icol++)
        {
            DTYPE x_icol = 0.5 + delta_x * icol;
            A_irow[icol] = 1.0 / (1.0 + x_icol - y_irow);
        }
    }
    FILE *ouf = fopen("A.csv", "w");
    for (int irow = 0; irow < nrow; irow++)
    {
        DTYPE *A_irow = A->data + irow * ncol;
        for (int icol = 0; icol < ncol - 1; icol++)
        {
            fprintf(ouf, "%.15lf, ", A_irow[icol]);
            //printf("%e ", A_irow[icol]);
        }
        fprintf(ouf, "%.15lf\n", A_irow[ncol - 1]);
        //printf("%e\n", A_irow[ncol - 1]);
    }
    fclose(ouf);

    int *J = (int*) malloc(sizeof(int) * nrow);
    DTYPE tol_norm;
    printf("tol_rank: ");
    scanf("%lf", &tol_norm);
    //int nrank;
    //printf("rank:");
    //scanf("%d", &nrank);
    mkl_set_num_threads(1);
    double st = H2P_get_wtime_sec();
    H2P_ID_compress(A, QR_REL_NRM, &tol_norm, &U, J);
    double et = H2P_get_wtime_sec();
    printf("U rank = %d, used time = %.4lf (s)\n", U->ncol, et - st);
    fflush(stdout);
    ouf = fopen("U.csv", "w");
    printf("U: \n");
    for (int irow = 0; irow < U->nrow; irow++)
    {
        DTYPE *U_irow = U->data + irow * U->ncol;
        for (int icol = 0; icol < U->ncol - 1; icol++) 
        {
            fprintf(ouf, "%.15lf, ", U_irow[icol]);
            //printf("% .4lf  ", U_irow[icol]);
        }
        fprintf(ouf, "%.15lf\n", U_irow[U->ncol - 1]);
        //printf("% .4lf  \n", U_irow[U->ncol - 1]);
    }
    fclose(ouf);
    
    printf("J: ");
    for (int icol = 0; icol < U->ncol; icol++) printf("%d ", J[icol]);
    printf("\n");
    
    free(J);
    H2P_dense_mat_destroy(U);
    H2P_dense_mat_destroy(A);
    return 0;
}