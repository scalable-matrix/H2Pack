#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>
#include <time.h>

#include "H2Pack_utils.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_ID_compress.h"

int main()
{
    int nrow, ncol;
    printf("matrix size: ");
    scanf("%d %d", &nrow, &ncol);
    H2P_dense_mat_t A, A0, U;
    H2P_dense_mat_init(&A, nrow, ncol);
    H2P_dense_mat_init(&A0, nrow, ncol);
    
    DTYPE A0_fnorm = 0.0;
    srand(time(NULL));
    DTYPE *x1 = (DTYPE*) malloc(sizeof(DTYPE) * nrow);
    DTYPE *y1 = (DTYPE*) malloc(sizeof(DTYPE) * nrow);
    DTYPE *x2 = (DTYPE*) malloc(sizeof(DTYPE) * ncol);
    DTYPE *y2 = (DTYPE*) malloc(sizeof(DTYPE) * ncol);
    for (int i = 0; i < nrow; i++) 
    {
        x1[i] = (DTYPE) rand() / (DTYPE) RAND_MAX;
        y1[i] = (DTYPE) rand() / (DTYPE) RAND_MAX;
    }
    for (int i = 0; i < ncol; i++) 
    {
        x2[i] = (DTYPE) rand() / (DTYPE) RAND_MAX + 1.0;
        y2[i] = (DTYPE) rand() / (DTYPE) RAND_MAX + 1.0;
    }
    for (int irow = 0; irow < nrow; irow++)
    {
        DTYPE *A_irow  = A->data + irow * ncol;
        DTYPE *A0_irow = A0->data + irow * ncol;
        for (int icol = 0; icol < ncol; icol++)
        {
            DTYPE dx = x1[irow] - x2[icol];
            DTYPE dy = y1[irow] - y2[icol];
            DTYPE d  = sqrt(dx * dx + dy * dy);
            A_irow[icol]  = 1.0 / d;
            A0_irow[icol] = A_irow[icol];
            A0_fnorm += A_irow[icol] * A_irow[icol];
        }
    }
    A0_fnorm = sqrt(A0_fnorm);
    mkl_set_num_threads(1);
    
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
    printf("norm_rel_tol: ");
    scanf("%lf", &tol_norm);
    H2P_ID_compress(A, QR_REL_NRM, &tol_norm, &U, J);  // Warm up
    double ut = 0.0;
    for (int i = 0; i < 10; i++)
    {
        memcpy(A->data, A0->data, sizeof(DTYPE) * nrow * ncol);
        A->nrow = nrow;
        A->ncol = ncol;
        A->ld = ncol;
        double st = H2P_get_wtime_sec();
        H2P_ID_compress(A, QR_REL_NRM, &tol_norm, &U, J);
        double et = H2P_get_wtime_sec();
        ut += et - st;
    }
    printf("U rank = %d, average used time = %.8lf (s)\n", U->ncol, ut / 10.0);
    fflush(stdout);
    
    ouf = fopen("U.csv", "w");
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
    
    //printf("A skeleton rows: ");
    //for (int i = 0; i < U->ncol; i++) printf("%d ", J[i]);
    //printf("\n");
    
    DTYPE *AJ = (DTYPE*) malloc(sizeof(DTYPE) * ncol * U->ncol);
    for (int i = 0; i < U->ncol; i++)
        memcpy(AJ + i * ncol, A0->data + J[i] * ncol, sizeof(DTYPE) * ncol);
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nrow, ncol, U->ncol,
        1.0, U->data, U->ncol, AJ, ncol, -1.0, A0->data, A0->ncol
    );
    DTYPE res_fnorm = 0.0;
    for (int i = 0; i < nrow * ncol; i++) 
        res_fnorm += A0->data[i] * A0->data[i];
    res_fnorm = sqrt(res_fnorm);
    printf("||A - A_{H2}||_fro / ||A||_fro = %e\n", res_fnorm / A0_fnorm);
    
    free(J);
    free(x1);
    free(y1);
    free(x2);
    free(y2);
    H2P_dense_mat_destroy(U);
    H2P_dense_mat_destroy(A);
    H2P_dense_mat_destroy(A0);
    return 0;
}