#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include <mkl.h>

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
    srand48(time(NULL));
    DTYPE *x1 = (DTYPE*) malloc(sizeof(DTYPE) * nrow);
    DTYPE *y1 = (DTYPE*) malloc(sizeof(DTYPE) * nrow);
    DTYPE *x2 = (DTYPE*) malloc(sizeof(DTYPE) * ncol);
    DTYPE *y2 = (DTYPE*) malloc(sizeof(DTYPE) * ncol);
    assert(x1 != NULL && x2 != NULL && y1 != NULL && y2 != NULL);
    for (int i = 0; i < nrow; i++) 
    {
        x1[i] = drand48();
        y1[i] = drand48();
    }
    for (int i = 0; i < ncol; i++) 
    {
        x2[i] = drand48() + 0.6;
        y2[i] = drand48() + 0.4;
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

    H2P_int_vec_t J;
    H2P_int_vec_init(&J, nrow);
    DTYPE tol_norm;
    printf("norm_rel_tol: ");
    scanf("%lf", &tol_norm);
    int nthreads = omp_get_max_threads();
    int *ID_buff = (int*) malloc(sizeof(int) * 4 * A->nrow);
    DTYPE *QR_buff = (DTYPE*) malloc(sizeof(DTYPE) * 2 * A->nrow);
    assert(ID_buff != NULL && QR_buff != NULL);
    H2P_ID_compress(A, QR_REL_NRM, &tol_norm, &U, J, nthreads, QR_buff, ID_buff);  // Warm up
    double ut = 0.0;
    for (int i = 0; i < 10; i++)
    {
        memcpy(A->data, A0->data, sizeof(DTYPE) * nrow * ncol);
        A->nrow = nrow;
        A->ncol = ncol;
        A->ld = ncol;
        double st = H2P_get_wtime_sec();
        H2P_ID_compress(A, QR_REL_NRM, &tol_norm, &U, J, nthreads, QR_buff, ID_buff);
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
        memcpy(AJ + i * ncol, A0->data + J->data[i] * ncol, sizeof(DTYPE) * ncol);
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nrow, ncol, U->ncol,
        1.0, U->data, U->ncol, AJ, ncol, -1.0, A0->data, A0->ncol
    );
    DTYPE res_fnorm = 0.0;
    for (int i = 0; i < nrow * ncol; i++) 
        res_fnorm += A0->data[i] * A0->data[i];
    res_fnorm = sqrt(res_fnorm);
    printf("||A - A_{H2}||_fro / ||A||_fro = %e\n", res_fnorm / A0_fnorm);
    
    free(ID_buff);
    free(QR_buff);
    free(x1);
    free(y1);
    free(x2);
    free(y2);
    H2P_int_vec_destroy(J);
    H2P_dense_mat_destroy(U);
    H2P_dense_mat_destroy(A);
    H2P_dense_mat_destroy(A0);
    return 0;
}