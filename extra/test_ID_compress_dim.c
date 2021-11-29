#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "H2Pack_aux_structs.h"
#include "H2Pack_ID_compress.h"
#include "utils.h"

void RPY_kernel_3d(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const int dim, DTYPE *mat, const int ldm
)
{
    const DTYPE a = 1.0, eta = 1.0;
    const DTYPE C   = 1.0 / (6.0 * M_PI * a * eta);
    const DTYPE aa  = a * a;
    const DTYPE a2  = 2.0 * a;
    const DTYPE aa2 = aa * 2.0;
    const DTYPE aa_2o3   = aa2 / 3.0;
    const DTYPE C_075    = C * 0.75;
    const DTYPE C_9o32oa = C * 9.0 / 32.0 / a;
    const DTYPE C_3o32oa = C * 3.0 / 32.0 / a;
    for (int i = 0; i < n0; i++)
    {
        DTYPE x0 = coord0[i];
        DTYPE y0 = coord0[i + ld0];
        DTYPE z0 = coord0[i + ld0 * 2];
        for (int j = 0; j < n1; j++)
        {
            DTYPE r0 = x0 - coord1[j];
            DTYPE r1 = y0 - coord1[j + ld1];
            DTYPE r2 = z0 - coord1[j + ld1 * 2];
            DTYPE s2 = r0 * r0 + r1 * r1 + r2 * r2;
            DTYPE s  = DSQRT(s2);
            DTYPE inv_s = 1.0 / s;
            r0 *= inv_s;
            r1 *= inv_s;
            r2 *= inv_s;
            DTYPE t1, t2;
            if (s < a2)
            {
                t1 = C - C_9o32oa * s;
                t2 = C_3o32oa * s;
            } else {
                t1 = C_075 / s * (1 + aa_2o3 / s2);
                t2 = C_075 / s * (1 - aa2 / s2); 
            }
            int base = 3 * i * ldm + 3 * j;
            #define krnl(k, l) mat[base + k * ldm + l]
            krnl(0, 0) = t2 * r0 * r0 + t1;
            krnl(0, 1) = t2 * r0 * r1;
            krnl(0, 2) = t2 * r0 * r2;
            krnl(1, 0) = t2 * r1 * r0;
            krnl(1, 1) = t2 * r1 * r1 + t1;
            krnl(1, 2) = t2 * r1 * r2;
            krnl(2, 0) = t2 * r2 * r0;
            krnl(2, 1) = t2 * r2 * r1;
            krnl(2, 2) = t2 * r2 * r2 + t1;
        }
    }
}


int main()
{
    int nrow, ncol, kdim = 3;
    printf("matrix size: ");
    scanf("%d%d", &nrow, &ncol);
    int A_nrow = nrow * kdim;
    int A_ncol = ncol * kdim;
    DTYPE tol_norm;
    printf("norm_rel_tol: ");
    scanf(DTYPE_FMTSTR, &tol_norm);
    
    H2P_dense_mat_p A, A0, U;
    H2P_int_vec_p J;
    H2P_dense_mat_init(&A, A_nrow, A_ncol);
    H2P_dense_mat_init(&A0, A_nrow, A_ncol);
    H2P_int_vec_init(&J, A_nrow);
    
    DTYPE *coord0 = (DTYPE*) malloc(sizeof(DTYPE) * A_nrow);
    DTYPE *coord1 = (DTYPE*) malloc(sizeof(DTYPE) * A_ncol);
    assert(coord0 != NULL && coord1 != NULL);
    DTYPE *x0 = coord0, *x1 = coord1;
    DTYPE *y0 = coord0 + nrow, *y1 = coord1 + ncol;
    DTYPE *z0 = coord0 + nrow * 2, *z1 = coord1 + ncol * 2;
    for (int i = 0; i < nrow; i++) 
    {
        x0[i] = (DTYPE) drand48();
        y0[i] = (DTYPE) drand48();
        z0[i] = (DTYPE) drand48();
    }
    for (int i = 0; i < ncol; i++) 
    {
        x1[i] = (DTYPE) (drand48() + 1.9);
        y1[i] = (DTYPE) (drand48() + 0.8);
        z1[i] = (DTYPE) (drand48() + 0.9);
    }
    
    RPY_kernel_3d(
        coord0, nrow, nrow, 
        coord1, ncol, ncol, 
        1, A->data, A_ncol
    );
    memcpy(A0->data, A->data, sizeof(DTYPE) * A_nrow * A_ncol);
    DTYPE A0_fnorm = 0.0;
    for (int i = 0; i < A_nrow * A_ncol; i++)
        A0_fnorm += A->data[i] * A->data[i];
    
    int n_thread = omp_get_max_threads();
    int QR_buff_size = (2 * kdim + 2) * A->ncol + (kdim + 1) * A->nrow;
    int   *ID_buff = (int *)   malloc(sizeof(int)   * A->nrow * 4);
    DTYPE *QR_buff = (DTYPE *) malloc(sizeof(DTYPE) * QR_buff_size);
    double st = get_wtime_sec();
    H2P_ID_compress(
        A, QR_REL_NRM, &tol_norm, &U, J, 
        n_thread, QR_buff, ID_buff, kdim
    );
    double ut = get_wtime_sec() - st;
    printf("H2P_ID_compress used %.3lf s\n", ut);
    
    DTYPE *AJ = (DTYPE*) malloc(sizeof(DTYPE) * U->ncol * A_ncol);
    for (int i = 0; i < J->length; i++)
    {
        int i30 = i * 3 + 0;
        int i31 = i * 3 + 1;
        int i32 = i * 3 + 2;
        int j30 = J->data[i] * 3 + 0;
        int j31 = J->data[i] * 3 + 1;
        int j32 = J->data[i] * 3 + 2;
        memcpy(AJ + i30*A_ncol, A0->data + j30*A_ncol, sizeof(DTYPE) * A_ncol);
        memcpy(AJ + i31*A_ncol, A0->data + j31*A_ncol, sizeof(DTYPE) * A_ncol);
        memcpy(AJ + i32*A_ncol, A0->data + j32*A_ncol, sizeof(DTYPE) * A_ncol);
    }
    CBLAS_GEMM(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrow, A_ncol, U->ncol,
        1.0, U->data, U->ncol, AJ, A_ncol, -1.0, A0->data, A_ncol
    );
    DTYPE res_fnorm = 0.0;
    for (int i = 0; i < A_nrow * A_ncol; i++)
        res_fnorm += A0->data[i] * A0->data[i];
    res_fnorm = DSQRT(res_fnorm);
    printf("U rank = %d (%d column blocks)\n", U->ncol, J->length);
    printf("||A - A_{ID}||_fro / ||A||_fro = %e\n", res_fnorm / A0_fnorm);
    
    free(QR_buff);
    free(ID_buff);
    free(coord0);
    free(coord1);
    H2P_int_vec_destroy(&J);
    H2P_dense_mat_destroy(&U);
    H2P_dense_mat_destroy(&A);
    H2P_dense_mat_destroy(&A0);
    return 0;
}