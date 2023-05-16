#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <float.h>

#include "omp.h"
#include "linalg_lib_wrapper.h"
#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "AFN_precond.h"

#if DTYPE_SIZE == DOUBLE_SIZE
#define NEXTAFTER nextafter
#endif
#if DTYPE_SIZE == FLOAT_SIZE
#define NEXTAFTER nextafterf
#endif

// Select k points from X using Farthest Point Sampling (FPS)
// Input parameters:
//   npt    : Number of points in coord
//   pt_dim : Dimension of each point
//   coord  : Matrix, size pt_dim-by-npt, coordinates of points
//   k      : Number of points to select, <= npt
// Output parameter:
//   idx : Vector, size min(k, npt), indices of selected points
void AFNi_FPS(const int npt, const int pt_dim, const DTYPE *coord, const int k, int *idx)
{
    DTYPE *workbuf = (DTYPE *) malloc(sizeof(DTYPE) * (2 * npt + pt_dim));
    DTYPE *center = workbuf;
    DTYPE *tmp_d  = center + pt_dim;
    DTYPE *min_d  = tmp_d  + npt;

    memset(center, 0, sizeof(DTYPE) * pt_dim);
    for (int i = 0; i < pt_dim; i++)
    {
        const DTYPE *coord_i = coord + i * npt;
        #pragma omp simd
        for (int j = 0; j < npt; j++) center[i] += coord_i[j];
        center[i] /= (DTYPE) npt;
    }

    H2P_calc_pdist2_OMP(center, 1, 1, coord, npt, npt, pt_dim, tmp_d, npt, 1);
    for (int j = 0; j < npt; j++) min_d[j] = DBL_MAX;
    DTYPE tmp = tmp_d[0];  idx[0] = 0;
    for (int j = 1; j < npt; j++)
        if (tmp_d[j] < tmp) { tmp = tmp_d[j]; idx[0] = j; }

    for (int i = 1; i < k; i++)
    {
        const DTYPE *coord_ls = coord + idx[i - 1];
        H2P_calc_pdist2_OMP(coord_ls, npt, 1, coord, npt, npt, pt_dim, tmp_d, npt, 1);
        for (int j = 0; j < npt; j++) min_d[j] = (tmp_d[j] < min_d[j]) ? tmp_d[j] : min_d[j];
        tmp = min_d[0];  idx[i] = 0;
        for (int j = 1; j < npt; j++)
            if (min_d[j] > tmp) { tmp = min_d[j]; idx[i] = j; }
    }

    free(workbuf);
}

// Sample and scale ss_npt points, then use them to estimate the rank
// Input and output parameters are the same as AFNi_rank_est()
int AFNi_rank_est_scaled(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const int ss_npt, const int n_rep
)
{
    int sample_r = 0, r = 0;
    DTYPE scale_factor = DPOW((DTYPE) ss_npt / (DTYPE) npt, 1.0 / (DTYPE) pt_dim);
    DTYPE *workbuf = (DTYPE *) malloc(sizeof(DTYPE) * (ss_npt * (pt_dim + 4 * ss_npt)));
    int *FPS_perm = (int *) malloc(sizeof(int) * ss_npt);
    int *sample_idx = (int *) malloc(sizeof(int) * ss_npt);
    DTYPE *Xs   = workbuf;
    DTYPE *Ks   = Xs  + ss_npt * pt_dim;
    DTYPE *K11  = Ks  + ss_npt * ss_npt;
    DTYPE *K1   = K11 + ss_npt * ss_npt;
    DTYPE *Knys = K1  + ss_npt * ss_npt;
    for (int i_rep = 0; i_rep < n_rep; i_rep++)
    {
        // Randomly sample and scale ss_npt points
        H2P_rand_sample(npt, ss_npt, sample_idx, NULL);
        for (int i = 0; i < pt_dim; i++)
        {
            DTYPE *Xs_i = Xs + i * ss_npt;
            const DTYPE *coord_i = coord + i * npt;
            for (int j = 0; j < ss_npt; j++)
                Xs_i[j] = coord_i[sample_idx[j]] * scale_factor;
        }
        // Reorder points using FPS for Nystrom (use Ks as a temporary buffer)
        AFNi_FPS(ss_npt, pt_dim, Xs, ss_npt, FPS_perm);
        H2P_gather_matrix_columns(Xs, ss_npt, Ks, ss_npt, pt_dim, FPS_perm, ss_npt);
        memcpy(Xs, Ks, sizeof(DTYPE) * ss_npt * pt_dim);
        // Shift the Ks matrix to make Nystrom stable
        krnl_eval(Xs, ss_npt, ss_npt, Xs, ss_npt, ss_npt, krnl_param, Ks, ss_npt);
        DTYPE Ks_fnorm = calc_2norm(ss_npt * ss_npt, Ks);
        DTYPE nu = DSQRT((DTYPE) ss_npt) * (NEXTAFTER(Ks_fnorm, Ks_fnorm + 1.0) - Ks_fnorm);
        for (int i = 0; i < ss_npt; i++) Ks[i * ss_npt + i] += nu;
        // Binary search to find the minimal rank
        int rs = 1, re = ss_npt, rc;
        while (rs < re)
        {
            rc = (rs + re) / 2;
            // Since Ks is reordered by FPS, use the first rc  
            // rows & columns as Nystrom basis is the best
            // K11 = Ks(1 : rc, 1 : rc);
            // K1  = Ks(1 : rc, :);
            copy_matrix(sizeof(DTYPE), rc, rc,     Ks, ss_npt, K11, rc,     0);
            copy_matrix(sizeof(DTYPE), rc, ss_npt, Ks, ss_npt, K1,  ss_npt, 0);
            // Knys = K1' * (K11 \ K1);  % K1' * inv(K11) * K1
            int *ipiv = sample_idx;  // sample_idx is no longer needed and its size >= rc
            LAPACK_GETRF(LAPACK_ROW_MAJOR, rc, rc, K11, rc, ipiv);
            LAPACK_GETRS(LAPACK_ROW_MAJOR, 'N', rc, ss_npt, K11, rc, ipiv, K1, ss_npt);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, ss_npt, ss_npt, rc,
                1.0, Ks, ss_npt, K1, ss_npt, 0.0, Knys, ss_npt
            );
            DTYPE err_fnorm, relerr;
            calc_err_2norm(ss_npt * ss_npt, Ks, Knys, &Ks_fnorm, &err_fnorm);
            relerr = err_fnorm / Ks_fnorm;
            if (relerr < 0.1) re = rc; 
            else rs = rc + 1;
        }  // End of while (rs < re)
        sample_r = (rs > sample_r) ? rs : sample_r;
    }  // End of i_rep loop
    r = DCEIL((DTYPE) sample_r * (DTYPE) npt / (DTYPE) ss_npt);
    free(workbuf);
    free(FPS_perm);
    return r;
}

// Use the original points to estimate the rank for Nystrom
// Input and output parameters are the same as AFNi_rank_est()
int AFNi_Nys_rank_est(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const DTYPE mu, const int max_k, const int ss_npt, const int n_rep
)
{
    int sample_r = 0, r = 0, max_k_ = max_k;
    if (max_k_ < ss_npt) max_k_ = ss_npt;
    DTYPE *workbuf = (DTYPE *) malloc(sizeof(DTYPE) * (max_k_ * pt_dim + max_k_ * max_k_ + max_k));
    int *sample_idx = (int *) malloc(sizeof(int) * max_k_);
    DTYPE *Xs = workbuf;
    DTYPE *Ks = Xs + max_k_ * pt_dim;
    DTYPE *ev = Ks + max_k_ * max_k_;
    for (int i_rep = 0; i_rep < n_rep; i_rep++)
    {
        // Randomly sample and scale min(ss_npt, max_k_) points
        DTYPE scale_factor = 1.0;
        int ss_npt_ = max_k_;
        if (max_k_ > ss_npt)
        {
            H2P_rand_sample(npt, ss_npt, sample_idx, NULL);
            scale_factor = DPOW((DTYPE) ss_npt / (DTYPE) max_k_, 1.0 / (DTYPE) pt_dim);
            ss_npt_ = ss_npt;
        } else {
            H2P_rand_sample(npt, max_k_, sample_idx, NULL);
        }
        for (int i = 0; i < pt_dim; i++)
        {
            DTYPE *Xs_i = Xs + i * ss_npt_;
            const DTYPE *coord_i = coord + i * npt;
            for (int j = 0; j < ss_npt_; j++)
                Xs_i[j] = coord_i[sample_idx[j]] * scale_factor;
        }
        // Ks = kernel(Xs, Xs) + mu * eye(size(Xs, 1));
        // ev = eig(Ks);
        krnl_eval(Xs, ss_npt_, ss_npt_, Xs, ss_npt_, ss_npt_, krnl_param, Ks, ss_npt_);
        for (int i = 0; i < ss_npt; i++) Ks[i * ss_npt + i] += mu;
        LAPACK_SYEVD(LAPACK_ROW_MAJOR, 'N', 'U', ss_npt_, Ks, ss_npt_, ev);
        // sample_r = max(sample_r, sum(ev > 1.1 * mu));
        int rc = 0;
        DTYPE threshold = 1.1 * mu;
        for (int i = 0; i < ss_npt_; i++) if (ev[i] > threshold) rc++;
        sample_r = (rc > sample_r) ? rc : sample_r;
    }  // End of i_rep loop
    r = DCEIL((DTYPE) sample_r * (DTYPE) max_k_ / (DTYPE) ss_npt);
    free(sample_idx);
    free(workbuf);
    return r;
}

// Estimate the rank for AFN
// Input parameters:
//   krnl_eval  : Pointer to kernel matrix evaluation function
//   krnl_param : Pointer to kernel function parameter array
//   npt        : Number of points in coord
//   pt_dim     : Dimension of each point
//   coord      : Matrix, size pt_dim-by-npt, coordinates of points
//   mu         : Scalar, diagonal shift of the kernel matrix
//   max_k      : Maximum global low-rank approximation rank (K11's size)
//   ss_npt     : Number of points in the sampling set
//   n_rep      : Number of repetitions for rank estimation
// Output parameter:
//   <ret> : Return the estimated rank
int AFNi_rank_est(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const DTYPE mu, const int max_k, const int ss_npt, const int n_rep
)
{
    int r_scaled = AFNi_rank_est_scaled(krnl_eval, krnl_param, npt, pt_dim, coord, ss_npt, n_rep);
    if (r_scaled > max_k)
    {
        // Estimated rank > max K11 size, will use AFN, return immediately
        return r_scaled;
    } else {
        // Estimated rank is small, will use Nystrom instead of AFN, 
        // use the original points to better estimate the rank
        int r_unscaled = AFNi_Nys_rank_est(krnl_eval, krnl_param, npt, pt_dim, coord, mu, max_k, ss_npt, n_rep);
        return r_unscaled;
    }
}

// CSR SpMV for the G matrix in AFN
void AFNi_CSR_SpMV(
    const int nrow, const int *row_ptr, const int *col_idx, 
    const DTYPE *val, const DTYPE *x, DTYPE *y
)
{
    // In the first few n1 rows, each row has less than fsai_npt nonzeros;
    // after that, each row has exactly fsai_npt nonzeros. Using a static
    // partitioning scheme should be good enough.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nrow; i++)
    {
        DTYPE res = 0;
        #pragma omp simd
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            res += val[j] * x[col_idx[j]];
        y[i] = res;
    }
}

// Build the Nystrom preconditioner
// See AFN_precond_init() for input parameters
void AFNi_Nys_precond_build(AFN_precond_p AFN_precond, const DTYPE mu, DTYPE *K11, DTYPE *K12)
{
    AFN_precond->is_nys = 1;
    AFN_precond->is_afn = 0;
    int n  = AFN_precond->n;
    int n1 = AFN_precond->n1;
    int n2 = AFN_precond->n2;
    
    // K1 = [K11, K12];
    DTYPE *K1 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n);
    copy_matrix(sizeof(DTYPE), n1, n1, K11, n1, K1,      n, 1);
    copy_matrix(sizeof(DTYPE), n1, n2, K12, n2, K1 + n1, n, 1);

    // nu = sqrt(n) * norm(K1, 'fro') * eps;
    DTYPE nu = 0;
    #pragma omp parallel for schedule(static) reduction(+:nu)
    for (int i = 0; i < n * n1; i++) nu += K1[i] * K1[i];
    nu = DSQRT((DTYPE) n) * DSQRT(nu) * D_EPS;

    // K11 = K11 + nu * eye(n1);
    for (int i = 0; i < n1; i++) K11[i * n1 + i] += nu;
    // invL = inv(chol(K11, 'lower'));
    DTYPE *invL = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    int info = 0;
    info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', n1, K11, n1);
    ASSERT_PRINTF(info == 0, "LAPACK_POTRF failed, info = %d\n", info);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n1 * n1; i++) invL[i] = 0;
    for (int i = 0; i < n1; i++) invL[i * n1 + i] = 1.0;
    CBLAS_TRSM(
        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, 
        CblasNonUnit, n1, n1, 1.0, K11, n1, invL, n1
    );
    // M = K1' * invL';
    DTYPE *M = (DTYPE *) malloc(sizeof(DTYPE) * n * n1);
    CBLAS_GEMM(
        CblasRowMajor, CblasTrans, CblasTrans, n, n1, n1, 
        1.0, K1, n, invL, n1, 0.0, M, n1
    );
    free(K1);
    free(invL);

    // [U, S, ~] = svd(M, 0);
    DTYPE *S = (DTYPE *) malloc(sizeof(DTYPE) * n1);
    //#define NYSTROM_SVD_DIRECT
    #ifdef NYSTROM_SVD_DIRECT
    DTYPE *superb = (DTYPE *) malloc(sizeof(DTYPE) * n1);
    info = LAPACK_GESVD(
        LAPACK_ROW_MAJOR, 'O', 'N', n, n1, M, n1, 
        S, NULL, n1, NULL, n1, superb
    );
    ASSERT_PRINTF(info == 0, "LAPACK_GESVD failed, info = %d\n", info);
    free(superb);
    AFN_precond->nys_U = M;
    int min_eigval_idx = n1 - 1;
    #else
    // Use EVD is usually faster but may be less accurate
    // MKL with ICC 19.1.3 has a bug in LAPACK_GESVD so we have to use EVD instead
    // MTM = M' * M;
    DTYPE *MTM = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    CBLAS_GEMM(
        CblasRowMajor, CblasTrans, CblasNoTrans, n1, n1, n, 
        1.0, M, n1, M, n1, 0.0, MTM, n1
    );
    // [V, S] = eig(MTM);
    // S = sqrt(S);
    // V = V * inv(S);
    info = LAPACK_SYEVD(LAPACK_ROW_MAJOR, 'V', 'U', n1, MTM, n1, S);
    for (int i = 0; i < n1; i++) S[i] = DSQRT(S[i]);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n1; i++)
        for (int j = 0; j < n1; j++) MTM[i * n1 + j] /= S[j];
    // U = M * V;
    DTYPE *U = (DTYPE *) malloc(sizeof(DTYPE) * n * n1);
    CBLAS_GEMM(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n1, n1, 
        1.0, M, n1, MTM, n1, 0.0, U, n1
    );
    AFN_precond->nys_U = U;
    free(M);
    free(MTM);
    int min_eigval_idx = 0;
    #endif

    // S = max(diag(S).^2 - nu, 0);
    // eta = S(n1) + mu;
    // nys_M = eta ./ (S + mu);
    DTYPE *nys_M = (DTYPE *) malloc(sizeof(DTYPE) * n1);
    for (int i = 0; i < n1; i++)
    {
        S[i] = S[i] * S[i] - nu;
        if (S[i] < 0) S[i] = 0;
    }
    DTYPE eta = S[min_eigval_idx] + mu;
    for (int i = 0; i < n1; i++) nys_M[i] = eta / (S[i] + mu);
    free(S);
    AFN_precond->nys_M = nys_M;
}

// Quick sort for (DTYPE, int) key-value pairs
void AFNi_qsort_DTYPE_int_key_val(DTYPE *key, int *val, const int l, const int r)
{
    int i = l, j = r, tmp_val;
    DTYPE tmp_key, mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            tmp_key = key[i]; key[i] = key[j]; key[j] = tmp_key;
            tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) AFNi_qsort_DTYPE_int_key_val(key, val, i, r);
    if (j > l) AFNi_qsort_DTYPE_int_key_val(key, val, l, j);
}

// Quick sort for (int, DTYPE) key-value pairs
void AFNi_qsort_int_DTYPE_key_val(int *key, DTYPE *val, const int l, const int r)
{
    int i = l, j = r, tmp_key;
    int mid_key = key[(l + r) / 2];
    DTYPE tmp_val;
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            tmp_key = key[i]; key[i] = key[j]; key[j] = tmp_key;
            tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) AFNi_qsort_int_DTYPE_key_val(key, val, i, r);
    if (j > l) AFNi_qsort_int_DTYPE_key_val(key, val, l, j);
}

// Convert a COO matrix to a sorted CSR matrix
void AFNi_COO2CSR(
    const int nrow, const int ncol, const int nnz,
    const int *row, const int *col, const DTYPE *val, 
    int **row_ptr_, int **col_idx_, DTYPE **csr_val_
)
{
    int *row_ptr = (int *) malloc(sizeof(int) * (nrow + 1));
    int *col_idx = (int *) malloc(sizeof(int) * nnz);
    DTYPE *csr_val = (DTYPE *) malloc(sizeof(DTYPE) * nnz);

    // Get the number of non-zeros in each row
    memset(row_ptr, 0, sizeof(int) * (nrow + 1));
    for (int i = 0; i < nnz; i++) row_ptr[row[i] + 1]++;

    // Calculate the displacement of 1st non-zero in each row
    for (int i = 2; i <= nrow; i++) row_ptr[i] += row_ptr[i - 1];

    // Use row_ptr to bucket sort col[] and val[]
    for (int i = 0; i < nnz; i++)
    {
        int idx = row_ptr[row[i]];
        col_idx[idx] = col[i];
        csr_val[idx] = val[i];
        row_ptr[row[i]]++;
    }

    // Reset row_ptr
    for (int i = nrow; i >= 1; i--) row_ptr[i] = row_ptr[i - 1];
    row_ptr[0] = 0;

    // Sort the non-zeros in each row according to column indices
    #pragma omp parallel for
    for (int i = 0; i < nrow; i++)
        AFNi_qsort_int_DTYPE_key_val(col_idx, csr_val, row_ptr[i], row_ptr[i + 1] - 1);

    *row_ptr_ = row_ptr;
    *col_idx_ = col_idx;
    *csr_val_ = csr_val;
}

// Build the AFN preconditioner
// See AFN_precond_init() for input parameters
void AFNi_AFN_precond_build(
    AFN_precond_p AFN_precond, const DTYPE mu, DTYPE *K11, DTYPE *K12, 
    const DTYPE *coord2, const int ld2, const int pt_dim, 
    kernel_eval_fptr krnl_eval, void *krnl_param, const int fsai_npt
)
{
    AFN_precond->is_nys = 0;
    AFN_precond->is_afn = 1;
    int n1 = AFN_precond->n1;
    int n2 = AFN_precond->n2;
    double st, et;
    double st0 = get_wtime_sec();

    st = get_wtime_sec();
    // K11 = K11 + mu * eye(n1);
    for (int i = 0; i < n1; i++) K11[i * n1 + i] += mu;
    // invL = inv(chol(K11, 'lower'));
    DTYPE *invL = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    int info = 0;
    info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', n1, K11, n1);
    ASSERT_PRINTF(info == 0, "LAPACK_POTRF failed, info = %d\n", info);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n1 * n1; i++) invL[i] = 0;
    for (int i = 0; i < n1; i++) invL[i * n1 + i] = 1.0;
    CBLAS_TRSM(
        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, 
        CblasNonUnit, n1, n1, 1.0, K11, n1, invL, n1
    );
    // V21 = K12' * invL';
    AFN_precond->afn_K12 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n2);
    #pragma omp parallel for schedule(static) 
    for (int i = 0; i < n1 * n2; i++) AFN_precond->afn_K12[i] = K12[i];
    DTYPE *V21 = (DTYPE *) malloc(sizeof(DTYPE) * n2 * n1);
    CBLAS_GEMM(
        CblasRowMajor, CblasTrans, CblasTrans, n2, n1, n1, 
        1.0, K12, n2, invL, n1, 0.0, V21, n1
    );
    free(invL);

    // Copy K12 and invert K11, currently K11 is overwritten by its Cholesky factor
    DTYPE *invK11  = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    DTYPE *K12_    = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n2);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n1; i++)
    {
        DTYPE *K11_i     = K11     + i * n1;
        DTYPE *invK11_i  = invK11  + i * n1;
        DTYPE *K12_i     = K12     + i * n2;
        DTYPE *K12_i_    = K12_    + i * n2;
        memcpy(invK11_i, K11_i, sizeof(DTYPE) * n1);
        memcpy(K12_i_,   K12_i, sizeof(DTYPE) * n2);
    }
    info = LAPACK_POTRI(LAPACK_ROW_MAJOR, 'L', n1, invK11, n1);
    ASSERT_PRINTF(info == 0, "LAPACK_POTRI failed, info = %d\n", info);
    // POTRI only returns the lower/upper triangular part, remember to symmetrize
    for (int i = 0; i < n1 - 1; i++)
        for (int j = i + 1; j < n1; j++)
            invK11[i * n1 + j] = invK11[j * n1 + i];
    AFN_precond->afn_invK11 = invK11;
    AFN_precond->afn_K12    = K12_;
    et = get_wtime_sec();
    AFN_precond->t_afn_mat = et - st;

    // FSAI for S = K22 - K12' * (K11 \ K12)
    int n_thread = omp_get_max_threads();
    int thread_mat_bufsize = fsai_npt * (pt_dim + fsai_npt + n2);
    int   *nn     = (int *)   malloc(sizeof(int)   * n2 * fsai_npt);
    int   *idxbuf = (int *)   malloc(sizeof(int)   * n_thread * n2);
    DTYPE *matbuf = (DTYPE *) malloc(sizeof(DTYPE) * n_thread * thread_mat_bufsize);
    int   *row    = (int *)   malloc(sizeof(int)   * n2 * fsai_npt);
    int   *col    = (int *)   malloc(sizeof(int)   * n2 * fsai_npt);
    DTYPE *val    = (DTYPE *) malloc(sizeof(DTYPE) * n2 * fsai_npt);
    int   *displs = (int *)   malloc(sizeof(int)   * (n2 + 1));
    displs[0] = 0;
    st = get_wtime_sec();
    BLAS_SET_NUM_THREADS(1);
    #pragma omp parallel if (n_thread > 1) num_threads(n_thread)
    {
        // In the first few fsai_npt rows, each row has less than fsai_npt nonzeros;
        // after that, each row has exactly fsai_npt nonzeros. Using a static
        // partitioning scheme should be good enough.
        int tid = omp_get_thread_num();
        int n2_start, n2_size;
        calc_block_spos_len(n2, n_thread, tid, &n2_start, &n2_size);
        int *nn_i = idxbuf + tid * n2;
        DTYPE *thread_matbuf = matbuf + tid * thread_mat_bufsize;
        DTYPE *nn_coord = thread_matbuf;
        DTYPE *tmpK     = nn_coord + fsai_npt * pt_dim;
        DTYPE *Vnn      = tmpK     + fsai_npt * fsai_npt;
        DTYPE *dist2_i  = Vnn;   // Vnn is not used at the beginning and size >= n2, reuse it
        // Do the large chunk of work first
        #pragma omp for schedule(dynamic, 256)
        for (int i = n2 - 1; i >= 0; i--)
        {
            int num_nn = 0;
            if (i < fsai_npt)
            {
                num_nn = i + 1;
                for (int j = 0; j < num_nn; j++) nn_i[j] = j;
            } else {
                num_nn = fsai_npt;
                // pdist2_i = pdist2(Xperm2(i, :), Xperm2(1 : i-1, :));
                H2P_calc_pdist2_OMP(coord2 + i, ld2, 1, coord2, ld2, i, pt_dim, dist2_i, n2, 1);
                // [~, idx_i] = sort(pdist2_i);
                // nn = [idx_i(1 : num_nn-1), i];
                for (int j = 0; j < i; j++) nn_i[j] = j;
                AFNi_qsort_DTYPE_int_key_val(dist2_i, nn_i, 0, i - 1);
                nn_i[num_nn - 1] = i;
            }
            displs[i + 1] = num_nn;
            // row(idx + (1 : num_nn)) = i;
            // col(idx + (1 : num_nn)) = nn;
            // Vnn = V21(nn, :);
            // Xnn = Xperm2(nn, :);
            for (int j = i * fsai_npt; j < i * fsai_npt + num_nn; j++)
            {
                int j0 = j - i * fsai_npt;
                row[j] = i;
                col[j] = nn_i[j0];
                memcpy(Vnn + j0 * n1, V21 + nn_i[j0] * n1, sizeof(DTYPE) * n1);
            }
            H2P_gather_matrix_columns(coord2, ld2, nn_coord, num_nn, pt_dim, nn_i, num_nn);
            // tmpK = kernel(Xnn, Xnn) + mu * eye(num_nn);
            // tmpK = tmpK - Vnn * Vnn';
            krnl_eval(nn_coord, num_nn, num_nn, nn_coord, num_nn, num_nn, krnl_param, tmpK, num_nn);
            for (int j = 0; j < num_nn; j++) tmpK[j * num_nn + j] += mu;
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasTrans, num_nn, num_nn, n1,
                -1.0, Vnn, n1, Vnn, n1, 1.0, tmpK, num_nn
            );
            // tmpU = [zeros(num_nn-1, 1); 1];
            // tmpY = tmpK \ tmpU;
            DTYPE *tmpU = val + i * fsai_npt;
            memset(tmpU, 0, sizeof(DTYPE) * num_nn);
            tmpU[num_nn - 1] = 1.0;
            // The standard way is using LAPACK_ROW_MAJOR here, but since tmpK is 
            // symmetric, we use LAPACK_COL_MAJOR to avoid internal transpose
            info = LAPACK_GETRF(LAPACK_COL_MAJOR, num_nn, num_nn, tmpK, num_nn, nn_i);
            ASSERT_PRINTF(info == 0, "LAPACK_GETRF return %d\n", info);
            info = LAPACK_GETRS(LAPACK_COL_MAJOR, 'N', num_nn, 1, tmpK, num_nn, nn_i, tmpU, num_nn);
            ASSERT_PRINTF(info == 0, "LAPACK_GETRS return %d\n", info);
            // val(idx + (1 : num_nn)) = tmpY / sqrt(tmpY(num_nn));
            DTYPE scale_factor = 1.0 / DSQRT(tmpU[num_nn - 1]);
            for (int j = 0; j < num_nn; j++) tmpU[j] *= scale_factor;
        }  // End of i loop
    }  // End of "#pragma omp parallel"
    BLAS_SET_NUM_THREADS(n_thread);
    free(nn);
    free(idxbuf);
    free(matbuf);
    free(V21);
    et = get_wtime_sec();
    AFN_precond->t_afn_fsai = et - st;

    // Build G and G^T CSR matrices
    st = get_wtime_sec();
    for (int i = 1; i <= n2; i++) displs[i] += displs[i - 1];
    int nnz = displs[n2];
    int   *row1 = (int *)   malloc(sizeof(int)   * nnz);
    int   *col1 = (int *)   malloc(sizeof(int)   * nnz);
    DTYPE *val1 = (DTYPE *) malloc(sizeof(DTYPE) * nnz);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n2; i++)
    {
        int nnz_i = displs[i + 1] - displs[i];
        memcpy(row1 + displs[i], row + i * fsai_npt, sizeof(int) * nnz_i);
        memcpy(col1 + displs[i], col + i * fsai_npt, sizeof(int) * nnz_i);
        memcpy(val1 + displs[i], val + i * fsai_npt, sizeof(DTYPE) * nnz_i);
    }
    free(row);
    free(col);
    free(val);
    free(displs);
    int *G_rowptr = NULL, *G_colidx = NULL, *GT_rowptr = NULL, *GT_colidx = NULL;
    DTYPE *G_val = NULL, *GT_val = NULL;
    AFNi_COO2CSR(n2, n2, nnz, row1, col1, val1, &G_rowptr, &G_colidx, &G_val);
    AFNi_COO2CSR(n2, n2, nnz, col1, row1, val1, &GT_rowptr, &GT_colidx, &GT_val);
    AFN_precond->afn_G_rowptr  = G_rowptr;
    AFN_precond->afn_G_colidx  = G_colidx;
    AFN_precond->afn_G_val     = G_val;
    AFN_precond->afn_GT_rowptr = GT_rowptr;
    AFN_precond->afn_GT_colidx = GT_colidx;
    AFN_precond->afn_GT_val    = GT_val;
    free(row1);
    free(col1);
    free(val1);
    et = get_wtime_sec();
    AFN_precond->t_afn_csr = et - st;

    double et0 = get_wtime_sec();
    AFN_precond->t_afn = et0 - st0;
}

// Construct an AFN preconditioner for a kernel matrix
void AFN_precond_build(
    kernel_eval_fptr krnl_eval, void *krnl_param, const int npt, const int pt_dim, 
    const DTYPE *coord, const DTYPE mu, const int max_k, const int ss_npt,
    const int fsai_npt, AFN_precond_p *AFN_precond_
)
{
    AFN_precond_p AFN_precond = (AFN_precond_p) malloc(sizeof(AFN_precond_s));
    memset(AFN_precond, 0, sizeof(AFN_precond_s));
    double st, et, st0, et0;
    st0 = get_wtime_sec();

    // 1. Estimate the numerical low rank of the kernel matrix + diagonal shift
    st = get_wtime_sec();
    int est_rank = AFNi_rank_est(krnl_eval, krnl_param, npt, pt_dim, coord, mu, max_k, ss_npt, 1);
    int n  = npt;
    int n1 = (est_rank < max_k) ? est_rank : max_k;
    int n2 = n - n1;
    AFN_precond->n  = n;
    AFN_precond->n1 = n1;
    AFN_precond->n2 = n2;
    AFN_precond->px = (DTYPE *) malloc(sizeof(DTYPE) * n);
    AFN_precond->py = (DTYPE *) malloc(sizeof(DTYPE) * n);
    AFN_precond->t1 = (DTYPE *) malloc(sizeof(DTYPE) * n);
    AFN_precond->t2 = (DTYPE *) malloc(sizeof(DTYPE) * n);
    et = get_wtime_sec();
    AFN_precond->t_rankest = et - st;
    AFN_precond->est_rank = est_rank;

    // 2. Use FPS to select n1 points, swap them to the front
    st = get_wtime_sec();
    AFN_precond->perm = (int *) malloc(sizeof(int) * n);
    int *perm = AFN_precond->perm;
    AFNi_FPS(npt, pt_dim, coord, n1, perm);
    uint8_t *flag = (uint8_t *) malloc(sizeof(uint8_t) * n);
    memset(flag, 0, sizeof(uint8_t) * n);
    for (int i = 0; i < n1; i++) flag[perm[i]] = 1;
    int idx = n1;
    for (int i = 0; i < n; i++)
        if (flag[i] == 0) perm[idx++] = i;
    DTYPE *coord_perm = (DTYPE *) malloc(sizeof(DTYPE) * npt * pt_dim);
    H2P_gather_matrix_columns(coord, npt, coord_perm, npt, pt_dim, perm, npt);
    et = get_wtime_sec();
    AFN_precond->t_fps= et - st;

    // 3. Calculate K11 and K12 used for both Nystrom and AFN
    st = get_wtime_sec();
    DTYPE *coord_n1 = coord_perm;
    DTYPE *coord_n2 = coord_perm + n1;
    DTYPE *K11 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n1);
    DTYPE *K12 = (DTYPE *) malloc(sizeof(DTYPE) * n1 * n2);
    #pragma omp parallel
    {
        int n_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int n1_blk_start, n1_blk_len;
        calc_block_spos_len(n1, n_thread, tid, &n1_blk_start, &n1_blk_len);
        DTYPE *K11_srow = K11 + n1_blk_start * n1;
        DTYPE *K12_srow = K12 + n1_blk_start * n2;    
        DTYPE *coord_n1_spos = coord_n1 + n1_blk_start;
        krnl_eval(
            coord_n1_spos, n, n1_blk_len, coord_n1, n, n1, 
            krnl_param, K11_srow, n1
        );
        krnl_eval(
            coord_n1_spos, n, n1_blk_len, coord_n2, n, n2, 
            krnl_param, K12_srow, n2
        );
    }  // End of "#pragma omp parallel"
    et = get_wtime_sec();
    AFN_precond->t_K11K12= et - st;

    // 4. Build the Nystrom or AFN preconditioner
    if (est_rank < max_k)
    {
        st = get_wtime_sec();
        AFNi_Nys_precond_build(AFN_precond, mu, K11, K12);
        et = get_wtime_sec();
        AFN_precond->t_nys = et - st;
    } else {
        AFNi_AFN_precond_build(
            AFN_precond, mu, K11, K12, coord_n2, n, pt_dim, 
            krnl_eval, krnl_param, fsai_npt
        );
    }

    free(flag);
    free(coord_perm);
    free(K11);
    free(K12);

    et0 = get_wtime_sec();
    AFN_precond->t_build = et0 - st0;
    *AFN_precond_ = AFN_precond;
}

// Destroy an initialized AFN_precond struct
void AFN_precond_destroy(AFN_precond_p *AFN_precond_)
{
    AFN_precond_p AFN_precond = *AFN_precond_;
    if (AFN_precond == NULL) return;
    free(AFN_precond->perm);
    free(AFN_precond->px);
    free(AFN_precond->py);
    free(AFN_precond->t1);
    free(AFN_precond->t2);
    free(AFN_precond->nys_U);
    free(AFN_precond->nys_M);
    free(AFN_precond->afn_G_rowptr);
    free(AFN_precond->afn_GT_rowptr);
    free(AFN_precond->afn_G_colidx);
    free(AFN_precond->afn_GT_colidx);
    free(AFN_precond->afn_G_val);
    free(AFN_precond->afn_GT_val);
    free(AFN_precond->afn_invK11);
    free(AFN_precond->afn_K12);
    free(AFN_precond);
    *AFN_precond_ = NULL;
}

// Apply an AFN preconditioner to a vector
void AFN_precond_apply(AFN_precond_p AFN_precond, const DTYPE *x, DTYPE *y)
{
    if (AFN_precond == NULL) return;
    int   n     = AFN_precond->n;
    int   n1    = AFN_precond->n1;
    int   n2    = AFN_precond->n2;
    int   *perm = AFN_precond->perm;
    DTYPE *px   = AFN_precond->px;
    DTYPE *py   = AFN_precond->py;
    DTYPE *t1   = AFN_precond->t1;
    DTYPE *t2   = AFN_precond->t2;
    double st = get_wtime_sec();

    // px = x(perm);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) px[i] = x[perm[i]];
    if (AFN_precond->is_nys)
    {
        DTYPE *nys_U = AFN_precond->nys_U;
        DTYPE *nys_M = AFN_precond->nys_M;
        // t1 = U' * px;
        CBLAS_GEMV(CblasRowMajor, CblasTrans, n, n1, 1.0, nys_U, n1, px, 1, 0.0, t1, 1);
        // py = px - U * t1;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) py[i] = px[i];
        CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n, n1, -1.0, nys_U, n1, t1, 1, 1.0, py, 1);
        // t1 = M .* t1;
        for (int i = 0; i < n1; i++) t1[i] *= nys_M[i];
        // py = py + U * t1;
        CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n, n1, 1.0, nys_U, n1, t1, 1, 1.0, py, 1);
    }
    if (AFN_precond->is_afn)
    {
        int   *afn_G_rowptr  = AFN_precond->afn_G_rowptr;
        int   *afn_GT_rowptr = AFN_precond->afn_GT_rowptr;
        int   *afn_G_colidx  = AFN_precond->afn_G_colidx;
        int   *afn_GT_colidx = AFN_precond->afn_GT_colidx;
        DTYPE *afn_G_val     = AFN_precond->afn_G_val;
        DTYPE *afn_GT_val    = AFN_precond->afn_GT_val;
        DTYPE *afn_invK11    = AFN_precond->afn_invK11;
        DTYPE *afn_K12       = AFN_precond->afn_K12;
        // x1 = px(1 : n1, :);
        // x2 = px(n1+1 : n, :);
        // t11 = iK11 * x1;  % Size n1
        DTYPE *x1 = px, *x2 = px + n1;
        DTYPE *t11 = t1, *t12 = t1 + n1;
        CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n1, n1, 1.0, afn_invK11, n1, x1, 1, 0.0, t11, 1);
        // t12 = x2 - K12' * t11;  % Size n2
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n2; i++) t12[i] = x2[i];
        CBLAS_GEMV(CblasRowMajor, CblasTrans, n1, n2, -1.0, afn_K12, n2, t11, 1, 1.0, t12, 1);
        // t22 = G * t12;  % Size n2
        DTYPE *y1 = py, *y2 = py + n1;
        DTYPE *t21 = t2, *t22 = t2 + n1;
        AFNi_CSR_SpMV(n2, afn_G_rowptr, afn_G_colidx, afn_G_val, t12, t22);
        // y2 = G' * t22;
        AFNi_CSR_SpMV(n2, afn_GT_rowptr, afn_GT_colidx, afn_GT_val, t22, y2);
        // t21 = x1 - K12 * y2;  % Size n1
        for (int i = 0; i < n1; i++) t21[i] = x1[i];
        CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n1, n2, -1.0, afn_K12, n2, y2, 1, 1.0, t21, 1);
        // y1 = iK11 * t21;
        CBLAS_GEMV(CblasRowMajor, CblasNoTrans, n1, n1, 1.0, afn_invK11, n1, t21, 1, 0.0, y1, 1);
        // py = [y1; y2];
    }
    // y(perm) = py;
    // Parallelizing the first loop may have severe false sharing issue
    for (int i = 0; i < n1; i++) y[perm[i]] = py[i];
    #pragma omp parallel for schedule(static)
    for (int i = n1; i < n; i++) y[perm[i]] = py[i];

    double et = get_wtime_sec();
    AFN_precond->t_apply += et - st;
    AFN_precond->n_apply++;
}

void AFN_precond_print_stat(AFN_precond_p AFN_precond)
{
    if (AFN_precond == NULL) return;
    printf("AFN preconditioner build time   = %.3f s\n", AFN_precond->t_build);
    printf("  * Rank estimation             = %.3f s\n", AFN_precond->t_rankest);
    printf("  * FPS select & permute        = %.3f s\n", AFN_precond->t_fps);
    printf("  * Build K11 and K12 blocks    = %.3f s\n", AFN_precond->t_K11K12);
    printf("  * Build Nystrom precond       = %.3f s\n", AFN_precond->t_nys);
    printf("  * Build AFN precond           = %.3f s\n", AFN_precond->t_afn);
    printf("    * Matrix operations         = %.3f s\n", AFN_precond->t_afn_mat);
    printf("    * Build FSAI                = %.3f s\n", AFN_precond->t_afn_fsai);
    printf("    * FASI matrix COO to CSR    = %.3f s\n", AFN_precond->t_afn_csr);
    if (AFN_precond->n_apply > 0)
    {
        double t_apply_avg = AFN_precond->t_apply / (double) AFN_precond->n_apply;
        printf("Apply preconditioner: %d times, per apply: %.3f s\n", AFN_precond->n_apply, t_apply_avg);
    }
}