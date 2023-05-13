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

// Select k points from X using Farthest Point Sampling (FPS)
// Input parameters:
//   npt    : Number of points in coord
//   pt_dim : Dimension of each point
//   coord  : Matrix, size pt_dim-by-npt, coordinates of points
//   k      : Number of points to select, <= npt
// Output parameter:
//   idx : Vector, size min(k, npt), indices of selected points
void AFNi_FPS(const int npt, const int pt_dim, DTYPE *coord, const int k, int *idx)
{
    DTYPE *workbuf = (DTYPE *) malloc(sizeof(DTYPE) * (2 * npt + pt_dim));
    DTYPE *center = workbuf;
    DTYPE *tmp_d  = center + pt_dim;
    DTYPE *min_d  = tmp_d  + npt;

    memset(center, 0, sizeof(DTYPE) * pt_dim);
    for (int i = 0; i < pt_dim; i++)
    {
        DTYPE *coord_i = coord + i * npt;
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
        DTYPE *coord_ls = coord + idx[i - 1];
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
    DTYPE Ks_fnorm, err_fnorm, relerr;
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
        DTYPE nu = DSQRT((DTYPE) ss_npt) * calc_2norm(ss_npt * ss_npt, Ks) * D_EPS;
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
        return AFNi_Nys_rank_est(krnl_eval, krnl_param, npt, pt_dim, coord, mu, max_k, ss_npt, n_rep);
    }
}