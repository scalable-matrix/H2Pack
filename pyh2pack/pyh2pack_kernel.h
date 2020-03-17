#ifndef __PYH2PACK_KERNELS_H__
#define __PYH2PACK_KERNELS_H__

#include <math.h>
#include <string.h>
#include "H2Pack_kernels.h"

#include "H2Pack_config.h"
#include "x86_intrin_wrapper.h" 


struct H2P_kernel
{
    char *krnl_name;            //  name of the kernel function. 
    int krnl_dim;               //  dimension of the kernel function's output (assume to be square presently)
    int pts_dim;                //  dimension of the input points. 
    int krnl_param_len;         //  number of kernel function parameters
    DTYPE *krnl_param;          //  kernel function parameter
    kernel_eval_fptr krnl_eval; //  function pointer for kernel_evaluation
    kernel_bimv_fptr krnl_bimv; //  function pointer for kernel_bimv
    int krnl_bimv_flops;        //  flops of applying krnl_bimv once. 
    int flag_proxysurface;      //  flag for whether to use proxy surface method. 
    char *param_descr;          //  description of the kernel function.
};
typedef struct H2P_kernel H2P_kernel;


const H2P_kernel kernel_list[] = 
{
    // 3D Gaussian kernel
    {
        .krnl_name = "Gaussian",
        .krnl_dim = 1,
        .pts_dim = 3,
        .krnl_param_len = 2,
        .krnl_param = (double []){1.0, 1.0},
        .krnl_eval = Gaussian_3d_eval_intrin_d, 
        .krnl_bimv = Gaussian_3d_krnl_bimv_intrin_d,
        .krnl_bimv_flops = Gaussian_3d_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "two parameters: (exponent, scaling factor)"
    }, 
    // 3D Matern kernel
    {
        .krnl_name = "Matern",
        .krnl_dim = 1,
        .pts_dim = 3,
        .krnl_param_len = 3,
        .krnl_param = (double []){1.5, 1, 1},  // (nu, sigma, rho)
        .krnl_eval = Matern_3d_eval_intrin_d, 
        .krnl_bimv = Matern_3d_krnl_bimv_intrin_d,
        .krnl_bimv_flops = Matern_3d_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "three parameters: (nu, sigma, rho)"
    },
    // 3D Coulomb kernel
    {
        .krnl_name = "Coulomb",
        .krnl_dim = 1,
        .pts_dim = 3,
        .krnl_param_len = 1,
        .krnl_param = (double []){1.0},  // (scaling factor)
        .krnl_eval = Coulomb_3d_eval_intrin_d, 
        .krnl_bimv = Coulomb_3d_krnl_bimv_intrin_d,
        .krnl_bimv_flops = Coulomb_3d_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "one parameters: (scaling factor)"
    },
    // 3D Stokes kernel
    {
        .krnl_name = "Stokes",
        .krnl_dim = 3,
        .pts_dim = 3,
        .krnl_param_len = 2,
        .krnl_param = (double []){0.5, 0.8},  // (scaling factor)
        .krnl_eval = Stokes_eval_std, 
        .krnl_bimv = Stokes_krnl_bimv_intrin_d,
        .krnl_bimv_flops = Stokes_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "two parameters: (eta,  a)"
    },
    // 3D RPY
    {
        .krnl_name = "RPY",
        .krnl_dim = 3,
        .pts_dim = 3,
        .krnl_param_len = 2,
        .krnl_param = (double []){1.0, 1.0},  // (scaling factor)
        .krnl_eval = RPY_eval_std,
        .krnl_bimv = RPY_krnl_bimv_intrin_d,
        .krnl_bimv_flops = RPY_krnl_bimv_flop,
        .flag_proxysurface = 1,
        .param_descr = "two parameters: (eta,  a)"
    }
};

H2P_kernel identify_kernel_info(const char* krnl_name, int krnl_dim, int pts_dim)
{
    int list_len = sizeof(kernel_list) / sizeof(H2P_kernel);
    for (int i = 0; i < list_len; i++)
        if ( (!strcmp(krnl_name, kernel_list[i].krnl_name)) && pts_dim == kernel_list[i].pts_dim && krnl_dim == kernel_list[i].krnl_dim)
            return kernel_list[i];
    return (H2P_kernel) {.krnl_name = NULL};
}

#endif