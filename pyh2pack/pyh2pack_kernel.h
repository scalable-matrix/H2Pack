#ifndef __PYH2PACK_KERNELS_H__
#define __PYH2PACK_KERNELS_H__

#include <math.h>
#include <string.h>
#include "H2Pack_kernels.h"
#include "H2Pack_config.h"
#include "ASTER/include/aster.h"


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
        .krnl_name = "Gaussian_3D",
        .krnl_dim = 1,
        .pts_dim = 3,
        .krnl_param_len = 1,
        .krnl_param = (DTYPE []){1.0},
        .krnl_eval = Gaussian_3D_eval_intrin_t, 
        .krnl_bimv = Gaussian_3D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Gaussian_3D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "one parameter l: K(x,y) = exp(-l|x-y|^2) "
    }, 
    // 2D Gaussian kernel
    {
        .krnl_name = "Gaussian_2D",
        .krnl_dim = 1,
        .pts_dim = 2,
        .krnl_param_len = 1,
        .krnl_param = (DTYPE []){1.0},
        .krnl_eval = Gaussian_2D_eval_intrin_t, 
        .krnl_bimv = Gaussian_2D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Gaussian_2D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "one parameter l: K(x,y) = exp(-l|x-y|^2) "
    }, 
    // 3D Exponential kernel
    {
        .krnl_name = "Exponential_3D",
        .krnl_dim = 1,
        .pts_dim = 3,
        .krnl_param_len = 1,
        .krnl_param = (DTYPE []){1.0},
        .krnl_eval = Expon_3D_eval_intrin_t, 
        .krnl_bimv = Expon_3D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Expon_3D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "one parameter l: K(x,y) = exp(-l|x-y|) "
    },
    // 2D Exponential kernel
    {
        .krnl_name = "Exponential_2D",
        .krnl_dim = 1,
        .pts_dim = 2,
        .krnl_param_len = 1,
        .krnl_param = (DTYPE []){1.0},
        .krnl_eval = Expon_2D_eval_intrin_t, 
        .krnl_bimv = Expon_2D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Expon_2D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "one parameter l: K(x,y) = exp(-l|x-y|) "
    },
    // 3D Matern 3/2 kernel
    {
        .krnl_name = "Matern32_3D",
        .krnl_dim = 1,
        .pts_dim = 3,
        .krnl_param_len = 1,
        .krnl_param = (DTYPE[]){1.0},
        .krnl_eval = Matern32_3D_eval_intrin_t, 
        .krnl_bimv = Matern32_3D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Matern32_3D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "one parameter l: K(x,y) = (1+sqrt(3)*l) exp(-sqrt(3)*l|x-y|) "
    },
    // 2D Matern 3/2 kernel
    {
        .krnl_name = "Matern32_2D",
        .krnl_dim = 1,
        .pts_dim = 2,
        .krnl_param_len = 1,
        .krnl_param = (DTYPE[]){1.0},
        .krnl_eval = Matern32_2D_eval_intrin_t, 
        .krnl_bimv = Matern32_2D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Matern32_2D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "one parameter l: K(x,y) = (1+sqrt(3)*l) exp(-sqrt(3)*l|x-y|) "
    },
    // 3D Matern 5/2 kernel
    {
        .krnl_name = "Matern52_3D",
        .krnl_dim = 1,
        .pts_dim = 3,
        .krnl_param_len = 1,
        .krnl_param = (DTYPE[]){1.0},  // (nu, sigma, rho)
        .krnl_eval = Matern52_3D_eval_intrin_t, 
        .krnl_bimv = Matern52_3D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Matern52_3D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "one parameter l: K(x,y) = (1+sqrt(5)*l+5/3*l) exp(-sqrt(5)*l|x-y|) "
    },
    // 2D Matern 5/2 kernel
    {
        .krnl_name = "Matern52_2D",
        .krnl_dim = 1,
        .pts_dim = 2,
        .krnl_param_len = 1,
        .krnl_param = (DTYPE[]){1.0},  // (nu, sigma, rho)
        .krnl_eval = Matern52_2D_eval_intrin_t, 
        .krnl_bimv = Matern52_2D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Matern52_2D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "one parameter l: K(x,y) = (1+sqrt(5)*l+5/3*l) exp(-sqrt(5)*l|x-y|) "
    },
    // 3D Quadratic kernel
    {
        .krnl_name = "Quadratic_3D",
        .krnl_dim = 1,
        .pts_dim = 3,
        .krnl_param_len = 2,
        .krnl_param = (DTYPE[]){1.0, -0.5},
        .krnl_eval = Quadratic_3D_eval_intrin_t, 
        .krnl_bimv = Quadratic_3D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Quadratic_3D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "two parameters c,a: K(x,y) = (1+c*|x-y|^2)^a"
    },
    // 2D Quadratic kernel
    {
        .krnl_name = "Quadratic_2D",
        .krnl_dim = 1,
        .pts_dim = 2,
        .krnl_param_len = 2,
        .krnl_param = (DTYPE[]){1.0, -0.5},
        .krnl_eval = Quadratic_2D_eval_intrin_t, 
        .krnl_bimv = Quadratic_2D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Quadratic_2D_krnl_bimv_flop,
        .flag_proxysurface = 0,
        .param_descr = "two parameters c,a: K(x,y) = (1+c*|x-y|^2)^a"
    },
    // 3D Coulomb kernel
    {
        .krnl_name = "Coulomb_3D",
        .krnl_dim = 1,
        .pts_dim = 3,
        .krnl_param_len = 0,
        .krnl_param = NULL, 
        .krnl_eval = Coulomb_3D_eval_intrin_t, 
        .krnl_bimv = Coulomb_3D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Coulomb_3D_krnl_bimv_flop,
        .flag_proxysurface = 1,
        .param_descr = "no parameters"
    },
    // 2D Laplace kernel
    {
        .krnl_name = "Laplace_2D",
        .krnl_dim = 1,
        .pts_dim = 2,
        .krnl_param_len = 0,
        .krnl_param = NULL, 
        .krnl_eval = Laplace_2D_eval_intrin_t, 
        .krnl_bimv = Laplace_2D_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Laplace_2D_krnl_bimv_flop,
        .flag_proxysurface = 1,
        .param_descr = "no parameters"
    },
    // 3D Stokes kernel
    {
        .krnl_name = "Stokes_3D",
        .krnl_dim = 3,
        .pts_dim = 3,
        .krnl_param_len = 2,
        .krnl_param = (DTYPE []){1.0, 0.1}, 
        .krnl_eval = Stokes_eval_std, 
        .krnl_bimv = Stokes_krnl_bimv_intrin_t,
        .krnl_bimv_flops = Stokes_krnl_bimv_flop,
        .flag_proxysurface = 1,
        .param_descr = "two parameters: (eta,  a)"
    },
    // 3D RPY
    {
        .krnl_name = "RPY",
        .krnl_dim = 3,
        .pts_dim = 3,
        .krnl_param_len = 1,
        .krnl_param = (DTYPE []){1.0},  
        .krnl_eval = RPY_eval_std,
        .krnl_bimv = RPY_krnl_bimv_intrin_t,
        .krnl_bimv_flops = RPY_krnl_bimv_flop,
        .flag_proxysurface = 1,
        .param_descr = "two parameters: eta"
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
