#ifndef __VEC_WRAPPER_FUNC_H__
#define __VEC_WRAPPER_FUNC_H__

#if defined(__AVX__)
#define HAVE_NATIVE_INTRINSIC
#include "avx_intrin_wrapper.h"
#endif

#if !defined(HAVE_NATIVE_INTRINSIC)
#warning No supported native intrinsic wrapper header. vec_wrapper_func.h will use emulated implementations.
#include "nointrin_wrapper.h"
#endif

#endif