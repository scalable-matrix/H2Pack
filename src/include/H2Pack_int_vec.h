#ifndef __H2PACK_INT_VEC_H__
#define __H2PACK_INT_VEC_H__

#include "H2Pack_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Integer vector, similar to std::vector in C++
struct H2P_int_vector
{
    int capacity;    // Capacity of this vector
    int length;      // Current length of this vector
    int *data;       // Data in this vector
};
typedef struct H2P_int_vector* H2P_int_vector_t;

// Initialize a H2P_int_vector structure
// Input parameter:
//   capacity : Initial capacity of the vector. If (capacity <= 0 || capacity >= 65536),
//              capacity will be set as 128.
// Output parameter:
//   int_vec_ : Initialized H2P_int_vector structure
void H2P_int_vector_init(H2P_int_vector_t *int_vec_, int capacity);

// Destroy a H2P_int_vector structure
// Input parameter:
//   int_vec : H2P_int_vector structure
void H2P_int_vector_destroy(H2P_int_vector_t int_vec);

// Push an integer to the tail of a H2P_int_vector
// Input parameters:
//   int_vec : H2P_int_vector structure
//   value   : Value to be pushed 
// Output parameter:
//   int_vec : H2P_int_vector structure with the pushed value
void H2P_int_vector_push_back(H2P_int_vector_t int_vec, int value);

#ifdef __cplusplus
}
#endif

#endif
