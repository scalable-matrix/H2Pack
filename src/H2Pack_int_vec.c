#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack_config.h"
#include "H2Pack_int_vec.h"

// Initialize a H2P_int_vector structure
void H2P_int_vector_init(H2P_int_vector_t *int_vec_, int capacity)
{
    if (capacity < 0 || capacity > 65536) capacity = 128;
    H2P_int_vector_t int_vec = (H2P_int_vector_t) malloc(sizeof(struct H2P_int_vector));
    assert(int_vec != NULL);
    int_vec->capacity = capacity;
    int_vec->length = 0;
    int_vec->data = (int*) malloc(sizeof(int) * capacity);
    assert(int_vec->data != NULL);
    *int_vec_ = int_vec;
}

// Destroy a H2P_int_vector structure
void H2P_int_vector_destroy(H2P_int_vector_t int_vec)
{
    free(int_vec->data);
    int_vec->capacity = 0;
    int_vec->length = 0;
}

// Push an integer to the tail of a H2P_int_vector
void H2P_int_vector_push_back(H2P_int_vector_t int_vec, int value)
{
    if (int_vec->capacity == int_vec->length)
    {
        int_vec->capacity *= 2;
        int *new_data = (int*) malloc(sizeof(int) * int_vec->capacity);
        assert(new_data != NULL);
        memcpy(new_data, int_vec->data, sizeof(int) * int_vec->length);
        free(int_vec->data);
        int_vec->data = new_data;
    }
    int_vec->data[int_vec->length] = value;
    int_vec->length++;
}
