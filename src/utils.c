#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include "config.h"
#include "utils.h"

// Get current wall-clock time, similar to omp_get_wtime()
double H2P_get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

// Allocate an memory block, aligned to ALIGN_SIZE if possible
void *H2P_malloc(size_t mem_size)
{
    void *ptr;
    #ifdef __INTEL_COMPILER
    ptr = _mm_malloc(mem_size, ALIGN_SIZE);
    #else
    ptr = malloc(mem_size);
    #endif
    return ptr;
}

// Free a memory block
void H2P_free(void *ptr)
{
    #ifdef __INTEL_COMPILER
    _mm_free(ptr);
    #else
    free(ptr);
    #endif
}
