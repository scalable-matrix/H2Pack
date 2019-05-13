#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include "H2Pack_config.h"

// Get wall-clock time, similar to omp_get_wtime()
double H2P_get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

// Allocate an aligned memory block
void *H2P_malloc_aligned(size_t mem_size)
{
    void *ptr;
    ptr = _mm_malloc(mem_size, ALIGN_SIZE);
    return ptr;
}

// Free a memory block allocated using H2P_malloc_aligned()
void H2P_free_aligned(void *ptr)
{
    _mm_free(ptr);
}

