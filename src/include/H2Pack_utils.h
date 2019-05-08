#ifndef __H2PACK_UTILS_H__
#define __H2PACK_UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(a, b)  ((a) < (b) ? (a) : (b))
#define MAX(a, b)  ((a) > (b) ? (a) : (b))

#define H2P_PRINTF(fmt, args...)                       \
        do                                             \
        {                                              \
            fprintf(stdout, "%s(), line %d: "fmt,      \
                    __FUNCTION__, __LINE__, ##args);   \
            fflush(stdout);                            \
        } while (0)

// Get wall-clock time, similar to omp_get_wtime()
double H2P_get_wtime_sec();

// Allocate an aligned memory block
void *H2P_malloc_aligned(size_t mem_size);

// Free a memory block allocated using H2P_malloc_aligned()
void H2P_free_aligned(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
