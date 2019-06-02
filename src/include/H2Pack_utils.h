#ifndef __H2PACK_UTILS_H__
#define __H2PACK_UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

// Helper functions used in H2Pack

#define MIN(a, b)  ((a) < (b) ? (a) : (b))
#define MAX(a, b)  ((a) > (b) ? (a) : (b))

#define H2P_PRINTF(fmt, args...)                       \
        do                                             \
        {                                              \
            fprintf(stdout, "%s, line %d: "fmt,        \
                    __FILE__, __LINE__, ##args);       \
            fflush(stdout);                            \
        } while (0)

// Get wall-clock time, similar to omp_get_wtime()
double H2P_get_wtime_sec();

// Allocate an aligned memory block
// Input parameter:
//   mem_size : Size of the memory block (bytes) 
void *H2P_malloc_aligned(size_t mem_size);

// Free a memory block allocated using H2P_malloc_aligned()
// Input parameter:
//   ptr : Pointer to the memory block
void H2P_free_aligned(void *ptr);

// Block partition of a set
// Input parameters:
//   n_elem : Number of elements to be partitioned 
//   n_blk  : Number of blocks
//   i_blk  : Index of the target block 
// Output parameters:
//   i_blk_spos : Index of the first element belongs to the i_blk-th block
//   i_blk_size : Size of the i_blk-th block
static inline void H2P_block_partition(
    const int n_elem, const int n_blk, const int i_blk, 
    int *i_blk_spos, int *i_blk_size
)
{
    int rem = n_elem % n_blk;
    int bs0 = n_elem / n_blk;
    int bs1 = bs0 + 1;
    if (i_blk < rem)
    {
        *i_blk_spos = bs1 * i_blk;
        *i_blk_size = bs1;
    } else {
        *i_blk_spos = bs0 * i_blk + rem;
        *i_blk_size = bs0;
    }
}

#ifdef __cplusplus
}
#endif

#endif
