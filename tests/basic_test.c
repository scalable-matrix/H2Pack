#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2Pack.h"

int main(int argc, char **argv)
{
    DTYPE *x, *y;
    int n = 1048576;
    if (argc >= 2)
    {
        n = atoi(argv[1]);
        if (n < 0 || n > 1048576 * 1048) n = 1048576;
    }
    double st = H2P_get_wtime_sec();
    x = (double*) H2P_malloc_aligned(sizeof(DTYPE) * n);
    y = (double*) H2P_malloc_aligned(sizeof(DTYPE) * n);
    assert(x != NULL && y != NULL);
    for (int i = 0; i < n; i++)
    {
        x[i] = (DTYPE) (i % 114);
        y[i] = (DTYPE) (i % 514);
    }
    H2P_free_aligned(x);
    H2P_free_aligned(y);
    double et = H2P_get_wtime_sec();
    H2P_PRINTF("Test done, n = %d, used time = %.3lf (s)\n", n, et - st);
    return 0;
}