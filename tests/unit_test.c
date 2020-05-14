#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "H2Pack.h"

int main(int argc, char **argv)
{
    double a[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    double tau[4];
    int info;
    info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, 3, 4, a, 4, tau);
    printf("%d\n", info);
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, 3, 2, 2, a, 4, tau);
    printf("%d\n", info);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++) printf("% 2.4lf ", a[4 * i + j]);
        printf("\n");
    }
    return 0;
}
