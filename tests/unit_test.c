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
    info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, 4, 3, a, 3, tau);
    printf("%d\n", info);
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, 4, 2, 2, a, 3, tau);
    printf("%d\n", info);
    printf("%lf  %lf  %lf\n", a[0], a[1],  a[2]);
    printf("%lf  %lf  %lf\n", a[3], a[4],  a[5]);
    printf("%lf  %lf  %lf\n", a[6], a[7],  a[8]);
    printf("%lf  %lf  %lf\n", a[9], a[10], a[11]);
    return 0;
}
