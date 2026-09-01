#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double geodesic_acceleration(double M, double r) {
    double f = 1.0 - (2.0 * M / r);
    return (M / (r * r)) * f;
}

int main() {
    printf("\033[1;32m[Success]\033[0m Executed Pure Self-Compiled C Luminous Binary!\n");
    double M = 1.0;
    double r = 6.0;
    double acc = geodesic_acceleration(M, r);
    printf("[Schwarzschild Geodesic Invariant (r=6M)]: %f\n", acc);
    return 0;
}
