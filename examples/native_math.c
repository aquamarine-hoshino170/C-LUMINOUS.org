// C Shared Library Source
double fast_power(double base, double exp) {
    double res = 1.0;
    for (int i = 0; i < (int)exp; i++) {
        res *= base;
    }
    return res;
}

double fast_add(double a, double b) {
    return a + b;
}
