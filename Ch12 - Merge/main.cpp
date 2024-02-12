#include <cstdio>
#include <iostream>
#include "circular_merge.cuh"

int main() {
    float *a, *b;
    int n = 10000, m = 20000;
    a = (float*) malloc(sizeof(float) * n);
    b = (float*) malloc(sizeof(float) * m);

    a[0] = b[0] = 0;
    for (int i = 1; i < n; i++) {
        a[i] = a[i - 1] + rand() / (float) RAND_MAX;
    }
    for (int i = 1; i < m; i++) {
        b[i] = b[i - 1] + rand() / (float) RAND_MAX;
    }

    float* c = circular_merge(a, b, n, m);

    for (int i = 0; i < 100; i++) {
        std::cout << a[i] << ' ' << b[i] << ' ' << c[i] << std::endl;
    }

    return 0;
}