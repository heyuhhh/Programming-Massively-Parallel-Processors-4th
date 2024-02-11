#include <cstdio>
#include <iostream>
#include "scan.cuh"
#define N 100000000

int main() {
    float *a = (float*) malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
    }

    float *b = (float*) malloc(sizeof(float) * N);
    exclusive_scan(b, a, N);

    float* c = (float*) malloc(sizeof(float) * N);
    c[0] = a[0];
    for (int i = 1; i < N; i++) {
        c[i] = a[i] + c[i - 1];
    }

    for (int i = 1000; i < 1100; i++) {
        std::cout << b[i] << ' ';
    }
    std::cout << std::endl;
    for (int i = 1000; i < 1100; i++) {
        std::cout << c[i] << ' ';
    }
    return 0;
}