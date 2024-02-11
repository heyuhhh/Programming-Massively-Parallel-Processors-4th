#include <cstdio>
#include <iostream>
#include "radix_sort.cuh"
#define N 10000000

int main() {
    int *a = (int*) malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) {
        a[i] = N - i;
    }

    radixSort(a, N);

    for (int i = 1000; i < 1100; i++) {
        std::cout << a[i] << ' ';
    }
    std::cout << std::endl;
    return 0;
}