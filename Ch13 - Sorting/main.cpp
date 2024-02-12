#include <cstdio>
#include <iostream>
#include <algorithm>
#include "sort.cuh"
#define N 10000000
 
int main() {
    float *a = (float*) malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float) RAND_MAX * 10;
    }
    // radixSort(a, N);
    mergeSort(a, N);

    for (int i = 0; i < 100; i++) {
        std::cout << a[i] << ' ';
    }
    std::cout << std::endl << std::endl;
    std::sort(a, a + N);
    for (int i = 0; i < 100; i++) {
        std::cout << a[i] << ' ';
    }
    std::cout << std::endl;
    return 0;
}