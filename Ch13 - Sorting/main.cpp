#include <cstdio>
#include <iostream>
#include<ctime>
#include <algorithm>
#include "sort.cuh"
#define N 10000000
 
int main() {
    int *a = (int*) malloc(sizeof(int) * N);
    int *b = (int*) malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) {
        a[i] = int(rand() / (float) RAND_MAX * 10);
        b[i] = int(rand() / (float) RAND_MAX * 5);
    }

    float esp_time_cpu;
	clock_t start_cpu, stop_cpu;
    radixSort(b, N);

    start_cpu = clock();
    radixSort(a, N);
    stop_cpu = clock();// end timing

	esp_time_cpu = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;

	printf("The time by host:\t%f(ms)\n", esp_time_cpu);

    // mergeSort(a, N);

    // for (int i = 0; i < 100; i++) {
    //     std::cout << a[i] << ' ';
    // }
    // std::cout << std::endl << std::endl;
    // std::sort(a, a + N);
    // for (int i = 0; i < 100; i++) {
    //     std::cout << a[i] << ' ';
    // }
    // std::cout << std::endl;
    return 0;
}