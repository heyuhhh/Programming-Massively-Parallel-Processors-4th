#include <iostream>
#include "radix_sort.cuh"

void radixSort(int* a, const int N) {
    int *a_d;
    cudaMallocManaged((void**) &a_d, sizeof(int) * N);
    int grid_size = (N + SECTION_SIZE * COARSE_FACTOR - 1) / (SECTION_SIZE * COARSE_FACTOR);
    cudaMemcpy(a_d, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    
    int *b_d, *c_d, *d_d;
    cudaMalloc((void**) &b_d, sizeof(int) * grid_size * (1 << Radix_N));
    cudaMallocManaged((void**) &c_d, sizeof(int) * (1 << Radix_N));
    cudaMallocManaged((void**) &d_d, sizeof(int) * N);
    for (int i = 0; i < 32 / Radix_N; i++) {
        // local radix sort
        dim3 grid(grid_size);
        dim3 block(SECTION_SIZE);
        cudaMemset(b_d, 0, sizeof(int) * (1 << Radix_N) * grid_size);
        cudaMemset(c_d, 0, sizeof(int) * (1 << Radix_N));
        local_radix_sort<<<grid, block>>>(a_d, b_d, c_d, N, i);
        exclusive_scan(c_d, c_d, 1 << Radix_N);

        for (int j = 0; j < 1 << Radix_N; j++) {
            exclusive_scan(b_d + j * grid_size, b_d + j * grid_size, grid_size);
        }
        
        global_radix_sort<<<grid_size, block>>>(a_d, b_d, c_d, d_d, N, i);
        cudaMemcpy(a_d, d_d, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(a, a_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
}