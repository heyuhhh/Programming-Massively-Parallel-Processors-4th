#include <iostream>
#include "sort.cuh"

void mergeSort(float* a, const int n) {
    float *a_d, *tmp;
    cudaMalloc((void**) &a_d, sizeof(float) * n);
    cudaMalloc((void**) &tmp, sizeof(float) * n);
    cudaMemcpy(a_d, a, sizeof(float) * n, cudaMemcpyHostToDevice);
    for (int stride = 1; stride < n; stride <<= 1) {
        int seg_num = (n + 2 * stride - 1) / (2 * stride);
        int blockDim = min(seg_num, SECTION_SIZE);
        int gridDim((seg_num + blockDim - 1) / blockDim);
        merge_sort<<<gridDim, blockDim>>>(a_d, tmp, n, stride);
    }
    cudaMemcpy(a, a_d, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    cudaFree(tmp);
}

void radixSort(int* a, const int N) {
    int *a_d;
    cudaMallocManaged((void**) &a_d, sizeof(int) * N);
    int grid_size = (N + SECTION_SIZE * COARSE_FACTOR - 1) / (SECTION_SIZE * COARSE_FACTOR);
    cudaMemcpy(a_d, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    
    int *b_d, *c_d, *d_d;
    cudaMalloc((void**) &b_d, sizeof(int) * grid_size * (1 << Radix_N));
    cudaMallocManaged((void**) &c_d, sizeof(int) * (1 << Radix_N));
    cudaMallocManaged((void**) &d_d, sizeof(int) * N);
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float total_time = 0;
    for (int tt = 0; tt < 10; tt++) {
        cudaEventRecord(start_event, 0);
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
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float time = 0;
        cudaEventElapsedTime(&time, start_event, stop_event);
        total_time += time;
    }
    
    printf("time is: %f\n", total_time / 1.0e3f / 10);

    cudaMemcpy(a, a_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(d_d);
}