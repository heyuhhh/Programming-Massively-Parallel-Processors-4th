#include "scan.cuh"
#include <cstdio>
#include <iostream>

void scan(float* out, const int N, int mode) {

    float *block_value;
    int blocks = (N + SECTION_SIZE * COARSE_FACTOR - 1) / (SECTION_SIZE * COARSE_FACTOR);

    cudaMalloc((void**) &block_value, sizeof(float) * blocks);
    
    // step 1
    scan_on_block<<<blocks, SECTION_SIZE>>>(out, block_value, N, mode);
    cudaDeviceSynchronize();
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    // step 2
    if (blocks > 2 * SECTION_SIZE) {
        scan(block_value, blocks, 0);
    } else {
        scan_BrentKung<<<1, blocks, sizeof(float) * blocks>>>(block_value, blocks, 0);
    }

    // step 3
    addLastElement<<<blocks, SECTION_SIZE>>>(out, block_value, N);

    cudaFree(block_value);
}

void inclusive_scan(float* out, float* in, const int N) {
    float *out_d;
    cudaMalloc((void**) &out_d, sizeof(float) * N);
    cudaMemcpy(out_d, in, sizeof(float) * N, cudaMemcpyHostToDevice);
    scan(out_d, N, 0);
    cudaMemcpy(out, out_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
}

void exclusive_scan(float* out, float* in, const int N) {
    float *out_d;
    cudaMalloc((void**) &out_d, sizeof(float) * N);
    cudaMemcpy(out_d, in, sizeof(float) * N, cudaMemcpyHostToDevice);
    scan(out_d, N, 1);
    cudaMemcpy(out, out_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
}


void scan(int* out, const int N, int mode) {

    int *block_value;
    int blocks = (N + SECTION_SIZE * COARSE_FACTOR - 1) / (SECTION_SIZE * COARSE_FACTOR);

    cudaMalloc((void**) &block_value, sizeof(int) * blocks);
    
    // step 1
    scan_on_block<<<blocks, SECTION_SIZE>>>(out, block_value, N, mode);
    cudaDeviceSynchronize();
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    // step 2
    if (blocks > 2 * SECTION_SIZE) {
        scan(block_value, blocks, 0);
    } else {
        scan_BrentKung<<<1, blocks, sizeof(int) * blocks>>>(block_value, blocks, 0);
    }

    // step 3
    addLastElement<<<blocks, SECTION_SIZE>>>(out, block_value, N);

    cudaFree(block_value);
}

void inclusive_scan(int* out, int* in, const int N) {
    int *out_d;
    cudaMalloc((void**) &out_d, sizeof(int) * N);
    cudaMemcpy(out_d, in, sizeof(int) * N, cudaMemcpyHostToDevice);
    scan(out_d, N, 0);
    cudaMemcpy(out, out_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
}

void exclusive_scan(int* out, int* in, const int N) {
    int *out_d;
    cudaMalloc((void**) &out_d, sizeof(int) * N);
    cudaMemcpy(out_d, in, sizeof(int) * N, cudaMemcpyHostToDevice);
    scan(out_d, N, 1);
    cudaMemcpy(out, out_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
}
