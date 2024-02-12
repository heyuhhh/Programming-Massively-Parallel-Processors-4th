#include <cstdio>
#include "merge_kernel.cuh"

float* circular_merge(float *a, float *b, int n, int m) {
    float* c = (float*) malloc(sizeof(float) * (n + m));
    float *a_d, *b_d, *c_d;
    cudaMalloc((void**) &a_d, sizeof(float) * n);
    cudaMalloc((void**) &b_d, sizeof(float) * m);
    cudaMalloc((void**) &c_d, sizeof(float) * (n + m));
    cudaMemcpy(a_d, a, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(float) * m, cudaMemcpyHostToDevice);

    dim3 gridDim((n + m + TILE_SIZE * COARSE_FACTOR - 1) / (TILE_SIZE * COARSE_FACTOR));
    dim3 blockDim(TILE_SIZE);

    circularMergeSort<<<gridDim, blockDim>>>(a_d, b_d, c_d, n, m, n + m);
    cudaDeviceSynchronize();
    cudaMemcpy(c, c_d, sizeof(float) * (n + m), cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return c;
}