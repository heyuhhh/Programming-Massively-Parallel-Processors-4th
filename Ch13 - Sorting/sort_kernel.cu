#include "sort_kernel.cuh"
#include <cstdio>
#define SECTION_SIZE 256
#define COARSE_FACTOR 4

__global__
void local_radix_sort(int* a, int* b, int* c, const int N, int iter) {
    __shared__ int a_s[SECTION_SIZE * COARSE_FACTOR];
    __shared__ int b_s[1 << Radix_N];
    __shared__ int a_mask[SECTION_SIZE * COARSE_FACTOR];
    __shared__ int tmp[SECTION_SIZE * COARSE_FACTOR + 1];
    if (threadIdx.x < 1 << Radix_N) {
        b_s[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        tmp[SECTION_SIZE * COARSE_FACTOR] = 0;
    }
    __syncthreads();
    int n = blockDim.x * blockIdx.x * COARSE_FACTOR + threadIdx.x;
    // global->shared memory
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if (n + i * blockDim.x < N) {
            a_s[threadIdx.x + i * blockDim.x] = a[n + i * blockDim.x];
        } else {
            a_s[threadIdx.x + i * blockDim.x] = MAX;
        }
    }
    __syncthreads();
    // local radix sort
    for (int i = 0; i < Radix_N; i++) {
        int bit = iter * Radix_N + i;
        for (int j = 0; j < COARSE_FACTOR; j++) {
            // avoid bank conflict
            a_mask[threadIdx.x + j * blockDim.x] = (a_s[threadIdx.x + j * blockDim.x] >> bit & 1);
            if (a_mask[threadIdx.x + j * blockDim.x]) {
                atomicAdd(&tmp[SECTION_SIZE * COARSE_FACTOR], 1);
            }
        }
        __syncthreads();
        // 0: index - # 1 before
        // 1: len - #1 + # 1 before
        // find "# 1 before" by exclusive scan

        exclusive_scan_device_BrentKung(a_mask, SECTION_SIZE * COARSE_FACTOR, threadIdx.x);
        // a_mask[i]: # 1 before i
        for (int j = 0; j < COARSE_FACTOR; j++) {
            int cur = threadIdx.x + j * blockDim.x;
            int nxt;
            if (a_s[cur] >> bit & 1) {
                nxt = SECTION_SIZE * COARSE_FACTOR - tmp[SECTION_SIZE * COARSE_FACTOR] + a_mask[cur];
            } else {
                nxt = cur - a_mask[cur];
            }
            tmp[nxt] = a_s[cur];
        }
        __syncthreads();
        for (int j = 0; j < COARSE_FACTOR; j++) {
            a_s[threadIdx.x + j * blockDim.x] = tmp[threadIdx.x + j * blockDim.x];
        }
        if (threadIdx.x == 0) {
            tmp[SECTION_SIZE * COARSE_FACTOR] = 0;
        }
        __syncthreads();
    }
    int bit = iter * Radix_N;
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if (n + i * blockDim.x < N) {
            atomicAdd(&b_s[a_s[threadIdx.x + i * blockDim.x] >> bit & Valid], 1);
            a[n + i * blockDim.x] = a_s[threadIdx.x + i * blockDim.x];
        }
        __syncthreads();
    }
    if (threadIdx.x <= Valid) {
        // [threadIdx.x] [blockIdx.x]
        b[threadIdx.x * gridDim.x + blockIdx.x] = b_s[threadIdx.x];
        atomicAdd(&c[threadIdx.x], b_s[threadIdx.x]);
    }
}

__global__
void global_radix_sort(int *a, int *b, int *c, int *d, const int N, int iter) {
    __shared__ int a_s[SECTION_SIZE * COARSE_FACTOR];
    __shared__ int bound_l[1 << Radix_N];
    if (threadIdx.x < 1 << Radix_N) {
        bound_l[threadIdx.x] = 0;
    }
    __syncthreads();
    int n = blockDim.x * blockIdx.x * COARSE_FACTOR + threadIdx.x;
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if (n + i * blockDim.x < N) {
            a_s[threadIdx.x + i * blockDim.x] = a[n + i * blockDim.x];
        }
    }
    __syncthreads();
    
    int bit = iter * Radix_N;
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int cur = threadIdx.x + i * blockDim.x;
        if (n + i * blockDim.x < N) {
            int val = (a_s[cur] >> bit & Valid);
            if (cur == 0 || val != (a_s[cur - 1] >> bit & Valid)) {
                bound_l[val] = cur;
            }
        }
    }
    __syncthreads();
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int cur = threadIdx.x + i * blockDim.x;
        if (n + i * blockDim.x < N) {
            int val = (a_s[cur] >> bit & Valid);
            int nxt = c[val] + b[val * gridDim.x + blockIdx.x] + (cur - bound_l[val]);
            d[nxt] = a_s[cur];
        }
    }
}