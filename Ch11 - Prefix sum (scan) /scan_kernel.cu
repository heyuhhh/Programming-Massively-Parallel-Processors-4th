#include "scan_kernel.cuh"
#include <cstdio>

__device__ int blockCounter;

__device__
void inclusive_scan_device_BrentKung(volatile float* a, const int N, const int tid) {
    int stride;
    for (stride = 1; stride < N; stride <<= 1) {
        int n = 2 * stride * (tid + 1) - 1;
        if (n < N) {
            a[n] += a[n - stride];
        }
        __syncthreads();
    }
    for (stride >>= 1; stride > 1; stride >>= 1) {
        int n = stride * (tid + 1) - 1;
        if (n + stride / 2 < N) {
            a[n + stride / 2] += a[n];
        }
        __syncthreads();
    }
}

__global__
void scan_SinglePass(float* a, float* flag, float* aux, const int N, int mode=0) {
    __shared__ float preSum[SECTION_SIZE * COARSE_FACTOR];
    __shared__ float lastValue[SECTION_SIZE];
    // dynamic block index
    __shared__ int bid;
    int tid = threadIdx.x;
    if (tid == 0) {
        blockCounter = 0;
        bid = atomicAdd(&blockCounter, 1);
    }
    __syncthreads();
    // read combined global memory
    int n = bid * SECTION_SIZE * COARSE_FACTOR + threadIdx.x;
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if (n + i * SECTION_SIZE >= mode && n + i * SECTION_SIZE < N) {
            preSum[threadIdx.x + i * SECTION_SIZE] = a[n + i * SECTION_SIZE - mode];
        } else {
            preSum[threadIdx.x + i * SECTION_SIZE] = 0;
        }
    }
    __syncthreads();
    // first phase: sequential scan for each thread
    for (int i = threadIdx.x * COARSE_FACTOR; i + 1 < (threadIdx.x + 1) * COARSE_FACTOR; i++) {
        preSum[i + 1] += preSum[i];
    }
    __syncthreads();
    // second phase: parallel scan for last element
    lastValue[threadIdx.x] = preSum[(threadIdx.x + 1) * COARSE_FACTOR - 1];
    __syncthreads();
    inclusive_scan_device_BrentKung(lastValue, SECTION_SIZE, threadIdx.x);
    // third phase: add each element on later block
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if ((threadIdx.x + 1) * COARSE_FACTOR + i < SECTION_SIZE * COARSE_FACTOR) {
            preSum[(threadIdx.x + 1) * COARSE_FACTOR + i] += lastValue[threadIdx.x];
        }
    }
    __syncthreads();
    
    // block sync
    __shared__ float value;
    if (tid == SECTION_SIZE - 1) {
        while (bid > 0 && atomicAdd(&flag[bid], 0) == 0);
        value = aux[bid];
        if (bid + 1 < blockDim.x) {
            aux[bid + 1] = value + lastValue[tid];
            __threadfence();
            atomicAdd(&flag[bid + 1], 1);
        }
    }
    __syncthreads();
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if ((threadIdx.x + 1) * COARSE_FACTOR + i < SECTION_SIZE * COARSE_FACTOR) {
            preSum[(threadIdx.x + 1) * COARSE_FACTOR + i] += value;
        }
    }
    __syncthreads();
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if (n + i * SECTION_SIZE < N) {
            a[n + i * SECTION_SIZE] = preSum[threadIdx.x + i * SECTION_SIZE];
        }
    }    
}

__global__
void scan_KoggleStone(float* a, const int N, int mode=0) {
    extern __shared__ float preSum[];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= mode && i < N) {
        preSum[i] = a[i - mode];
    } else {
        preSum[i] = 0;
    }
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        __syncthreads(); // 这里放在前面是因为上面也更新了preSum的
        float tmp;
        if (threadIdx.x >= stride) {
            tmp = preSum[threadIdx.x] + preSum[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            preSum[threadIdx.x] = tmp;
        }
    }
    if (i < N) {
        a[i] = preSum[threadIdx.x];
    }
}

__global__
void scan_BrentKung(float* a, const int N, int mode=0) {
    extern __shared__ float preSum[];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= mode && i < N) {
        preSum[i] = a[i - mode];
    } else {
        preSum[i] = 0;
    }
    __syncthreads();
    /*
        forward process:
            二进制位从低到高依次处理，最终 i 这个位置的值表示 (i-lowbit(i), i] 的和
            第j次迭代处理第j位，迭代完成后，后j位处理完毕，每个数能表示的范围为 2^j
    */
    int stride;
    for (stride = 1; stride < blockDim.x; stride <<= 1) {
        int n = 2 * stride * (threadIdx.x + 1) - 1;
        if (n < blockDim.x) {
            preSum[n] += preSum[n - stride];
        }
        __syncthreads();
    }
    /*
        reverse process:

        二进制位从高到低进行修正，维护前缀和，每个线程处理前面的xxxx位
        假设前面若干位已经处理好，那么把所有的结果 xxxx00000(已更新) 加上 xxxx10000(没更新)
        即可得到当前这一位包含前面所有位的前缀和结果

            01000000 blockDim
            0x100000 1st iter
            0xx10000 2nd iter
            0xxx1000 3rd iter
            0xxxx100 4th iter
            ...
    */
    for (stride >>= 1; stride > 1; stride >>= 1) {
        int n = stride * (threadIdx.x + 1) - 1;
        if (n + stride / 2 < blockDim.x) {
            preSum[n + stride / 2] += preSum[n];
        }
        __syncthreads();
    }
    if (i < N) {
        a[i] = preSum[threadIdx.x];
    }   
}

__global__
void addLastElement(float* a, float* b, const int N) {
    float lastValue = 0;
    if (blockIdx.x > 0) {
        lastValue = b[blockIdx.x - 1];
    }
    int n = blockIdx.x * SECTION_SIZE * COARSE_FACTOR + threadIdx.x;
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if (n + i * SECTION_SIZE < N) {
            a[n + i * SECTION_SIZE] += lastValue;
        }
    }
}

__global__
void scan_on_block(float* a, float* b, const int N, int mode) {
    __shared__ float preSum[SECTION_SIZE * COARSE_FACTOR];
    __shared__ float lastValue[SECTION_SIZE];
    // read combined global memory
    int n = blockIdx.x * SECTION_SIZE * COARSE_FACTOR + threadIdx.x;
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if (n + i * SECTION_SIZE >= mode && n + i * SECTION_SIZE < N) {
            preSum[threadIdx.x + i * SECTION_SIZE] = a[n + i * SECTION_SIZE - mode];
        } else {
            preSum[threadIdx.x + i * SECTION_SIZE] = 0;
        }
    }
    __syncthreads();
    // first phase: sequential scan for each thread
    for (int i = threadIdx.x * COARSE_FACTOR; i + 1 < (threadIdx.x + 1) * COARSE_FACTOR; i++) {
        preSum[i + 1] += preSum[i];
    }
    __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     for (int i = 0; i < SECTION_SIZE * COARSE_FACTOR; i++) {
    //         printf("%f\n", preSum[i]);
    //     }
    // }
    // second phase: parallel scan for last element
    lastValue[threadIdx.x] = preSum[(threadIdx.x + 1) * COARSE_FACTOR - 1];
    __syncthreads();
    inclusive_scan_device_BrentKung(lastValue, SECTION_SIZE, threadIdx.x);
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     for (int i = 0; i < SECTION_SIZE; i++) {
    //         printf("%f\n", lastValue[i]);
    //     }
    // }
    // third phase: add each element on later block
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if ((threadIdx.x + 1) * COARSE_FACTOR + i < SECTION_SIZE * COARSE_FACTOR) {
            preSum[(threadIdx.x + 1) * COARSE_FACTOR + i] += lastValue[threadIdx.x];
        }
    }
    __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     for (int i = 0; i < SECTION_SIZE * COARSE_FACTOR; i++) {
    //         printf("%f\n", preSum[i]);
    //     }
    // }
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if (n + i * SECTION_SIZE < N) {
            a[n + i * SECTION_SIZE] = preSum[threadIdx.x + i * SECTION_SIZE];
        }
    }
    if (threadIdx.x == 0) {
        b[blockIdx.x] = preSum[SECTION_SIZE * COARSE_FACTOR - 1];
    }    
}