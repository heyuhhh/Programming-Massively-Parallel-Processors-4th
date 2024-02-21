#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define N 10000000
#define THREAD_NUM 128

__global__
void reduce_kernel(float* a, float* b, const int MAX) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float a_sm[]; // 在配置中声明，注意是字节数
    a_sm[threadIdx.x] = 0;
    if (n < MAX) {
        // thread coarsening
        // 这里换了一种写法，没有COARSE FACTOR参与，因为这样比较方便控制 block 的数量
        // 第二次reduction会很方便
        float value = 0.0;
        for (int i = n; i < MAX; i += blockDim.x * gridDim.x) {
            value += a[i];
        }
        a_sm[threadIdx.x] = value;
        __syncthreads();
        // start reduction
        // offset从大到小枚举，利于合并访存（共享内存能减少bank冲突，相邻线程访问连续的地址）
        // 控制分支也能减少，活跃线程束会更加紧凑
        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                a_sm[threadIdx.x] += a_sm[threadIdx.x + offset];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            b[blockIdx.x] = a_sm[0];
        }
    }
}

// 使用volatile关键字，使得a_sm的访问不会被优化
// 由于是一个线程束，所以不存在顺序问题
__device__
void warpReduce(volatile float* a_sm, int tid) { 
    a_sm[tid] += a_sm[tid + 32];
    a_sm[tid] += a_sm[tid + 16];
    a_sm[tid] += a_sm[tid + 8];
    a_sm[tid] += a_sm[tid + 4];
    a_sm[tid] += a_sm[tid + 2];
    a_sm[tid] += a_sm[tid + 1];
}

__global__
void reduce_kernel_v2(float* a, float* b, const int MAX) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float a_sm[];
    a_sm[threadIdx.x] = 0;
    if (n < MAX) {
        // thread coarsening
        float value = 0.0;
        for (int i = n; i < MAX; i += blockDim.x * gridDim.x) {
            value += a[i];
        }
        a_sm[threadIdx.x] = value;
        __syncthreads();
        // start reduction
        for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
            if (threadIdx.x < offset) {
                a_sm[threadIdx.x] += a_sm[threadIdx.x + offset];
            }
        }
        __syncthreads();
        // 进一步优化，当活跃线程 <= 32 时，考虑协作组或者直接循环展开
        if (threadIdx.x < 32) {
            warpReduce(a_sm, threadIdx.x);
        }

        if (threadIdx.x == 0) {
            b[blockIdx.x] = a_sm[0];
        }
    }
}


int main() {
    float* a;
    float ans = 0;
    cudaMallocManaged((void**) &a, sizeof(float) * N);
    cudaMemPrefetchAsync(a, sizeof(float) * N, cudaCpuDeviceId);
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float)RAND_MAX;
        ans += a[i];
    }
    int blockCount = 1024;
    float* b, *c;
    cudaMallocManaged((void**) &b, sizeof(float) * blockCount);
    cudaMallocManaged((void**) &c, sizeof(float));
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaMemPrefetchAsync((void**) &a, sizeof(float) * N, deviceId);
    cudaMemPrefetchAsync((void**) &b, sizeof(float) * blockCount, deviceId);
    cudaMemPrefetchAsync((void**) &c, sizeof(float), deviceId);

    cudaMemsetAsync(b, 0, sizeof(float) * blockCount);
    cudaMemsetAsync(c, 0, sizeof(float));
    reduce_kernel<<<blockCount, THREAD_NUM, sizeof(float) * THREAD_NUM>>>(a, b, N);
    cudaDeviceSynchronize();
    reduce_kernel<<<1, 256, sizeof(float) * 256>>>(b, c, blockCount);
    cudaDeviceSynchronize();
    printf("cpu: %f, kernel_v1: %f ", ans, c[0]);

    cudaMemsetAsync(b, 0, sizeof(float) * blockCount);
    cudaMemsetAsync(c, 0, sizeof(float));
    reduce_kernel_v2<<<blockCount, THREAD_NUM, sizeof(float) * THREAD_NUM>>>(a, b, N);
    cudaDeviceSynchronize();
    reduce_kernel_v2<<<1, 256, sizeof(float) * 256>>>(b, c, blockCount);
    cudaDeviceSynchronize();

    printf("kernel_v2: %f\n", c[0]);
    

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}