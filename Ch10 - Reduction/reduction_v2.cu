// 基于 warp reduce 进行规约
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

#define WARP_SIZE 32
#define INT4(VALUE) (reinterpret_cast<int4*>((&VALUE))[0])
#define FLOAT4(VALUE) (reinterpret_cast<float4*>((&VALUE))[0])

// Warp Reduce Sum
// 使用 __shfl_xor_sync 进行线程束内部规约
template<const int kWarpSize = WARP_SIZE>
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        /*
            mask = 16: [0, 15] ~ [16, 31] 进行规约
            mask = 8 : [0,  7] ~ [ 8, 15] 进行规约
            ...
            其余值不重要
        */
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Warp Reduce Max
template<const int kWarpSize = WARP_SIZE>
__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// Block Reduce Sum
template<const int NUM_THREADS=256>
__device__ float block_reduce_sum(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    val = warp_reduce_sum<WARP_SIZE>(val);
    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();
    val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
    val = warp_reduce_sum<NUM_WARPS>(val);
    return val;
}

// Block Reduce Max
template<const int NUM_THREADS=256>
__device__ float block_reduce_max(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    val = warp_reduce_max<WARP_SIZE>(val);
    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();
    val = (lane < NUM_WARPS) ? smem[lane] : -MAXFLOAT;
    val = warp_reduce_max<NUM_WARPS>(val);
    return val;
}

// Block All Reduce Sum
// a: N*1, y=sum(a)
template<const int NUM_THREADS=256>
__global__ void block_all_reduce_sum(float* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    // 注意这里要让 data 保留在寄存器中
    float sum = (idx < N) ? a[idx] : 0.0f;
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();
    sum = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum<NUM_WARPS>(sum);
    }
    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

template<const int NUM_THREADS=256 / 4>
__global__ void block_all_reduce_sum_vec4(float* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + threadIdx.x) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    float4 reg_a;
    if (idx < N) {
        reg_a = FLOAT4(a[idx]);
    } else {
        reg_a = make_float4(0, 0, 0, 0);
    }
    float sum = 0.0f;
    if (idx + 0 < N) {
        sum += reg_a.x;
    }
    if (idx + 1 < N) {
        sum += reg_a.y;
    }
    if (idx + 2 < N) {
        sum += reg_a.z;
    }
    if (idx + 3 < N) {
        sum += reg_a.w;
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();

    sum = lane < NUM_WARPS ? smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum<NUM_WARPS>(sum);
    }
    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

__device__
void warpReduce(volatile float* a_sm, int tid) { 
    a_sm[tid] += a_sm[tid + 32];
    a_sm[tid] += a_sm[tid + 16];
    a_sm[tid] += a_sm[tid + 8];
    a_sm[tid] += a_sm[tid + 4];
    a_sm[tid] += a_sm[tid + 2];
    a_sm[tid] += a_sm[tid + 1];
}

// thread coarse 访问 4 个数
// 多阶段 reduce
// 只在最后阶段进行 warp reduce
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

template<const int NUM_THREADS=256 / 4>
__global__
void reduce_kernel_v3(float* a, float* b, const int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + threadIdx.x) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    float4 reg_a;
    if (idx < N) {
        reg_a = FLOAT4(a[idx]);
    } else {
        reg_a = make_float4(0, 0, 0, 0);
    }
    float sum = 0.0f;
    if (idx + 0 < N) {
        sum += reg_a.x;
    }
    if (idx + 1 < N) {
        sum += reg_a.y;
    }
    if (idx + 2 < N) {
        sum += reg_a.z;
    }
    if (idx + 3 < N) {
        sum += reg_a.w;
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();

    sum = lane < NUM_WARPS ? smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum<NUM_WARPS>(sum);
    }
    if (tid == 0) {
        b[blockIdx.x] = sum;
    }
}

float testPerformance(dim3 gridDim, dim3 blockDim, const int N, const int repeat, int algorithm);

int main() {
    int N_list[16];
    for (int i = 0; i < 16; i++) {
        N_list[i] = (1 << (i + 10));
    }
    const int outer_repeat = 10, inner_repeat = 1;
    const int TESTNUM = 16;
    for (int i = 0; i < TESTNUM; i++) {
        const int N = N_list[i];
        const int NUM_THREADS = 256;
        double total_sec_0 = 0.0, total_sec_1 = 0.0, total_sec_2 = 0.0;
        dim3 gridDim((N + NUM_THREADS / 4 - 1) / (NUM_THREADS / 4));
        dim3 blockDim(NUM_THREADS >> 2);   
        for (int j = 0; j < outer_repeat; j++) {
            double this_sec_0 = testPerformance(gridDim, blockDim, N, inner_repeat, 0);
            total_sec_0 += this_sec_0;

            double this_sec_1 = testPerformance(gridDim, blockDim, N, inner_repeat, 1);
            total_sec_1 += this_sec_1;

            double this_sec_2 = testPerformance(gridDim, blockDim, N, inner_repeat, 2);
            total_sec_2 += this_sec_2;
        }
        double avg_sec_0 = total_sec_0 / outer_repeat;
        double avg_sec_1 = total_sec_1 / outer_repeat;
        double avg_sec_2 = total_sec_2 / outer_repeat;

        double avg_Gflops_0 = (double)N / 1024 / 1024 / 1024 / avg_sec_0;
        double avg_Gflops_1 = (double)N / 1024 / 1024 / 1024 / avg_sec_1;
        double avg_Gflops_2 = (double)N / 1024 / 1024 / 1024 / avg_sec_2;
        printf("Alg0: N = %6d, Time = %12.8lf s, AVG Performance = %10.4lf Gflops\n", N, avg_sec_0, avg_Gflops_0);
        printf("Alg1: N = %6d, Time = %12.8lf s, AVG Performance = %10.4lf Gflops\n", N, avg_sec_1, avg_Gflops_1);
        printf("Alg2: N = %6d, Time = %12.8lf s, AVG Performance = %10.4lf Gflops\n\n", N, avg_sec_2, avg_Gflops_2);
    }
}

float testPerformance(dim3 gridDim, dim3 blockDim, const int N, const int repeat, int algorithm) {
    float* a;
    cudaMallocManaged((void**) &a, sizeof(float) * N);
    srand(time(NULL));
    float ans = 0;
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float) RAND_MAX;
        ans += a[i];
    }
    int blockCount = gridDim.x;
    float* b, *c;
    cudaMallocManaged((void**) &b, sizeof(float) * blockCount);
    cudaMallocManaged((void**) &c, sizeof(float));
    cudaMemset(b, 0, sizeof(float) * blockCount);
    cudaMemset(c, 0, sizeof(float));
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    float max_error = 0.0;

    if (algorithm == 0) {
        reduce_kernel_v2<<<gridDim, blockDim, sizeof(float) * blockDim.x>>>(a, b, N);
        cudaDeviceSynchronize();
        reduce_kernel_v2<<<1, blockDim, sizeof(float) * blockDim.x>>>(b, c, blockCount);
        cudaDeviceSynchronize();
        max_error = max(max_error, c[0] - ans);
    } else if (algorithm == 1) {
        block_all_reduce_sum_vec4<<<gridDim, blockDim>>>(a, c, N);
        cudaDeviceSynchronize();
        max_error = max(max_error, c[0] - ans);
    } else {
        reduce_kernel_v3<<<gridDim, blockDim, sizeof(float) * blockDim.x>>>(a, b, N);
        cudaDeviceSynchronize();
        reduce_kernel_v3<<<1, blockDim, sizeof(float) * blockDim.x>>>(b, c, blockCount);
        cudaDeviceSynchronize();
        max_error = max(max_error, c[0] - ans);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    printf("ans=%f, c[0]=%f\n", ans, c[0]);

    printf("Algorithm %d's max error is : %f\n", algorithm, max_error);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000 / repeat;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return sec;
}