#include "merge_kernel.cuh"
#include <cstdio>

__device__ __host__
int co_rank(int k, volatile float* A, int n, volatile float* B, int m) {
    int l = 0, r = min(k, n), mid;
    while (l < r) {
        mid = (l + r) >> 1;
        if (k - mid - 1 >= m || (k - mid - 1 >= 0 && A[mid] < B[k - mid - 1])) {
            l = mid + 1;
        } else {
            r = mid;
        } 
    }
    return r;
}

__device__ __host__
int co_rank_cir(int k, float* A, int n, float* B, int m, int A_start, int B_start) {
    int l = 0, r = min(k, n), mid;
    while (l < r) {
        mid = (l + r) >> 1;
        int A_ind = (A_start + mid) % TILE_SIZE;
        int B_ind = (B_start + k - mid) % TILE_SIZE;
        if (k - mid - 1 >= m || (k - mid - 1 >= 0 && A[A_ind] < B[(B_ind - 1 + TILE_SIZE) % TILE_SIZE])) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return r;
}

__device__ __host__
void merge_circular(volatile float* A, int A_start, int A_len,
                    volatile float* B, int B_start, int B_len, float* C) {
    int A_cnt = 0, B_cnt = 0;
    while (A_cnt < A_len && B_cnt < B_len) {
        if (A[A_start] < B[B_start]) {
            C[A_cnt + B_cnt] = A[A_start];
            A_start = (A_start + 1) % TILE_SIZE;
            ++A_cnt;
        } else {
            C[A_cnt + B_cnt] = B[B_start];
            B_start = (B_start + 1) % TILE_SIZE;
            ++B_cnt;
        }
    }
    while (A_cnt < A_len) {
        C[A_cnt + B_cnt] = A[A_start];
        A_start = (A_start + 1) % TILE_SIZE;
        ++A_cnt;
    }
    while (B_cnt < B_len) {
        C[A_cnt + B_cnt] = B[B_start];
        B_start = (B_start + 1) % TILE_SIZE;
        ++B_cnt;
    }
}

__global__
void circularMergeSort(float* A, float* B, float* C, const int N, const int M, const int K) {
    __shared__ float AB_S[TILE_SIZE << 1];
    float* A_S = AB_S;
    float* B_S = AB_S + TILE_SIZE;
    int C_cur = blockIdx.x * TILE_SIZE;
    int C_nxt = min((blockIdx.x + 1) * TILE_SIZE, K);
    // 找到这个块的下标范围
    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_cur, A, N, B, M);
        A_S[1] = co_rank(C_nxt, A, N, B, M);
    }
    __syncthreads();
    int A_cur = A_S[0], A_nxt = A_S[1];
    int B_cur = C_cur - A_cur, B_nxt = C_nxt - A_nxt;
    __syncthreads();
    // 变量初始化
    int A_length = A_nxt - A_cur;
    int B_length = B_nxt - B_cur;
    int C_length = C_nxt - C_cur;
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("%d %d %d %d %d %d\n", A_cur, A_nxt, B_cur, B_nxt, C_cur, C_nxt);
    // }
    int counter = 0, totIteration = (C_length + TILE_SIZE - 1) / TILE_SIZE;
    int A_S_start = 0, B_S_start = 0;
    int A_S_consumed = TILE_SIZE, B_S_consumed = TILE_SIZE;
    int A_consumed = 0, B_consumed = 0, C_completed = 0;
    while (counter < totIteration) {
        ++counter;
        // 读取数据到共享内存
        for (int i = 0; i < A_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed && 
                    i + threadIdx.x < A_S_consumed) {
                A_S[(A_S_start + i + threadIdx.x + TILE_SIZE - A_consumed) % TILE_SIZE] = 
                    A[A_cur + A_consumed + i + threadIdx.x];
            }
        }
        for (int i = 0; i < B_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed && 
                    i + threadIdx.x < B_S_consumed) {
                B_S[(B_S_start + i + threadIdx.x + TILE_SIZE - B_consumed) % TILE_SIZE] = 
                    B[B_cur + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();
        // if (threadIdx.x == 0 && blockIdx.x == 0)
        // printf("%d\n", counter);
        // 找到每个线程的范围
        int c_cur = threadIdx.x * (TILE_SIZE / blockDim.x);
        int c_nxt = (threadIdx.x + 1) * (TILE_SIZE / blockDim.x);
        
        c_cur = (c_cur <= C_length - C_completed ? c_cur : C_length - C_completed);
        c_nxt = (c_nxt <= C_length - C_completed ? c_nxt : C_length - C_completed);

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
            // for (int i = 0; i < 10; i++) {
            //     printf("%f %f\n", A_S[i], B_S[i]);
            // }
        //     printf("%d, %d, %d\n", c_nxt, A_S_start, B_S_start);
        // }
        int a_cur = co_rank_cir(c_cur, A_S, min(TILE_SIZE, A_length - A_consumed),
                                    B_S, min(TILE_SIZE, B_length - B_consumed),
                                    A_S_start, B_S_start);
        int b_cur = c_cur - a_cur;
        int a_nxt = co_rank_cir(c_nxt, A_S, min(TILE_SIZE, A_length - A_consumed),
                                    B_S, min(TILE_SIZE, B_length - B_consumed),
                                    A_S_start, B_S_start);
        int b_nxt = c_nxt - a_nxt;

        // if (threadIdx.x + blockIdx.x == 0) {
        //     printf("%d %d %d %d %d %d\n", a_cur, a_nxt, b_cur, b_nxt, c_cur, c_nxt);
        // }
        // 归并
        // printf("%d %d %d %d\n", C_cur, C_completed, threadIdx.x, C_cur + C_completed + c_cur);
        // if (threadIdx.x + blockIdx.x == 0) {
        //     printf("%d %d %d %d\n", C_cur, C_completed, threadIdx.x, C_cur + C_completed + c_cur);
        // }
        merge_circular(A_S, (A_S_start + a_cur) % TILE_SIZE, a_nxt - a_cur,
                        B_S, (B_S_start + b_cur) % TILE_SIZE, b_nxt - b_cur, 
                        C + C_cur + C_completed + c_cur);

        // if (threadIdx.x + blockIdx.x == 0) {
        //     printf("%f %f %f %f\n", C[0], C[1], C[2], C[3]);
        // }
        // 更新相关变量
        A_S_consumed = co_rank_cir(min(TILE_SIZE, C_length - C_completed), 
                                    A_S, min(TILE_SIZE, A_length - A_consumed),
                                    B_S, min(TILE_SIZE, B_length - B_consumed),
                                    A_S_start, B_S_start);
        B_S_consumed = min(TILE_SIZE, C_length - C_completed) - A_S_consumed;
        A_consumed += A_S_consumed;
        B_consumed += B_S_consumed;
        C_completed += min(TILE_SIZE, C_length - C_completed);

        A_S_start = (A_S_start + A_S_consumed) % TILE_SIZE;
        B_S_start = (B_S_start + B_S_consumed) % TILE_SIZE;
        __syncthreads();
    }
}