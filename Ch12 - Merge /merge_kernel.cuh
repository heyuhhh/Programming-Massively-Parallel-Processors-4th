#include "../utils/config.cuh"

__device__ __host__
int co_rank(int k, volatile float* A, int n, volatile float* B, int m);

__device__ __host__
int co_rank_cir(int k, float* A, int n, float* B, int m, int A_start, int B_start);

__device__ __host__
void merge_circular(volatile float* A, int A_start, int A_len,
                    volatile float* B, int B_start, int B_len, float* C);

__global__
void circularMergeSort(float* A, float* B, float* C, const int N, const int M, const int K);

