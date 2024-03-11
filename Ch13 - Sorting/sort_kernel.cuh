#define Radix_N 2
#define MAX 0xffff
#define Valid ((1 << Radix_N) - 1)
#include "../Ch11 - Prefix sum (scan) /scan.cuh"
#include "../Ch12 - Merge/merge_kernel.cuh"
#include "../utils/config.cuh"

__device__ __host__
extern void merge_seq(float* A, int A_len, float* B, int B_len, float* C, int st);

__global__
void local_radix_sort(int* a, int* b, int* c, const int N, int iter);

__global__
void global_radix_sort(int *a, int *b, int *c, int *d, const int N, int iter);

__global__
void merge_sort(float* a, float* c, const int N, int stride);