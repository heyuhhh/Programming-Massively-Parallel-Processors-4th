/*

xxxx xxxx xxxx xxxx xxxx
xxxxxxxxxxxxxxxxxxxxxxxxx

1234567812345678
*/


#include <cstdio>

#define N 100000000
#define NUM_BINS 7
#define COARSE_FACTOR 32

__global__
void simple_histogram_kernel(char* s_d, int* hist) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N - 1) {
        atomicAdd(&hist[(s_d[n] - 'a') / 4], 1);
    }
}

__global__
void histogram_kernel(char* s_d, int* hist) {
    int n = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    int ed = min(N - 1, (blockIdx.x + 1) * blockDim.x * COARSE_FACTOR);
    __shared__ int hist_s[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++) {
        hist_s[i] = 0;
    }
    for (int i = n; i < ed; i += blockDim.x) {
        atomicAdd(&hist_s[(s_d[i] - 'a') / 4], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        int value = hist_s[i];
        if (value > 0) {
            atomicAdd(&hist[i], hist_s[i]);
        }
    }
}

__global__
void agg_histogram_kernel(char* s_d, int* hist) {
    int n = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    int ed = min(N - 1, (blockIdx.x + 1) * blockDim.x * COARSE_FACTOR);
    __shared__ int hist_s[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++) {
        hist_s[i] = 0;
    }
    int last_bin = -1, accmulator = 0;
    for (int i = n; i < ed; i += blockDim.x) {
        if ((s_d[i] - 'a') / 4 == last_bin) {
            accmulator += 1;
        } else {
            if (last_bin != -1 && accmulator > 0) {
                atomicAdd(&hist[last_bin], accmulator);
            }
            last_bin = (s_d[i] - 'a') / 4;
            accmulator = 1;
        }
    }
    atomicAdd(&hist[last_bin], accmulator);
    __syncthreads();
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        int value = hist_s[i];
        if (value > 0) {
            atomicAdd(&hist[i], hist_s[i]);
        }
    }
}

int main() {
    char *s = (char*) malloc(sizeof(char) * N);
    int *cpu_hist = (int*) malloc(sizeof(int) * NUM_BINS);
    for (int i = 0; i < NUM_BINS; i++) {
        cpu_hist[i] = 0;
    }
    for (int i = 0; i < N - 1; i++) {
        s[i] = rand() % 26 + 'a';
        cpu_hist[(s[i] - 'a') / 4]++;
    }
    s[N - 1] = '\0';
    
    char *s_d;
    int *hist;
    cudaMalloc((void**) &s_d, sizeof(char) * N);
    cudaMalloc((void**) &hist, sizeof(int) * NUM_BINS);
    cudaMemcpy(s_d, s, sizeof(char) * N, cudaMemcpyHostToDevice);

    int block = 128;
    int grid = (N + block - 1) / block;
    simple_histogram_kernel<<<grid, block>>>(s_d, hist);
    cudaDeviceSynchronize();

    cudaMemsetAsync(hist, 0, sizeof(int) * NUM_BINS);
    int block2 = 128;
    int grid2 = (N + block * COARSE_FACTOR - 1) / (block * COARSE_FACTOR);
    histogram_kernel<<<grid2, block2>>>(s_d, hist);
    cudaDeviceSynchronize();

    cudaMemsetAsync(hist, 0, sizeof(int) * NUM_BINS);
    int block3 = 128;
    int grid3 = (N + block * COARSE_FACTOR - 1) / (block * COARSE_FACTOR);
    agg_histogram_kernel<<<grid3, block3>>>(s_d, hist);
    cudaDeviceSynchronize();

    int *gpu_hist = (int*) malloc(sizeof(int) * NUM_BINS);
    cudaMemcpy(gpu_hist, hist, sizeof(int) * NUM_BINS, cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_BINS; i++) {
        printf("%d %d\n", cpu_hist[i], gpu_hist[i]);
        if (cpu_hist[i] != gpu_hist[i]) {
            printf("%d 有差异\n", i);
            printf("cpu hist: %d, gpu hist: %d\n", cpu_hist[i], gpu_hist[i]);
            return 1;
        }
    }
    
    printf("对比无差异\n");

    cudaFree(s_d);
    cudaFree(hist);
    free(cpu_hist);
    free(gpu_hist);
    free(s);
    return 0;
}