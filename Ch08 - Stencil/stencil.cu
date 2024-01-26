#include <cstdio>

#define N 100
#define M 102
#define C 10

#define in(i, j, k) _in[(i) * M * C + (j) * C + k]
#define out(i, j, k) _out[(i) * M * C + (j) * C + k]

#define OUT_TILE_DIM 16
#define IN_TILE_DIM OUT_TILE_DIM + 2

/*
thread coarseing 通过串行化某些操作来节约空间，减轻共享内存以及核函数大小的压力，提高运算传输比
如果没有这个的话，首先三维核函数大小较低，其次数据很多重复传输，复用率不高
register tiling 基于 thread coarseing 的优化
*/
__global__
void stencilScan(float *_in, float *_out) {
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1; // 左上偏移一维
    __shared__ float in_ds[IN_TILE_DIM][IN_TILE_DIM];

    float curValue, lastValue, nextValue; // register tiling
    for (int k = 0; k < C; k++) {  // thread coarseing
        if (row >= 0 && row < N && col >= 0 && col < M) {
            in_ds[threadIdx.y][threadIdx.x] = in(row, col, k);
        } else {
            in_ds[threadIdx.y][threadIdx.x] = 0;
        }
        curValue = in_ds[threadIdx.y][threadIdx.x];
        if (k + 1 < C) {
            nextValue = in(row, col, k + 1);
        } else {
            nextValue = 0;
        }
        __syncthreads();
        if (threadIdx.y > 0 && threadIdx.y < IN_TILE_DIM - 1 &&
            threadIdx.x > 0 && threadIdx.x < IN_TILE_DIM - 1 &&
            row > 0 && row < N - 1 && col > 0 && col < M - 1 && k > 0 && k < C - 1) {
            out(row, col, k) = lastValue + nextValue + in_ds[threadIdx.y - 1][threadIdx.x] +
                    in_ds[threadIdx.y + 1][threadIdx.x] + 
                    in_ds[threadIdx.y][threadIdx.x - 1] + 
                    in_ds[threadIdx.y][threadIdx.x + 1] - curValue * 6;  
        }
        lastValue = curValue;
        curValue = nextValue;
        __syncthreads();
    }
}

void cpuStencilScan(float *_out, float *_in) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < M - 1; j++) {
            for (int k = 1; k < C - 1; k++) {
                out(i, j, k) = in(i - 1, j, k) + in(i + 1, j, k) + 
                                in(i, j - 1, k) + in(i, j + 1, k) + 
                                in(i, j, k - 1) + in(i, j, k + 1) - 6 * in(i, j, k);
            }
        }
    }
}

int main() {
    float *P = (float*) malloc(sizeof(float) * N * M * C);
    for (int i = 0; i < N * M * C; i++) {
        P[i] = (float) rand() / RAND_MAX * 255;
    }
    float *P_d, *R_d;
    float *res1, *res2;
    res1 = (float*) malloc(sizeof(float) * N * M * C);
    res2 = (float*) malloc(sizeof(float) * N * M * C);

    for (int i = 0; i < N * M * C; i++) {
        res1[i] = res2[i] = P[i];
    }
    cpuStencilScan(res2, P);

    cudaMalloc((void**) &P_d, sizeof(float) * N * M * C);
    cudaMalloc((void**) &R_d, sizeof(float) * N * M * C);
    cudaMemcpy(P_d, P, sizeof(float) * N * M * C, cudaMemcpyHostToDevice);
    cudaMemcpy(R_d, res1, sizeof(float) * N * M * C, cudaMemcpyHostToDevice);

    dim3 grid((M + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    dim3 block(IN_TILE_DIM, IN_TILE_DIM);

    stencilScan<<<grid, block>>>(P_d, R_d);
    cudaDeviceSynchronize();
    printf("after the kernel, %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(res1, R_d, sizeof(float) * N * M * C, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N * M * C; i++) {
        if (abs(res1[i] - res2[i]) > 1e-3) {
            printf("(%d, %d, %d) 有差异\n", i / (M * C), i % (M * C) / C, i % (M * C) % C);
            printf("res1: %f, res2: %f\n", res1[i], res2[i]);
            return 1;
        }
    }
    printf("对比无差异\n");

    cudaFree(P_d);
    cudaFree(R_d);
    free(res1);
    free(res2);
    free(P);
    return 0;
}