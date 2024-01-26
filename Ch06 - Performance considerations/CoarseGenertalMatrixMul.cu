#include <iostream>
#include <cstdio>

#define N 1024
#define M 2048
#define K 900
#define TILE 32
#define COARSE_FACTOR 4

__global__
void matrixMul(int* A, int* B, int* C) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < N && col < K) {
        int res = 0;
        for (int k = 0; k < M; k++) {
            res += A[row * M + k] * B[k * K + col];
        }
        C[row * K + col] = res;
    }
}

__global__
void GeneralMatrixMul(int* A, int* B, int* C) {
    __shared__ int Ads[TILE][TILE];
    __shared__ int Bds[TILE][TILE];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int res[COARSE_FACTOR];
    for (int j = 0; j < COARSE_FACTOR; j++) {
        res[j] = 0;
    }
    for (int i = 0; i < (M + TILE - 1) / TILE; i++) {
        if (by * TILE + ty < N && i * TILE + tx < M) {
            Ads[ty][tx] = A[(by * TILE + ty) * M + i * TILE + tx];
        } else {
            Ads[ty][tx] = 0;
        }
        for (int j = 0; j < COARSE_FACTOR; j++) {
            if (i * TILE + ty < M && bx * COARSE_FACTOR * TILE + j * TILE + tx < K) {
                Bds[ty][tx] = B[(i * TILE + ty) * K + bx * COARSE_FACTOR * TILE + j * TILE + tx];
            } else {
                Bds[ty][tx] = 0;
            }
            __syncthreads(); // read after write
            for (int k = 0; k < TILE; k++) {
                res[j] += Ads[ty][k] * Bds[k][tx];
            }
            __syncthreads(); // write after read
        }
    }
    for (int j = 0; j < COARSE_FACTOR; j++) {
        if (by * TILE + ty < N && bx * COARSE_FACTOR * TILE + j * TILE + tx < K) {
            C[(by * TILE + ty) * K + bx * COARSE_FACTOR * TILE + j * TILE + tx] = res[j];
        }
    }
}

int main() {
    int *A, *B, *C;
    cudaMallocManaged(&A, sizeof(int) * N * M);
    cudaMallocManaged(&B, sizeof(int) * M * K);
    cudaMallocManaged(&C, sizeof(int) * N * K);

    int deviceId;
    cudaGetDevice(&deviceId);

    cudaMemPrefetchAsync(A, sizeof(int) * N * M, cudaCpuDeviceId);
    cudaMemPrefetchAsync(B, sizeof(int) * M * K, cudaCpuDeviceId);

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i * M + j] = rand() % 200;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            B[i * K + j] = rand() % 200;
        }
    }

    int* ans = (int*) malloc(sizeof(int) * N * K);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            ans[i * K + j] = 0; // 初始化，malloc不会自动初始化
        }
    }
    for (int k = 0; k < M; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                ans[i * K + j] += A[i * M + k] * B[k * K + j];
            }
        }
    }

    cudaMemPrefetchAsync(A, sizeof(int) * N * M, deviceId);
    cudaMemPrefetchAsync(B, sizeof(int) * M * K, deviceId);
    cudaMemPrefetchAsync(C, sizeof(int) * N * K, deviceId);

    dim3 grid((K + TILE * COARSE_FACTOR - 1) / (TILE * COARSE_FACTOR), (N + TILE - 1) / TILE); // 注意这里，只跟ouput的维度相关
    dim3 block(TILE, TILE);

    // matrixMul<<<grid, block>>>(A, B, C);
    // cudaDeviceSynchronize();

    GeneralMatrixMul<<<grid, block>>>(A, B, C);
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(C, sizeof(int) * N * K, cudaCpuDeviceId);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            if (ans[i * K + j] != C[i * K + j]) {
                printf("(%d, %d) 有差异\n", i, j);
                printf("ans: %d, C: %d\n", ans[i * K + j], C[i * K + j]);
                return 1;
            }
        }
    }

    printf("比对无差异\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(ans);

    return 0;
}   