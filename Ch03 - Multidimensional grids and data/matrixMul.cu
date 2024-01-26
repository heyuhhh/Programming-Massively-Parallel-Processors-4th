#include <iostream>
#include <cstdio>

#define N 1000

__global__
void matrixMul(int* A, int* B, int* C, const int n) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < n && col < n) {
        int res = 0;
        for (int k = 0; k < n; k++) {
            res += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = res;
    }
}

int main() {
    int size = N * N;
    int *A, *B, *C;
    cudaMallocManaged(&A, sizeof(int) * size);
    cudaMallocManaged(&B, sizeof(int) * size);
    cudaMallocManaged(&C, sizeof(int) * size);

    int deviceId;
    cudaGetDevice(&deviceId);

    cudaMemPrefetchAsync(A, sizeof(int) * size, cudaCpuDeviceId);
    cudaMemPrefetchAsync(B, sizeof(int) * size, cudaCpuDeviceId);

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 200;
            B[i * N + j] = rand() % 200;
        }
    }

    int* ans = (int*) malloc(sizeof(int) * size);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ans[i * N + j] = 0; // 初始化，malloc不会自动初始化
        }
    }
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                ans[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    cudaMemPrefetchAsync(A, sizeof(int) * size, deviceId);
    cudaMemPrefetchAsync(B, sizeof(int) * size, deviceId);
    cudaMemPrefetchAsync(C, sizeof(int) * size, deviceId);

    dim3 grid((N + 15) / 16, (N + 15) / 16);
    dim3 block(16, 16);

    matrixMul<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(C, sizeof(int) * size, cudaCpuDeviceId);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (ans[i * N + j] != C[i * N + j]) {
                printf("(%d, %d) 有差异\n", i, j);
                printf("ans: %d, C: %d\n", ans[i * N + j], C[i * N + j]);
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