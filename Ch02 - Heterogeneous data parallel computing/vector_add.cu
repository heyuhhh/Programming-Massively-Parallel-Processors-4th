#include <cstdio>
#include <iostream>

__global__
void vec_add(float* A, float* B, float* C, const int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    float *a, *b, *c;

    int len = n * sizeof(float);

    int deviceId;
    cudaGetDevice(&deviceId);
    // 使用统一内存管理
    cudaMallocManaged(&a, len);
    cudaMallocManaged(&b, len);
    cudaMallocManaged(&c, len);

    // 异步预取到cpu上
    cudaMemPrefetchAsync(a, len, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, len, cudaCpuDeviceId);

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i + i;
    }

    float *ans = (float*)malloc(len);
    for (int i = 0; i < n; i++) {
        ans[i] = a[i] + b[i];
    }

    // 异步预取到gpu上
    cudaMemPrefetchAsync(a, len, deviceId);
    cudaMemPrefetchAsync(b, len, deviceId);
    cudaMemPrefetchAsync(c, len, deviceId);

    vec_add<<<(n + 255) / 256, 256>>>(a, b, c, n);
    // 异步调用，所以需要同步一下
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(c, len, cudaCpuDeviceId);

    for (int i = 0; i < n; i++) {
        if (c[i] != ans[i]) {
            std::cout << "有差异!" << std::endl;
            return 1;
        }
    }
    std::cout << "无差异" << std::endl;

    // 释放空间
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free(ans);

    return 0;
}