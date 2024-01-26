#include <cstdio>

#define N 10024
#define M 11000
#define C 3
#define TILE 16
#define R 5

__constant__ float F_c[R * R];

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
__global__ void convolution(float *I,  float *P) {

    __shared__ float N_ds[w][w];
    
    int k;
    for (k = 0; k < C; k++) {
        
        // First batch loading
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;
        int destX = dest % w;
        int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
        int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
        int src = (srcY * M + srcX) * C + k;
        
        if (srcY >= 0 && srcY < N && srcX >= 0 && srcX < M) {
            N_ds[destY][destX] = I[src];
        } else {
            N_ds[destY][destX] = 0;
        }

        __syncthreads();
    
        float accum = 0;
        int y, x;
        for (y = 0; y < Mask_width; y++) {
            for (x = 0; x < Mask_width; x++) {
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * F_c[y * Mask_width + x];
            }
        }
        
        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < N && x < M)
            P[(y * M + x) * C + k] = accum;
        __syncthreads();
    }
}

// 外围直接访问全局内存
__global__ void convolution_tiled_2D_const_mem_kernel(float *P, float *res) {
    __shared__ float P_s[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    for (int k = 0; k < C; k++) { // 枚举通道
        if (row < N && col < M) {
            P_s[threadIdx.y][threadIdx.x] = P[row * M * C + col * C + k];
        } else {
            P_s[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        if (row < N && col < M) {
            int stRow = blockIdx.y * TILE;
            int stCol = blockIdx.x * TILE;
            float value = 0.0f;
            for (int i = 0; i < R; i++) {
                for (int j = 0; j < R; j++) {
                    int curRow = row - R / 2 + i, curCol = col - R / 2 + j;
                    if (curRow >= stRow && curRow < stRow + TILE && curCol >= stCol && curCol < stCol + TILE) {
                        value += P_s[curRow % TILE][curCol % TILE] * F_c[i * R + j];
                    } else {
                        if (curRow >= 0 && curRow < N && curCol >= 0 && curCol < M) { // 外围就直接访问全局内存（大概率存储在缓存中
                            value += P[curRow * M * C + curCol * C + k] * F_c[i * R + j];
                        }
                    }
                }
            }
            res[row * M * C + col * C + k] = value;
        }
        __syncthreads(); // 注意：不能在 if 分支里面进行同步
    }
}

// 更大的input tile，会有少量线程浪费
__global__ void convolution_tiled_2D_const_mem_kernel_v2(float *P, float *res) {
    __shared__ float P_s[TILE + R - 1][TILE + R - 1];
    int row = blockIdx.y * TILE + threadIdx.y - R / 2; // 注意这里是 TILE 而不是 TILE + R - 1
    int col = blockIdx.x * TILE + threadIdx.x - R / 2; // 我们要以每个TILE找到对应的左上角，后者会产生偏移
    for (int k = 0; k < C; k++) {
        if (row >= 0 && row < N && col >= 0 && col < M) {
            P_s[threadIdx.y][threadIdx.x] = P[row * M * C + col * C + k];
        } else {
            P_s[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        if (threadIdx.y >= R / 2 && threadIdx.y < TILE + R - 1 - R / 2 && 
            threadIdx.x >= R / 2 && threadIdx.x < TILE + R - 1 - R / 2 &&
            row < N && col < M) {
            float value = 0.0f;
            for (int i = 0; i < R; i++) {
                for (int j = 0; j < R; j++) {
                    value += F_c[i * R + j] * P_s[threadIdx.y - R / 2 + i][threadIdx.x - R / 2 + j];
                }
            }
            res[row * M * C + col * C + k] = value;
        }
        __syncthreads();
    }
}

int main() {
    float *P = (float*) malloc(sizeof(float) * N * M * C);
    float *F = (float*) malloc(sizeof(float) * R * R);
    // srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < C; k++) {
                P[i * M * C + j * C + k] = rand() % 256;
            }
        }
    }
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            F[i * R + j] = float(rand() % 256) / 256 / (R * R);
        }
    }
    cudaMemcpyToSymbol(F_c, F, sizeof(float) * R * R); // 将静态全局变量传输到设备中
    
    float *P_d, *R_d, *R2_d, *R3_d;
    cudaMalloc((void**) &P_d, sizeof(float) * N * M * C);
    cudaMalloc((void**) &R_d, sizeof(float) * N * M * C);
    cudaMalloc((void**) &R2_d, sizeof(float) * N * M * C);
    cudaMalloc((void**) &R3_d, sizeof(float) * N * M * C);
    cudaMemcpy(P_d, P, sizeof(float) * N * M * C, cudaMemcpyHostToDevice); // 将图像传输到设备中

    dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    convolution_tiled_2D_const_mem_kernel<<<grid, block>>>(P_d, R_d);
    cudaDeviceSynchronize();
    printf("after kernel function : %s\n",cudaGetErrorString(cudaGetLastError()));
    

    dim3 grid2((M + R - 1 + TILE - 1) / TILE, (N + R - 1 + TILE - 1) / TILE);
    dim3 block2(TILE + R - 1, TILE + R - 1);
    convolution_tiled_2D_const_mem_kernel_v2<<<grid2, block2>>>(P_d, R2_d);
    cudaDeviceSynchronize();
    printf("after kernel v2 function : %s\n",cudaGetErrorString(cudaGetLastError()));

    // 下面这部分有 bug
    // dim3 grid3((float)(M + TILE_WIDTH - 1) / TILE_WIDTH,
    //              (float)(N + TILE_WIDTH - 1) / TILE_WIDTH);
    // dim3 block3(TILE_WIDTH, TILE_WIDTH);
    // convolution<<<grid3, block3>>>(P_d, R3_d);
    // cudaDeviceSynchronize();
    // printf("after kernel v3 function : %s\n",cudaGetErrorString(cudaGetLastError()));

    float *res1, *res2, *res3;
    res1 = (float*) malloc(sizeof(float) * N * M * C);
    res2 = (float*) malloc(sizeof(float) * N * M * C);
    res3 = (float*) malloc(sizeof(float) * N * M * C);
    cudaMemcpy(res1, R_d, sizeof(float) * N * M * C, cudaMemcpyDeviceToHost);
    cudaMemcpy(res2, R2_d, sizeof(float) * N * M * C, cudaMemcpyDeviceToHost);
    cudaMemcpy(res3, R3_d, sizeof(float) * N * M * C, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < C; k++) {
                // printf("%f\n", res2[i * M * C + j * C + k]);
                if (res1[i * M * C + j * C + k] != res2[i * M * C + j * C + k]) {
                    printf("(%d %d %d) 有差异\n", i, j, k);
                    printf("v1 = %f, v2 = %f, v3 = %f\n", res1[i * M * C + j * C + k], res2[i * M * C + j * C + k], res3[i * M * C + j * C + k]);
                    return 1;
                }
            }
        }
    }

    printf("比对无差异\n");

    cudaFree(P_d);
    cudaFree(R_d);
    cudaFree(R2_d);
    free(res1);
    free(res2);
    free(P);
    free(F);
    return 0;
}