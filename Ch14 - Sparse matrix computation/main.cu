#include <cstdio>
#include <iostream>
#include "sparse_matrix.cuh"
#define MAX 100
#define TILE_DIM 32
#define BLOCK_ROWS (TILE_DIM / 4)

int main() {
    int rows = 40, cols = 30;
    int *A, *A_out;
    cudaMallocManaged((void**) &A, rows * cols * sizeof(int));
    cudaMallocManaged((void**) &A_out, rows * cols * sizeof(int));
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            A[i * cols + j] = rand() % MAX;
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
    dim3 dimGrid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    transpose<<<dimGrid, dimBlock>>>(A,  A_out,  rows, cols);
    cudaDeviceSynchronize();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << A[i * cols + j] << ' ';
        } std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            std::cout << A_out[i * rows + j] << ' ';
        } std::cout << std::endl;
    }
    return 0;
}