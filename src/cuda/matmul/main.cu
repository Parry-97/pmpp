/**
 * @file main.cu
 * @brief Matrix to Matrix multplication kernel implementation
 * @author Param Pal Singh
 * @chapter 3
 *
 */

#include <cmath>
#include <stdio.h>

/** @brief Thread block dimension for matrix multiplication */
const int BLOCK_SIZE = 2;

/**
 * @brief CUDA kernel for matrix multiplication
 *
 * Performs matrix multiplication P = M * N.
 *
 * @param M First input matrix (row-major)
 * @param N Second input matrix (row-major)
 * @param P Output matrix (row-major)
 * @param width Width and height of square matrices
 */
__global__ void matmul_kernel(float *M, float *N, float *P, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < width && col < width) {
    float outVal = 0;
    for (int i = 0; i < width; i++) {
      outVal += M[row * width + i] * N[i * width + col];
    }

    P[row * width + col] = outVal;
  }
}

/**
 * @brief Host wrapper for matrix multiplication
 *
 * @param h_N First input matrix on host
 * @param h_M Second input matrix on host
 * @param h_P Output matrix on host
 * @param width Width and height of square matrices
 */
void matmul(float *h_N, float *h_M, float *h_P, int width) {
  int size = width * width * sizeof(float);
  float *d_N, *d_M, *d_P;

  // Allocate device memory
  cudaMalloc((void **)&d_N, size);
  cudaMalloc((void **)&d_M, size);
  cudaMalloc((void **)&d_P, size);

  // Copy data to device
  cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);

  dim3 blockSize = dim3(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize =
      dim3(ceil(width / float(BLOCK_SIZE)), ceil(width / float(BLOCK_SIZE)));

  // Launch kernel
  matmul_kernel<<<gridSize, blockSize>>>(d_N, d_M, d_P, width);

  // Copy result back to host
  cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_N);
  cudaFree(d_M);
  cudaFree(d_P);
}

/**
 * @brief Main function
 */
int main() {
  printf("matmul - CUDA implementation\n");
  printf("Chapter 3: Matrix to Matrix multplication kernel implementation\n\n");

  // Simple 3x3 matrix multiplication test
  int width = 3;
  float h_M[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  float h_N[] = {1.0, 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.0, 1.0}; // Identity matrix

  float h_P[width * width];

  printf("Matrix M (3x3):\n");
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      printf("%.1f ", h_M[i * width + j]);
    }
    printf("\n");
  }

  printf("\nMatrix N (3x3 Identity):\n");
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      printf("%.1f ", h_N[i * width + j]);
    }
    printf("\n");
  }

  // Run matrix multiplication
  matmul(h_M, h_N, h_P, width);

  printf("\nResult (M * N):\n");
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      printf("%.1f ", h_P[i * width + j]);
    }
    printf("\n");
  }

  return 0;
}
