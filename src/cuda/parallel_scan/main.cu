/**
 * @file main.cu
 * @brief Parallel scan implementation in CUDA
 * @author Param Pal Singh
 * @chapter 11
 *
 */

#include <stdio.h>

/**
 * @brief CUDA kernel for parallel_scan
 *
 * TODO: Implement parallel_scan kernel
 *
 * @param input Input data
 * @param output Output data
 * @param n Size of the data
 */
__global__ void parallel_scan_kernel(float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    // TODO: Implement kernel logic
    output[i] = input[i];
  }
}

const int SECTION_SIZE = 16;

__global__ void kogge_stone_scan_kernel(float *X, float *Y, unsigned int N) {
  // Shared memory buffer for the section
  __shared__ float XY[SECTION_SIZE];

  // Calculate global index
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data from global memory to shared memory
  // Pad with 0 if the index is out of bounds
  if (i < N) {
    XY[threadIdx.x] = X[i];
  } else {
    XY[threadIdx.x] = 0.0f;
  }

  // Iterative scan loop
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads(); // Ensure previous iteration's writes are visible

    float temp;

    // If the thread index is high enough to have a neighbor 'stride' away
    if (threadIdx.x >= stride) {
      temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
    }

    __syncthreads(); // Barrier to prevent write-after-read race conditions

    // Write the computed sum to shared memory
    if (threadIdx.x >= stride) {
      XY[threadIdx.x] = temp;
    }
  }

  // Write results back to global memory
  if (i < N) {
    Y[i] = XY[threadIdx.x];
  }
}

/**
 * @brief Host wrapper function for parallel_scan
 *
 * @param h_input Host input data
 * @param h_output Host output data
 * @param n Size of the data
 */
void parallel_scan(float *h_input, float *h_output, int n) {
  int size = n * sizeof(float);
  float *d_input, *d_output;

  // Allocate device memory
  cudaMalloc((void **)&d_input, size);
  cudaMalloc((void **)&d_output, size);

  // Copy data to device
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  dim3 blockSize;
  dim3 gridSize;

  // Launch kernel
  parallel_scan_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

  // Copy result back to host
  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Main function
 */
int main() {
  printf("parallel_scan - CUDA implementation\n");
  printf("Chapter 11: Parallel scan implementation in CUDA\n\n");

  // TODO: Implement test/demo code

  return 0;
}
