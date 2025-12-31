/**
 * @file main.cu
 * @brief Parallel reduction implementation in CUDA
 * @author Param Pal Singh
 * @chapter 10
 *
 */

#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>

/**
 * @brief CUDA kernel for reduction
 *
 * TODO: Implement reduction kernel
 *
 * @param input Input data
 * @param output Output data
 * @param n Size of the data
 */
__global__ void simple_reduction_kernel(float *input, float *output) {
  unsigned int i = 2 * threadIdx.x;

  // NOTE: We start off with the even threads indices and elements (0, 2, 4 ...)
  // At each stride (1, 2, 4, ...) we accumulate the element referenced by i
  // and the i + stride element, while having covered the entire input array by
  // the end
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    // INFO: Each step in the reduction process is accompanied by sychronization
    __syncthreads();
  }

  // WARN: We are accumulating the final result in the first element
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

/**
 * @brief Host wrapper function for reduction
 *
 * @param h_input Host input data
 * @param h_output Host output data
 * @param n Size of the data
 */
void reduction(float *h_input, float *h_output, int n) {
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
  simple_reduction_kernel<<<gridSize, blockSize>>>(d_input, d_output);

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
  printf("reduction - CUDA implementation\n");
  printf("Chapter 10: Parallel reduction implementation in CUDA\n\n");

  // TODO: Implement test/demo code

  return 0;
}
