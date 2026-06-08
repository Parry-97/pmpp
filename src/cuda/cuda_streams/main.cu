/**
 * @file main.cu
 * @brief CUDA streams exploration
 * @author Param Pal Singh
 * @chapter 3
 */

#include <iostream>
#include <ostream>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err),           \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/**
 * @brief CUDA kernel for cuda_streams
 */
__global__ void cuda_streams_kernel(float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    output[i] = input[i];
  }
}

// Fixed: Only print the first 'limit' elements so we don't spam the terminal
void print_array(float *v, int n, int limit = 5) {
  for (int i = 0; i < n && i < limit; i++) {
    std::cout << v[i] << " ";
  }
  if (n > limit)
    std::cout << "... (and " << (n - limit) << " more)";
}

/**
 * @brief Host wrapper function for cuda_streams
 */
void cuda_streams(float *h_input, float *h_output, int n) {
  int size = n * sizeof(float);
  float *d_input, *d_output;

  // FIX 1: Use cudaMalloc for ACTUAL Device (GPU) memory allocation
  CUDA_CHECK(cudaMalloc((void **)&d_input, size));
  CUDA_CHECK(cudaMalloc((void **)&d_output, size));

  cudaStream_t stream_handle;
  CUDA_CHECK(cudaStreamCreate(&stream_handle));

  // Wrap async calls in CUDA_CHECK to catch issues early
  CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice,
                             stream_handle));

  std::cout << "The initial input vector is: ";
  print_array(h_input, n);
  std::cout << std::endl;

  // FIX 2: Calculate correct grid/block dimensions for 10M elements
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel across multiple blocks safely
  cuda_streams_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream_handle>>>(
      d_input, d_output, n);

  // Check if the kernel launch configuration itself was valid
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, size, cudaMemcpyDeviceToHost,
                             stream_handle));

  // Sync the stream so the host data is actually ready before we print it
  CUDA_CHECK(cudaStreamSynchronize(stream_handle));

  std::cout << "The final output vector is:  ";
  print_array(h_output, n);
  std::cout << std::endl;

  CUDA_CHECK(cudaStreamDestroy(stream_handle));

  // FIX 1 cleanup: Use regular cudaFree for device pointers
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

/**
 * @brief Main function
 */
int main() {
  printf("Chapter 3: CUDA streams exploration\n\n");

  int n = 10000000; // 10 Million elements

  // Dynamically allocate both host arrays on CPU
  float *input = new float[n];
  float *output = new float[n];

  // Fill the input array with dummy data
  for (int i = 0; i < n; i++) {
    input[i] = 1.0f * i;
    output[i] = 0.0f;
  }

  cuda_streams(input, output, n);

  // Explicitly force a profiling flush and cleanup
  cudaDeviceSynchronize();

  delete[] input;
  delete[] output;
  return 0;
}
