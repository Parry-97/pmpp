/**
 * @file main.cu
 * @brief CUDA Events exploration
 * @author Param Pal Singh
 * @chapter 3
 *
 */

#include <cstdlib>
#include <iostream>
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
 * @brief CUDA kernel for cuda_events
 *
 * TODO: Implement cuda_events kernel
 *
 * @param input Input data
 * @param output Output data
 * @param n Size of the data
 */
__global__ void cuda_events_kernel(float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    // TODO: Implement kernel logic
    output[i] = input[i] + 34;
  }
}

void print_array(float *v, int n) {
  for (int i = 0; i < n; i++) {
    std::cout << v[i] << " ";
  }
}

/**
 * @brief Host wrapper function for cuda_events
 *
 * @param h_input Host input data
 * @param h_output Host output data
 * @param n Size of the data
 */
void cuda_events(float *h_input, float *h_output, int n) {
  int size = n * sizeof(float);
  float *d_input, *d_output;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaEvent_t start_event;
  CUDA_CHECK(cudaEventCreate(&start_event));

  cudaEvent_t stop_event;
  CUDA_CHECK(cudaEventCreate(&stop_event));

  // Allocate device memory
  CUDA_CHECK(cudaMallocHost((void **)&d_input, size));
  CUDA_CHECK(cudaMallocHost((void **)&d_output, size));

  // Copy data to device
  cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream);

  int blockSize = n;
  int gridSize = 1;

  // Launch kernel
  cudaEventRecord(start_event, stream);
  cuda_events_kernel<<<gridSize, blockSize, 20, stream>>>(d_input, d_output, n);
  cudaEventRecord(stop_event, stream);

  cudaStreamSynchronize(stream);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

  std::cout << "The kernel execution time is " << elapsed_time << "ms"
            << std::endl;

  // Copy result back to host
  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

  print_array(h_output, n);

  cudaStreamDestroy(stream);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  // Free device memory
  cudaFreeHost(d_input);
  cudaFreeHost(d_output);
}

/**
 * @brief Main function
 */
int main() {
  std::cout << "cuda_events - CUDA implementation\n" << std::endl;
  std::cout << "Chapter 3: CUDA Events exploration\n" << std::endl;

  int n = 1024;
  float *input = (float *)(malloc(n * sizeof(float)));
  float *output = (float *)(malloc(n * sizeof(float)));

  cuda_events(input, output, n);
  cudaDeviceSynchronize();

  free(input);
  free(output);

  return 0;
}
