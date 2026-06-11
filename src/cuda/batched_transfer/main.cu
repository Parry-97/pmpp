/**
 * @file main.cu
 * @brief CUDA Batched Transfer exploration
 * @author Param Pal Singh
 * @chapter 3
 *
 */

#include <iostream>
#include <stdio.h>
#include <vector>

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
 * @brief CUDA kernel for batched_transfer
 *
 * TODO: Implement batched_transfer kernel
 *
 * @param input Input data
 * @param output Output data
 * @param n Size of the data
 */
__global__ void batched_transfer_kernel(float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    // TODO: Implement kernel logic
    output[i] = input[i];
  }
}

void print_array(float *v, int n, int limit = 5) {
  for (int i = 0; i < n && i < limit; i++) {
    std::cout << v[i] << " ";
  }
  if (n > limit)
    std::cout << "... (and " << (n - limit) << " more)";
}

/**
 * @brief Host wrapper function for batched_transfer
 *
 * @param h_input Host input data
 * @param h_output Host output data
 * @param n Size of the data
 */
void batched_transfer(float *h_input, float *h_output, int n) {
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
  batched_transfer_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

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
  printf("batched_transfer - CUDA implementation\n");
  printf("Chapter 3: CUDA Batched Transfer exploration\n\n");

  int batch_size = 2000;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // TODO: Implement test/demo code
  std::vector<void *> srcs(batch_size);
  std::vector<void *> dsts(batch_size);
  std::vector<size_t> sizes(batch_size);
  // Allocate the source and destination buffers
  // initialize with the stream number
  for (size_t i = 0; i < batch_size; i++) {
    cudaMallocHost(&srcs[i], sizes[i]);
    cudaMalloc(&dsts[i], sizes[i]);
    cudaMemsetAsync(srcs[i], 0, sizes[i], stream);
  }
  // Setup attributes for this batch of copies
  cudaMemcpyAttributes attrs = {};
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  // All copies in the batch have same copy attributes.size_t attrsIdxs = 0;
  // Index of the attributes return 0;
  size_t attrsIdxs = 0;
  cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size, &attrs,
                       &attrsIdxs, 1 /* numAttrs */, nullptr /* failIdx */,
                       stream);
}
