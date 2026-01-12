/**
 * @file main.cu
 * @brief Parallel reduction implementation in CUDA
 * @author Param Pal Singh
 * @chapter 10
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void checkCuda(cudaError_t result, const char *func, const char *file,
               int line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorString(result),
            func);
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

/**
 * @brief CUDA kernel for reduction (Interleaved Addressing)
 *
 * This kernel uses interleaved addressing which causes high control divergence.
 * It operates directly on global memory.
 *
 * @param input Input data (size 2 * blockDim.x)
 * @param output Output data (single float)
 */
__global__ void simple_reduction_kernel(float *input, float *output) {
  unsigned int i = 2 * threadIdx.x;

  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

/**
 * @brief CUDA kernel for reduction (Convergent Addressing)
 *
 * This implementation uses convergent (contiguous) addressing to minimize
 * control divergence. However, it still operates on global memory,
 * suffering from memory divergence/latency.
 *
 * @param input Input data (size 2 * blockDim.x)
 * @param output Output data (single float)
 */
__global__ void convergent_reduction_kernel(float *input, float *output) {
  unsigned int i = threadIdx.x;

  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

/**
 * @brief Improved Reduction Kernel (Shared Memory + Thread Coarsening)
 *
 * Improvements:
 * 1. Shared Memory: Loads data into shared memory to avoid repeated global
 * memory accesses during the reduction tree phase.
 * 2. Thread Coarsening: Uses a grid-stride loop (or simple stride since we are
 * single-block) to accumulate multiple elements per thread. This allows the
 * kernel to process an input array of arbitrary size (n) with a single block,
 * and hides global memory latency.
 *
 * @param input Input data
 * @param output Output data
 * @param n Size of the data
 */
__global__ void coarsened_reduction_kernel(float *input, float *output, int n) {
  // Dynamically allocated shared memory
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = tid;

  // Initialize local accumulator
  float sum = 0.0f;

  // --- Thread Coarsening Phase ---
  // Each thread strides over the input array.
  // Stride is blockDim.x because we are using only 1 block.
  while (i < n) {
    sum += input[i];
    i += blockDim.x;
  }

  // Store the partial sum into shared memory
  sdata[tid] = sum;
  __syncthreads();

  // --- Shared Memory Reduction Phase (Convergent) ---
  // Reduce the values in shared memory.
  // Only valid if blockDim.x is a power of 2.
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  // Write the final result
  if (tid == 0) {
    *output = sdata[0];
  }
}

void checkResult(float host_ref, float gpu_ref) {
  // Use a slightly larger epsilon for float accumulation errors
  double epsilon = 1.0e-2;
  if (fabs(host_ref - gpu_ref) > epsilon) {
    printf("FAILED: Host: %f, GPU: %f, Diff: %f\n", host_ref, gpu_ref,
           fabs(host_ref - gpu_ref));
  } else {
    printf("PASSED\n");
  }
}

int main() {
  printf("Reduction - CUDA Implementation\n");
  printf("Comparing Simple, Convergent, and Coarsened kernels.\n\n");

  // Settings
  // block_size must be power of 2 for these kernels
  int block_size = 512;

  // simple/convergent kernels handle exactly 2 * block_size elements
  int n_small = 2 * block_size;

  // coarsened kernel can handle larger arrays
  int n_large = 1 << 16; // 65536 elements

  size_t bytes_small = n_small * sizeof(float);
  size_t bytes_large = n_large * sizeof(float);

  // Allocate Host Memory
  float *h_input_small = (float *)malloc(bytes_small);
  float *h_input_large = (float *)malloc(bytes_large);
  float h_output = 0.0f;

  // Initialize Data
  srand(time(NULL));
  float host_ref_small = 0.0f;
  for (int i = 0; i < n_small; i++) {
    h_input_small[i] = (float)(rand() % 100) / 100.0f;
    host_ref_small += h_input_small[i];
  }

  float host_ref_large = 0.0f;
  for (int i = 0; i < n_large; i++) {
    h_input_large[i] = (float)(rand() % 100) / 100.0f;
    host_ref_large += h_input_large[i];
  }

  // printf("Expected Sum Small: %f\n", host_ref_small); // Debug
  // printf("Expected Sum Large: %f\n", host_ref_large); // Debug

  // Allocate Device Memory
  float *d_input, *d_output;
  // Allocate enough for the large test
  checkCudaErrors(cudaMalloc((void **)&d_input, bytes_large));
  checkCudaErrors(cudaMalloc((void **)&d_output, sizeof(float)));

  // --- Test 1: Simple Reduction ---
  printf("1. Simple Reduction (n=%d, global mem, interleaved)...\n", n_small);
  checkCudaErrors(
      cudaMemcpy(d_input, h_input_small, bytes_small, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_output, 0, sizeof(float))); // Reset output
  simple_reduction_kernel<<<1, block_size>>>(d_input, d_output);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(
      cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
  checkResult(host_ref_small, h_output);

  // --- Test 2: Convergent Reduction ---
  printf("2. Convergent Reduction (n=%d, global mem, contiguous)...\n",
         n_small);
  checkCudaErrors(
      cudaMemcpy(d_input, h_input_small, bytes_small, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_output, 0, sizeof(float))); // Reset output
  convergent_reduction_kernel<<<1, block_size>>>(d_input, d_output);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(
      cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
  checkResult(host_ref_small, h_output);

  // --- Test 3: Coarsened Reduction (Small Input) ---
  printf("3. Coarsened Reduction (n=%d, shared mem + coarsening)...\n",
         n_small);
  checkCudaErrors(
      cudaMemcpy(d_input, h_input_small, bytes_small, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_output, 0, sizeof(float))); // Reset output
  // Shared memory size needed: block_size * sizeof(float)
  coarsened_reduction_kernel<<<1, block_size, block_size * sizeof(float)>>>(
      d_input, d_output, n_small);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(
      cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
  checkResult(host_ref_small, h_output);

  // --- Test 4: Coarsened Reduction (Large Input) ---
  printf("4. Coarsened Reduction (n=%d, showing true coarsening)...\n",
         n_large);
  checkCudaErrors(
      cudaMemcpy(d_input, h_input_large, bytes_large, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_output, 0, sizeof(float))); // Reset output
  coarsened_reduction_kernel<<<1, block_size, block_size * sizeof(float)>>>(
      d_input, d_output, n_large);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(
      cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
  checkResult(host_ref_large, h_output);

  // Free Memory
  free(h_input_small);
  free(h_input_large);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
