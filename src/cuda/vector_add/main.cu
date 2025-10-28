/**
 * @file main.cu
 * @brief Vector addition kernel implementation
 * @author Param Pal Singh
 * @chapter 2
 *
 */

#include <stdio.h>

__global__ void hello_from_gpu() { printf("Hello from the GPU!\n"); }

/**
 * @brief CUDA kernel for vector addition
 *
 * Computes element-wise addition of two vectors in parallel.
 *
 * @param d_A First input vector on device
 * @param d_B Second input vector on device
 * @param d_C Output vector on device
 * @param n Length of the vectors
 */
__global__ void vec_add_kernel(float *d_A, float *d_B, float *d_C, int n) {

  // NOTE: The `__global__` keyword indicates that the function is a kernel and
  // that it can be called to generate a grid of threads on a device.

  // NOTE: a unique global index i is calculated
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // WARN: This is because not all vector lengths can be expressed as multiples
  // of the block size. For example, let’s assume that the vector length is 100,
  // the smallest efficient thread block dimension is 32. Assume that we picked
  // 32 as block size. One would need to launch 4 thread blocks to process all
  // the 100 vector elements. However, the 4 thread blocks would have 128
  // threads. We need to disable the last 28 threads in thread block 3 from
  // doing work not expected by the original program.
  // Since all threads are to execute the same code, all will test their i
  // values against n, which is 100. With the if (i<n) statement, the first 100
  // threads will perform the addition, whereas the last 28 will not. This
  // allows the kernel to be called to process vectors of arbitrary lengths.
  if (i < n) {

    // NOTE: Note that all the thread blocks operate on different parts of the
    // vectors. They can be executed in any arbitrary order. The programmer must
    // not make any assumptions regarding execution order.
    d_C[i] = d_A[i] + d_B[i];
  }
}

/**
 * @brief Sequential vector addition on host
 *
 * @param h_A First input vector on host
 * @param h_B Second input vector on host
 * @param h_C Output vector on host
 * @param n Length of the vectors
 */

void seq_vec_add(float *h_A, float *h_B, float *h_C, int n) {
  for (int i = 0; i < n; i++) {
    h_C[i] = h_A[i] + h_B[i];
  }
}

/**
 * @brief Parallel vector addition using CUDA
 *
 * Allocates device memory, copies data to device, launches kernel, and copies
 * result back.
 *
 * @param h_A First input vector on host
 * @param h_B Second input vector on host
 * @param h_C Output vector on host
 * @param n Length of the vectors
 */
void par_vecAdd(float *h_A, float *h_B, float *h_C, int n) {
  int size = n * sizeof(float);
  float *d_A, *d_B, *d_C;

  // WARN: The first parameter to the cudaMalloc function is the address of a
  // pointer variable that will be set to point to the allocated object. The
  // address of the pointer variable should be cast to (void **) because the
  // function expects a generic pointer; the memory allocation function is a
  // generic function that is not restricted to any particular type of objects.
  //
  // NOTE: This parameter allows the cudaMalloc function to write the address of
  // the allocated memory into the provided pointer variable regardless of its
  // type. The host code that calls kernels passes this pointer value to the
  // kernels that need to access the allocated memory object.
  cudaError_t errA = cudaMalloc((void **)&d_A, size);
  if (errA != cudaSuccess) {
    printf("%s in %s at line % d\n", cudaGetErrorString(errA), __FILE__,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  // WARN: CUDA API functions return flags that indicate whether an error has
  // occurred when they served the request. Most errors are due to inappropriate
  // argument values used in the call.
  cudaError_t errB = cudaMalloc((void **)&d_B, size);
  if (errA != cudaSuccess) {
    printf("%s in %s at line % d\n", cudaGetErrorString(errB), __FILE__,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaError_t errC = cudaMalloc((void **)&d_C, size);
  if (errA != cudaSuccess) {
    printf("%s in %s at line % d\n", cudaGetErrorString(errC), __FILE__,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  // NOTE: The cudaMemcpy function takes four parameters. The first parameter is
  // a pointer to the destination location for the data object to be copied. The
  // second parameter points to the source location. The third parameter
  // specifies the number of bytes to be copied. The fourth parameter indicates
  // the types of memory involved in the copy: from host to host, from host to
  // device, from device to host, and from device to device.
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // NOTE: When the host code calls a kernel, it sets the grid and thread block
  // dimensions via execution configuration parameters.
  // The first configuration parameter gives the number of blocks in the grid.
  // The second specifies the number of threads in each block.
  vec_add_kernel<<<ceil(n / 256.0), 256>>>(d_A, d_B, d_C, n);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // NOTE: cudaFree does not need to change the value it only needs to use the
  // value of A_d to return the allocated memory back to the available pool.
  // Thus only the value and not the address of A_d is passed as an argument.
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {
  printf("vector addition - CUDA implementation\n");
  printf("Chapter 2: Vector addition\n\n");

  // Simple test with small vectors
  int n = 10;
  float h_A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  float h_B[] = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  float h_C[n];

  printf("Input vectors:\n");
  printf("A: ");
  for (int i = 0; i < n; i++)
    printf("%.1f ", h_A[i]);
  printf("\nB: ");
  for (int i = 0; i < n; i++)
    printf("%.1f ", h_B[i]);
  printf("\n\n");

  // Run parallel vector addition
  par_vecAdd(h_A, h_B, h_C, n);

  printf("Result (A + B):\n");
  printf("C: ");
  for (int i = 0; i < n; i++)
    printf("%.1f ", h_C[i]);
  printf("\n");

  return 0;
}
