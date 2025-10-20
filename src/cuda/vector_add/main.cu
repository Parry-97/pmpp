#include <stdio.h>

__global__ void hello_from_gpu() { printf("Hello from the GPU!\n"); }

/**
 * @brief Sequential vector addition
 *
 * This functions adds two vector in a sequential fashion.
 *
 * @param h_A The first vector
 * @param h_B The second vector
 * @param h_C The result vector
 * @param n The length of the vectors
 * */

void seq_vec_add(float *h_A, float *h_B, float *h_C, int n) {
  for (int i = 0; i < n; i++) {
    h_C[i] = h_A[i] + h_B[i];
  }
}

void vecAdd(float *h_A, float *h_B, float *h_C, int n) {
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
  cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyHostToDevice);

  // NOTE: cudaFree does not need to change the value it only needs to use the
  // value of A_d to return the allocated memory back to the available pool.
  // Thus only the value and not the address of A_d is passed as an argument.
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {
  printf("Hello from the CPU!\n");
  hello_from_gpu<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
