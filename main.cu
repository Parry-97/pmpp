#include <stdio.h>

__global__ void hello_from_gpu() { printf("Hello from the GPU!\n"); }

int main() {
  printf("Hello from the CPU!\n");
  hello_from_gpu<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
