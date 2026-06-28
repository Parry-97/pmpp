/**
 * @file main.cu
 * @brief CUDA Thrust Intro
 * @author Param Pal Singh
 * @chapter 3
 *
 */

#include <cstdlib>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

int main() {
  // 1. Allocate a vector on the host (CPU) and fill it with random numbers
  thrust::host_vector<int> h_vec(1 << 20); // ~1 million elements
  thrust::generate(h_vec.begin(), h_vec.end(), rand);

  // 2. Transfer data to the device (GPU VRAM)
  // This implicitly executes a cudaMemcpy under the hood
  thrust::device_vector<int> d_vec = h_vec;

  // 3. Perform a massively parallel sort on the GPU
  thrust::sort(d_vec.begin(), d_vec.end());

  // 4. Transfer the sorted data back to the host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  std::cout << "Successfully sorted " << h_vec.size() << " elements."
            << std::endl;
  return 0;
}
