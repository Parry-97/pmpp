/**
 * @file main.cu
 * @brief CUDA Programmatic Kernel Launch
 * @author Param Pal Singh
 * @chapter 3
 *
 */

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
 * @brief CUDA kernel for programmatic_launch
 *
 * TODO: Implement programmatic_launch kernel
 *
 * @param input Input data
 * @param output Output data
 * @param n Size of the data
 */
__global__ void primary_kernel() {
  // Initial work that should finish before starting secondary kernel
  // Trigger the secondary kernel
  cudaTriggerProgrammaticLaunchCompletion();
}

__global__ void secondary_kernel() {
  // Initialization, Independent work, etc.
  // Will block until all primary kernels the secondary kernel is dependent on
  // have completed and flushed results to global memory
  cudaGridDependencySynchronize();
  // Dependent work
}

void print_array(float *v, int n, int limit = 5) {
  for (int i = 0; i < n && i < limit; i++) {
    std::cout << v[i] << " ";
  }
  if (n > limit)
    std::cout << "... (and " << (n - limit) << " more)";
}

/**
 * @brief Host wrapper function for programmatic_launch
 *
 * @param h_input Host input data
 * @param h_output Host output data
 * @param n Size of the data
 */
void programmatic_launch() {
  // int size = n * sizeof(float);
  // float *d_input, *d_output;

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  dim3 blockDim;
  dim3 gridDim;

  // Launch kernel
  // Set up the attribute
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;

  // Set the attribute in a kernel launch configuration
  cudaLaunchConfig_t config = {0};

  // Base launch configuration
  config.gridDim = gridDim;
  config.blockDim = blockDim;
  config.dynamicSmemBytes = 0;

  // Add Special attribute for PDL
  config.attrs = attribute;
  config.numAttrs = 1;

  // Launch primary kernel
  primary_kernel<<<gridDim, blockDim, 0, stream>>>();

  // Launch secondary (dependent) kernel using the configuration with the
  // attribute
  cudaLaunchKernelEx(&config, secondary_kernel);
}

/**
 * @brief Main function
 */
int main() {
  printf("programmatic_launch - CUDA implementation\n");
  printf("Chapter 3: CUDA Programmatic Kernel Launch\n\n");

  // TODO: Implement test/demo code
  programmatic_launch();

  return 0;
}
