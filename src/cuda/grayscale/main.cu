/**
 * @file main.cu
 * @brief RGB to grayscale image conversion
 * @author pops
 * @chapter 3
 *
 * Part of the PMPP learning repository
 */

#include <stdio.h>

const int CHANNELS = 3;

/**
 * @brief CUDA kernel for grayscale
 *
 * This kernel Implements grayscale transformation on a pixel
 *
 * @param input encoded image of unsigned chars
 * @param output Output image
 * @param height height of the image
 * @param width width of the image
 */
__global__ void grayscale_kernel(unsigned char *input, unsigned char *output,
                                 int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int outOffset = row * width + col;

    int inputOffset = outOffset * CHANNELS;

    // NOTE: Each pixel is 3 consecutive chars for the 3 channels
    // We read the r, g, and b value from the three consecutive byte locations
    // of the input array
    unsigned char red = input[inputOffset];
    unsigned char green = input[inputOffset + 1];
    unsigned char blue = input[inputOffset + 2];

    output[outOffset] = 0.21 * red + 0.71 * green + 0.07 * blue;
  }
}

/**
 * @brief Host wrapper function for grayscale
 *
 * @param h_input Host input data
 * @param h_output Host output data
 * @param n Size of the data
 */
void grayscale(unsigned char *h_input, unsigned char *h_output, int width,
               int height) {
  int size = width * height * sizeof(unsigned char);
  unsigned char *d_input, *d_output;

  // Allocate device memory
  cudaMalloc((void **)&d_input, size);
  cudaMalloc((void **)&d_output, size);

  // Copy data to device
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 blockSize = dim3(16, 16);
  dim3 gridSize = dim3(ceil(width / 16.0), ceil(height / 16.0));
  grayscale_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

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
  printf("grayscale - CUDA implementation\n");
  printf("Chapter 3: RGB to grayscale image conversion\n\n");

  // TODO: Implement test/demo code
}
