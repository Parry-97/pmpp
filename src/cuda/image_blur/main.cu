/**
 * @file main.cu
 * @brief Image Blur kernel implementation
 * @author Param Pal Singh
 * @chapter 3
 *
 */

#include <stdio.h>

const int BLUR_SIZE = 1;

/**
 * @brief CUDA kernel for image_blur
 *
 * TODO: Implement image_blur kernel
 *
 * @param input Input data
 * @param output Output data
 * @param n Size of the data
 */
__global__ void image_blur_kernel(unsigned char *input, unsigned char *output,
                                  int width, int height) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    int num_pixels = 0;
    int pixel_sum = 0;

    for (int i = row - BLUR_SIZE; i <= row + BLUR_SIZE; i++) {
      for (int j = col - BLUR_SIZE; j <= col + BLUR_SIZE; j++) {
        if (i >= 0 && i < width && j >= 0 && j < height) {
          pixel_sum += input[i * width + j];
          num_pixels++;
        }
      }
    }
    output[row * width + col] = (unsigned char)(pixel_sum / num_pixels);
  }
}

/**
 * @brief Host wrapper function for image_blur
 *
 * @param h_input Host input data
 * @param h_output Host output data
 * @param n Size of the data
 */
void image_blur(unsigned char *h_input, unsigned char *h_output, int width,
                int height) {
  int size = width * height * sizeof(unsigned char);
  unsigned char *d_input, *d_output;

  // Allocate device memory
  cudaMalloc((void **)&d_input, size);
  cudaMalloc((void **)&d_output, size);

  // Copy data to device
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  // NOTE: We are gonna assume the same block dimensions for grayscale are
  // used
  dim3 blockSize = dim3(16, 16);
  dim3 gridSize = dim3(ceil(width / 16.0), ceil(height / 16.0));
  // Launch kernel
  image_blur_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

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
  printf("image_blur - CUDA implementation\n");
  printf("Chapter 3: Image Blur kernel implementation\n\n");

  // TODO: Implement test/demo code

  return 0;
}
