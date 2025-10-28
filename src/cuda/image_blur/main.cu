/**
 * @file main.cu
 * @brief Image Blur kernel implementation
 * @author Param Pal Singh
 * @chapter 3
 *
 */

#include <stdio.h>

/** @brief Blur kernel radius (box blur filter size) */
const int BLUR_SIZE = 1;

/**
 * @brief CUDA kernel for image blur
 *
 * Applies box blur filter to grayscale image.
 *
 * @param input Grayscale input image
 * @param output Blurred output image
 * @param width Image width in pixels
 * @param height Image height in pixels
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
 * @brief Host wrapper for image blur
 *
 * @param h_input Grayscale input image on host
 * @param h_output Blurred output image on host
 * @param width Image width in pixels
 * @param height Image height in pixels
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

  // Simple 4x4 grayscale image test
  int width = 4;
  int height = 4;

  // Grayscale image with a bright center
  unsigned char h_input[] = {10, 20,  30,  40, 50, 255, 255, 60,
                             70, 255, 255, 80, 90, 100, 110, 120};

  unsigned char h_output[width * height];

  printf("Input image (4x4):\n");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%3d ", h_input[i * width + j]);
    }
    printf("\n");
  }

  // Run blur operation
  image_blur(h_input, h_output, width, height);

  printf("\nBlurred image:\n");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%3d ", h_output[i * width + j]);
    }
    printf("\n");
  }

  return 0;
}
