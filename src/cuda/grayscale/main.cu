/**
 * @file main.cu
 * @brief RGB to grayscale image conversion
 * @author pops
 * @chapter 3
 *
 * Part of the PMPP learning repository
 */

#include <stdio.h>

/** @brief Number of color channels in RGB image */
const int CHANNELS = 3;

/**
 * @brief CUDA kernel for RGB to grayscale conversion
 *
 * Converts RGB image to grayscale using weighted average.
 *
 * @param input RGB input image (3 bytes per pixel)
 * @param output Grayscale output image (1 byte per pixel)
 * @param width Image width in pixels
 * @param height Image height in pixels
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
 * @brief Host wrapper for grayscale conversion
 *
 * @param h_input RGB input image on host
 * @param h_output Grayscale output image on host
 * @param width Image width in pixels
 * @param height Image height in pixels
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

  // Simple 2x2 RGB image test (4 pixels)
  int width = 2;
  int height = 2;

  // RGB image: Red, Green, Blue, White pixels
  unsigned char h_input[] = {
      255, 0,   0,   // Red pixel
      0,   255, 0,   // Green pixel
      0,   0,   255, // Blue pixel
      255, 255, 255  // White pixel
  };

  unsigned char h_output[width * height];

  printf("Input RGB image (2x2):\n");
  printf("Pixel (0,0): R=%d G=%d B=%d\n", h_input[0], h_input[1], h_input[2]);
  printf("Pixel (0,1): R=%d G=%d B=%d\n", h_input[3], h_input[4], h_input[5]);
  printf("Pixel (1,0): R=%d G=%d B=%d\n", h_input[6], h_input[7], h_input[8]);
  printf("Pixel (1,1): R=%d G=%d B=%d\n\n", h_input[9], h_input[10],
         h_input[11]);

  // Run grayscale conversion
  grayscale(h_input, h_output, width, height);

  printf("Grayscale output:\n");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%3d ", h_output[i * width + j]);
    }
    printf("\n");
  }

  return 0;
}
