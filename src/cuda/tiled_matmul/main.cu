/**
 * @file main.cu
 * @brief Tiled matrix multiplication implementation
 * @author Param Pal Singh
 * @chapter 5
 *
 */

#include <stdio.h>

// NOTE: Size of tiles and typically blocks as well.
// This makes sure each block is in charge of a single tile
#define TILE_WIDTH 16

/**
 * @brief CUDA kernel for tiled_matmul
 *
 * The functions implements tiled_matmul kernel
 *
 * @param input Input data
 * @param output Output data
 * @param n Size of the data
 */
__global__ void tiled_matmul_kernel(float *M, float *N, float *P, int width) {

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  // NOTE: Loop over the M and N tiles required to compute P elements
  float Pvalue = 0;
  for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
    // NOTE: Collaborative loading of M and N tiles into shared memory
    // For Mds we load the row tiles, whereas for Nds we load the column tiles
    // The tiles are like temporary variables that are overwritten at each phase
    // For given output block / tile in P we load and use
    // all the corresponding row and column tiles
    Mds[ty][tx] = M[row * width + (ph * TILE_WIDTH + tx)];
    Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];

    __syncthreads();
    // NOTE: Since we are syncing threads at each phase after each load,
    // all the necessary tiles  for the output block (tile of output P) have
    // been loaded (IN PARALLEL by other threads in the block) (Fig 5.10)
    // Threadwise, we are loading the rows and columns, but considering it
    // blockwise, we are loading the tiles. All in parallel per block.
    //
    // WARN: Each single block/output tile P will load at each phase a tile
    // of M and N. Compute the intermediate value and do the same across the
    // whole matrix N or M width, overwriting the temporary value of the Mds and
    // Nds. So the 1 to 1 mapping from threads to output is the crucial one to
    // remember . A thread uses a whole row and column from the input to compute
    // the output

    // NOTE: Computing of partial results based on tile of data that
    // has been computed in the block
    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  // NOTE: The final Pvalue is accumulated from all the tiles and phases in the
  // loop.
  P[row * width + col] = Pvalue;
}

/**
 * @brief Host wrapper function for tiled_matmul
 *
 * @param h_input Host input data
 * @param h_output Host output data
 * @param n Size of the data
 */
void tiled_matmul(float *h_input, float *h_output, int n) {
  int size = n * sizeof(float);
  float *d_input, *d_output;

  // Allocate device memory
  cudaMalloc((void **)&d_input, size);
  cudaMalloc((void **)&d_output, size);

  // Copy data to device
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  dim3 blockSize;
  dim3 gridSize;

  // Launch kernel
  // tiled_matmul_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

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
  printf("tiled_matmul - CUDA implementation\n");
  printf("Chapter 5: Tiled matrix multiplication implementation\n\n");

  // TODO: Implement test/demo code

  return 0;
}
