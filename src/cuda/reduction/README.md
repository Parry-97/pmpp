# Parallel Reduction in CUDA

This directory contains implementations of the Parallel Reduction algorithm, focusing on the transition from interleaved addressing to convergent addressing to eliminate control divergence.

## Key Concepts

### Thread-to-Element Ratio
In these implementations, the kernel is designed to reduce **2N elements** using **N threads**.
- The first step of the reduction folds the second half of the data array onto the first half.
- **Example:** To reduce an array of 1024 elements, you should launch the kernel with a block size of 512 threads (`blockDim.x = 512`).

### Simple vs. Convergent Kernels
1. **Simple Reduction:** Uses interleaved addressing (`threadIdx.x % stride == 0`). This causes high **Control Divergence** because active and inactive threads are mixed within the same warp.
2. **Convergent Reduction:** Uses contiguous addressing (`threadIdx.x < stride`). This ensures that active threads are packed into the same warps, allowing the hardware to skip inactive warps entirely.

## Constraints
- **Single Block Limitation:** The current implementation uses `threadIdx.x` as the global index. This means the kernel only works for a single block of threads.
- **Max Data Size:** Since the maximum threads per block is typically 1024, this kernel can reduce a maximum of **2048 elements**.
- **Power of Two:** The implementation assumes the input size is a power of two.
