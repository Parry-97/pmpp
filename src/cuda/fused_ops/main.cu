/**
 * @file main.cu
 * @brief Demonstration of Fused vs Unfused kernels in CUDA
 *
 * This example compares two approaches to computing: D = (A * B) + C
 * 1. Unfused: Launch Mul kernel -> Global Memory -> Launch Add kernel
 * 2. Fused: Launch MulAdd kernel (intermediate stays in registers)
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// -------------------------------------------------------------------------
// PART 1: UNFUSED KERNELS (Traditional Approach)
// -------------------------------------------------------------------------

/**
 * @brief Kernel 1: Multiply A * B -> Temp
 * Reads A, B from Global Memory. Writes Temp to Global Memory.
 */
__global__ void unfused_mul_kernel(const float* A, const float* B, float* Temp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Temp[i] = A[i] * B[i]; // Global Write
    }
}

/**
 * @brief Kernel 2: Add Temp + C -> D
 * Reads Temp, C from Global Memory. Writes D to Global Memory.
 */
__global__ void unfused_add_kernel(const float* Temp, const float* C, float* D, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        D[i] = Temp[i] + C[i]; // Global Read (Temp) + Global Write (D)
    }
}

// -------------------------------------------------------------------------
// PART 2: FUSED KERNEL (Optimized Approach)
// -------------------------------------------------------------------------

/**
 * @brief Fused Kernel: (A * B) + C -> D
 * Reads A, B, C. Computes result in registers.
 * Writes D.
 * SAVES: 1 Global Write (Temp) + 1 Global Read (Temp)
 */
__global__ void fused_mul_add_kernel(const float* A, const float* B, const float* C, float* D, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // The multiplication result (A[i] * B[i]) lives ONLY in a register
        // It is never written to slow global memory.
        float intermediate = A[i] * B[i]; 
        D[i] = intermediate + C[i];
    }
}

// -------------------------------------------------------------------------
// HOST CODE
// -------------------------------------------------------------------------

void verify_result(float* h_ref, float* h_test, int n, const char* name) {
    for (int i = 0; i < n; i++) {
        if (fabs(h_ref[i] - h_test[i]) > 1e-5) {
            printf("Mismatch in %s at index %d: Ref %f != Test %f\n", name, i, h_ref[i], h_test[i]);
            return;
        }
    }
    printf("Passed: %s matches reference.\n", name);
}

int main() {
    int n = 1 << 20; // 1 million elements
    size_t bytes = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    printf("Running Fused vs Unfused Demo on %d elements...\n", n);

    // Host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_Ref = (float*)malloc(bytes); // CPU result
    float *h_Unfused = (float*)malloc(bytes);
    float *h_Fused = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
        h_C[i] = 3.0f;
        h_Ref[i] = (h_A[i] * h_B[i]) + h_C[i]; // Expected: 5.0
    }

    // Device memory
    float *d_A, *d_B, *d_C, *d_Temp, *d_Out;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_Temp, bytes)); // Only needed for unfused
    CUDA_CHECK(cudaMalloc(&d_Out, bytes));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    // ----------------------------------------------------------------
    // RUN UNFUSED
    // ----------------------------------------------------------------
    unfused_mul_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_Temp, n);
    unfused_add_kernel<<<numBlocks, blockSize>>>(d_Temp, d_C, d_Out, n);
    
    CUDA_CHECK(cudaMemcpy(h_Unfused, d_Out, bytes, cudaMemcpyDeviceToHost));
    verify_result(h_Ref, h_Unfused, n, "Unfused Kernel");

    // ----------------------------------------------------------------
    // RUN FUSED
    // ----------------------------------------------------------------
    // Clear output to be sure
    CUDA_CHECK(cudaMemset(d_Out, 0, bytes));
    
    fused_mul_add_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, d_Out, n);
    
    CUDA_CHECK(cudaMemcpy(h_Fused, d_Out, bytes, cudaMemcpyDeviceToHost));
    verify_result(h_Ref, h_Fused, n, "Fused Kernel");

    printf("\nDifferences:\n");
    printf("1. Unfused used 2 kernel launches + 1 extra Global Memory buffer (Temp).\n");
    printf("2. Fused used 1 kernel launch + intermediate value was kept in registers.\n");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_Temp); cudaFree(d_Out);
    free(h_A); free(h_B); free(h_C); free(h_Ref); free(h_Unfused); free(h_Fused);

    return 0;
}
