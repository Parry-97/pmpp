# Triton Kernel Patterns

This document shows common GPU kernel patterns, comparing your CUDA implementations to their Triton equivalents.

## Pattern 1: Vector Addition (1D)

### Your CUDA Version

```cuda
__global__ void vec_add_kernel(float *d_A, float *d_B, float *d_C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_C[i] = d_A[i] + d_B[i];
    }
}
// Launch: vec_add_kernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
```

### Triton Equivalent

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

# Launch
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output
```

### Key Differences

| CUDA | Triton |
|------|--------|
| `int i = ...` (scalar) | `offsets = ...` (vector) |
| `if (i < n)` | `mask = offsets < n` |
| One thread → one element | One program → BLOCK_SIZE elements |

---

## Pattern 2: Matrix Multiplication (Naive)

### Your CUDA Version

```cuda
__global__ void matmul_kernel(float *M, float *N, float *P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < width && col < width) {
        float outVal = 0;
        for (int i = 0; i < width; i++) {
            outVal += M[row * width + i] * N[i * width + col];
        }
        P[row * width + col] = outVal;
    }
}
```

### Triton Equivalent (Naive - for understanding)

```python
@triton.jit
def matmul_naive_kernel(
    M_ptr, N_ptr, P_ptr,
    M_rows, M_cols, N_cols,
    stride_mm, stride_mk,
    stride_nk, stride_nn,
    stride_pm, stride_pn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Which output block are we computing?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create index ranges for this block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Row indices
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Col indices
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K dimension (one element at a time - inefficient!)
    for k in range(M_cols):
        # Load one column of M block and one row of N block
        m = tl.load(M_ptr + rm[:, None] * stride_mm + k * stride_mk,
                    mask=rm[:, None] < M_rows, other=0.0)
        n = tl.load(N_ptr + k * stride_nk + rn[None, :] * stride_nn,
                    mask=rn[None, :] < N_cols, other=0.0)
        acc += m * n  # Outer product accumulation
    
    # Store result
    mask = (rm[:, None] < M_rows) & (rn[None, :] < N_cols)
    tl.store(P_ptr + rm[:, None] * stride_pm + rn[None, :] * stride_pn,
             acc, mask=mask)
```

---

## Pattern 3: Tiled Matrix Multiplication (Optimized)

### Your CUDA Version (Tiled)

```cuda
__global__ void tiled_matmul_kernel(float *M, float *N, float *P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        // Collaborative loading
        Mds[ty][tx] = M[row * width + (ph * TILE_WIDTH + tx)];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[row * width + col] = Pvalue;
}
```

### Triton Equivalent (The Real Power)

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program mapping (which output tile?)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Offset calculations
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to first tiles of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulator - USE FLOAT32 FOR PRECISION
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles with boundary checks
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Tile multiplication (maps to tensor cores!)
        accumulator = tl.dot(a, b, accumulator)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert and store
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

### What Triton Does For You (That You Did Manually in CUDA)

| Your CUDA Work | Triton Compiler Handles |
|----------------|-------------------------|
| `__shared__ float Mds[...]` | Automatic shared memory allocation |
| `__syncthreads()` | Automatic synchronization |
| `Mds[ty][tx] = ...` | Optimized tile loading |
| Thread indexing math | Automatic thread mapping |
| Bank conflict avoidance | Automatic swizzling |

---

## Pattern 4: Image Processing (2D Grid)

### Your CUDA Version (Grayscale)

```cuda
__global__ void grayscale_kernel(unsigned char *input, unsigned char *output,
                                 int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int outOffset = row * width + col;
        int inputOffset = outOffset * CHANNELS;
        
        unsigned char red = input[inputOffset];
        unsigned char green = input[inputOffset + 1];
        unsigned char blue = input[inputOffset + 2];
        
        output[outOffset] = 0.21 * red + 0.71 * green + 0.07 * blue;
    }
}
```

### Triton Equivalent

```python
@triton.jit
def grayscale_kernel(
    input_ptr, output_ptr,
    width, height,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    # 2D program grid
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # Generate 2D indices for this block
    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    
    # Create 2D mask
    mask = (offs_x[None, :] < width) & (offs_y[:, None] < height)
    
    # Output pixel positions (row-major)
    out_offsets = offs_y[:, None] * width + offs_x[None, :]
    
    # Input pixel positions (RGB interleaved)
    in_offsets = out_offsets * 3
    
    # Load RGB values (each is a 2D block)
    red = tl.load(input_ptr + in_offsets, mask=mask, other=0).to(tl.float32)
    green = tl.load(input_ptr + in_offsets + 1, mask=mask, other=0).to(tl.float32)
    blue = tl.load(input_ptr + in_offsets + 2, mask=mask, other=0).to(tl.float32)
    
    # Grayscale conversion
    gray = (0.21 * red + 0.71 * green + 0.07 * blue).to(tl.uint8)
    
    # Store result
    tl.store(output_ptr + out_offsets, gray, mask=mask)

# Launch with 2D grid
grid = (triton.cdiv(width, BLOCK_X), triton.cdiv(height, BLOCK_Y))
grayscale_kernel[grid](input_ptr, output_ptr, width, height, BLOCK_X=16, BLOCK_Y=16)
```

---

## Pattern 5: Reduction (Sum)

### CUDA Pattern

```cuda
__global__ void sum_kernel(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(output, sdata[0]);
}
```

### Triton Equivalent

```python
@triton.jit
def sum_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load block of data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Block-level sum (compiler handles the tree reduction)
    block_sum = tl.sum(x, axis=0)
    
    # Atomic add to global output
    tl.atomic_add(output_ptr, block_sum)
```

The magic: `tl.sum()` compiles to an efficient parallel reduction - you don't write the tree!

---

## Pattern 6: Softmax (Fused Operations)

### Why Fusion Matters

Without fusion (PyTorch naive):

```python
# 3 separate kernel launches, 3 global memory round-trips
max_val = x.max(dim=-1, keepdim=True)  # Read x, write max
x_shifted = x - max_val                  # Read x, max; write shifted
exp_x = torch.exp(x_shifted)             # Read shifted, write exp
sum_exp = exp_x.sum(dim=-1, keepdim=True)
output = exp_x / sum_exp                 # Read exp, sum; write output
```

### Triton Fused Softmax

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_start = row_idx * input_stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load entire row
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # Fused operations - all in registers, no global memory round-trips!
    row_max = tl.max(row, axis=0)
    row_shifted = row - row_max
    numerator = tl.exp(row_shifted)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    
    # Single store
    tl.store(output_ptr + row_start + col_offsets, output, mask=mask)
```

**Key Insight:** All intermediate values stay in registers. One global read, one global write. This is the main performance advantage of writing custom Triton kernels.

---

## Kernel Launch Patterns

### Basic Launch

```python
# 1D grid
grid = (triton.cdiv(n, BLOCK_SIZE),)
kernel[grid](args..., BLOCK_SIZE=BLOCK_SIZE)

# 2D grid
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
kernel[grid](args..., BLOCK_M=64, BLOCK_N=64)
```

### Lambda Grid (Dynamic Configuration)

```python
# Grid size depends on constexpr values chosen by autotuner
grid = lambda META: (
    triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
)
kernel[grid](args...)
```

### Wrapper Function Pattern

```python
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Validate inputs
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous()
    
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Configure grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c
```
