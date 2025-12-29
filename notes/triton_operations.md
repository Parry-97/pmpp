# Triton Language Operations Reference

## Core Import Pattern

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
```

## Memory Operations

### 1. Loading Data: `tl.load`

```python
# Basic load with mask for bounds checking
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```

**Parameters:**

- `ptr + offsets`: Pointer arithmetic (scalar ptr + vector of offsets = vector of ptrs)
- `mask`: Boolean tensor - only load where True
- `other`: Default value where mask is False (prevents undefined behavior)

**CUDA Equivalent:**

```cuda
// CUDA - per thread
if (i < n) {
    data = ptr[i];
}

// Triton - per block
mask = offsets < n
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### 2. Storing Data: `tl.store`

```python
tl.store(ptr + offsets, value, mask=mask)
```

**Critical Rule:** Without mask, you may write garbage to invalid memory locations.

### 3. Block Pointers (Advanced)

Block pointers are a higher-level abstraction for managing 2D/3D memory access. They simplify tiled access patterns (like in MatMul or Conv) by handling pointer arithmetic, striding, and boundary conditions automatically.

**Why use them?**

- **Simplicity:** No need to manually calculate `offs_am[:, None] * stride_am + offs_k[None, :]`.
- **Safety:** Built-in boundary checks (`boundary_check` parameter).
- **Performance:** Can help the compiler generate more optimal memory access instructions (e.g., `ldmatrix` on NVIDIA GPUs) and reduce register usage.

#### 3.1 Creating a Block Pointer: `tl.make_block_ptr`

```python
block_ptr = tl.make_block_ptr(
    base=ptr,              # Base pointer to the tensor
    shape=(M, N),          # Original shape of the tensor (global dims)
    strides=(stride_m, stride_n), # Strides of the tensor
    offsets=(start_m, start_n),   # Starting offset of the block
    block_shape=(BLOCK_M, BLOCK_N), # Shape of the block to load
    order=(1, 0)           # Order of dimensions in memory (1,0 for row-major/C-contiguous)
)
```

- **`order` parameter**: Specifies which dimension moves fastest in memory.
  - For row-major (C-style) `(M, N)` array, the stride is `(N, 1)`. The columns (dim 1) are contiguous, so `order=(1, 0)`.
  - For column-major (F-style) `(M, N)` array, the stride is `(1, M)`. The rows (dim 0) are contiguous, so `order=(0, 1)`.

#### 3.2 Loading and Storing

**Loading:**

```python

# Load a block. 

# boundary_check=(0, 1) handles masking if the block exceeds (M, N).

# padding_option="zero" (default) fills out-of-bounds with 0.

tile = tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

```

**Storing:**

```python

# Store a block.

tl.store(block_ptr, tile, boundary_check=(0, 1))

```

**Understanding `boundary_check`**:

The `boundary_check` argument is a tuple of integers indicating which dimensions require bounds protection.

- **How it works**: It compares the block's current range against the `shape` provided in `make_block_ptr`.

- **Behavior**:

  - **On Load**: If indices exceed the shape, values are replaced with `padding_option` (default 0).
  - **On Store**: If indices exceed the shape, the write is discarded (masked out).

- **Comparison**:

  - *Manual*: `mask = (rows[:, None] < M) & (cols[None, :] < N)` -> `tl.load(..., mask=mask)`
  - *Block Ptr*: `boundary_check=(0, 1)` (Handling is automatic).

#### 3.3 Advancing the Pointer: `tl.advance`

Instead of creating a new pointer for the next tile, you update the existing one. This is efficient and keeps the code clean.

```python
# Move the block by [BLOCK_M, 0] (e.g., down one block-row)
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
```

#### 3.4 Example: Tiled Matrix Multiplication Loop

```python
# Initialize block pointer for A (M x K)
a_ptr = tl.make_block_ptr(
    base=A, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0)
)

# Initialize block pointer for B (K x N)
b_ptr = tl.make_block_ptr(
    base=B, shape=(K, N), strides=(stride_bk, stride_bn),
    offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N),
    order=(1, 0)
)

accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

for k in range(0, K, BLOCK_K):
    # Load tiles with automatic boundary handling
    # If the matrix dimension isn't a multiple of BLOCK_SIZE, this handles it safely.
    a = tl.load(a_ptr, boundary_check=(0, 1))
    b = tl.load(b_ptr, boundary_check=(0, 1))
    
    # Compute
    accumulator = tl.dot(a, b, accumulator)
    
    # Advance pointers to next K-block
    # A moves horizontally (K dim is dim 1): (0, BLOCK_K)
    # B moves vertically (K dim is dim 0): (BLOCK_K, 0)
    a_ptr = tl.advance(a_ptr, (0, BLOCK_K))
    b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))
```

## Index Generation

### 1. `tl.arange` - The Foundation

```python
# Creates [0, 1, 2, ..., n-1]
indices = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE must be constexpr

# IMPORTANT: Both bounds must be powers of 2 for best performance
# Good:  tl.arange(0, 64), tl.arange(0, 256)
# Works: tl.arange(0, 100) but may be slower
```

### 2. Building 2D Index Grids

```python
# For matrix operations - create row and column indices
BLOCK_M, BLOCK_N = 64, 64

row_idx = tl.arange(0, BLOCK_M)  # Shape: (BLOCK_M,)
col_idx = tl.arange(0, BLOCK_N)  # Shape: (BLOCK_N,)

# Expand dimensions to broadcast
row_idx = row_idx[:, None]  # Shape: (BLOCK_M, 1)
col_idx = col_idx[None, :]  # Shape: (1, BLOCK_N)

# Now they broadcast to create (BLOCK_M, BLOCK_N) grids
# row_idx[:, None] + col_idx[None, :] → (BLOCK_M, BLOCK_N)
```

## Pointer Arithmetic

### The Broadcasting Rule

```python
# Scalar pointer + vector of offsets = vector of pointers
base_ptr = x_ptr                        # Scalar pointer
offsets = tl.arange(0, BLOCK_SIZE)      # Vector [0, 1, ..., BLOCK_SIZE-1]
ptrs = base_ptr + offsets               # Vector of pointers!
```

### 2D Pointer Patterns

```python
# For a row-major matrix with stride `stride_row` between rows:
row_offsets = tl.arange(0, BLOCK_M)[:, None] * stride_row  # (BLOCK_M, 1)
col_offsets = tl.arange(0, BLOCK_N)[None, :]               # (1, BLOCK_N)

# Full offset grid - broadcasts to (BLOCK_M, BLOCK_N)
offsets_2d = row_offsets + col_offsets

# Load a 2D tile
tile = tl.load(matrix_ptr + offsets_2d, mask=mask_2d)
```

## Math Operations

### Element-wise Operations (Just Like NumPy)

```python
# Arithmetic
c = a + b
c = a - b
c = a * b
c = a / b

# Math functions
y = tl.exp(x)
y = tl.log(x)
y = tl.sqrt(x)
y = tl.sin(x)
y = tl.cos(x)
y = tl.abs(x)

# Comparisons (return boolean tensors)
mask = x > 0
mask = x == y
```

### Reduction Operations

```python
# Sum along axis
total = tl.sum(x, axis=0)

# Max/Min
max_val = tl.max(x, axis=0)
min_val = tl.min(x, axis=0)

# With indices
max_idx = tl.argmax(x, axis=0)
```

### Matrix Multiplication: `tl.dot`

```python
# Matrix multiply two tiles
# a: (BLOCK_M, BLOCK_K)
# b: (BLOCK_K, BLOCK_N)
# Result: (BLOCK_M, BLOCK_N)
c = tl.dot(a, b)

# Accumulate into existing tensor (for tiled matmul)
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
accumulator = tl.dot(a, b, accumulator)
```

**Important:** `tl.dot` maps to tensor cores when available - this is where Triton shines!

## Control Flow and Conditionals

### Using `tl.where` (Replaces if-else)

```python
# Element-wise conditional
result = tl.where(condition, value_if_true, value_if_false)

# Example: ReLU activation
output = tl.where(x > 0, x, 0.0)

# Example: Leaky ReLU
output = tl.where(x > 0, x, 0.01 * x)
```

### Loops in Kernels

```python
# Standard Python for-loops work
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    # Load tiles
    a = tl.load(a_ptr + ...)
    b = tl.load(b_ptr + ...)
    # Accumulate
    accumulator = tl.dot(a, b, accumulator)

# Triton-specific range with optimizations
for k in tl.range(0, K, BLOCK_K):
    # tl.range can enable software pipelining
    ...
```

## Type Conversions

```python
# Cast to different dtype
x_fp16 = x.to(tl.float16)
x_fp32 = x.to(tl.float32)
x_int32 = x.to(tl.int32)

# Common pattern: accumulate in fp32, store in fp16
accumulator = tl.zeros((M, N), dtype=tl.float32)  # High precision
# ... computation ...
output = accumulator.to(tl.float16)  # Convert for storage
tl.store(output_ptr + offsets, output, mask=mask)
```

## Creating Tensors

```python
# Zeros
zeros = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

# Full (constant value)
ones = tl.full((BLOCK_M, BLOCK_N), value=1.0, dtype=tl.float32)
```

## Shape Manipulation

```python
# Reshape (total elements must match)
reshaped = tl.reshape(x, (new_dim1, new_dim2))

# Transpose/Permute
transposed = tl.trans(x)  # 2D transpose
permuted = tl.permute(x, (1, 0, 2))  # General permutation

# Expand dims (add size-1 dimension)
expanded = tl.expand_dims(x, axis=0)
# Or use slicing syntax
expanded = x[None, :]  # Add dim at front
expanded = x[:, None]  # Add dim at end

# Broadcast to shape
broadcasted = tl.broadcast_to(x, (M, N))
```

## Atomic Operations

```python
# Atomic add (for reductions across programs)
tl.atomic_add(ptr + offset, value, mask=mask)

# Other atomics
tl.atomic_max(ptr, value)
tl.atomic_min(ptr, value)
tl.atomic_and(ptr, value)
tl.atomic_or(ptr, value)
tl.atomic_xor(ptr, value)
tl.atomic_cas(ptr, compare, value)  # Compare-and-swap
```

## Compiler Hints

```python
# Tell compiler values are contiguous (enables optimizations)
offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)

# Tell compiler about value constraints
tl.assume(stride > 0)
tl.assume(pid >= 0)

# Debug barrier (rarely needed - compiler handles sync)
tl.debug_barrier()
```

## Common Dtype Constants

```python
tl.float16   # fp16
tl.bfloat16  # bf16
tl.float32   # fp32
tl.float64   # fp64
tl.int8
tl.int16
tl.int32
tl.int64
tl.uint8
tl.uint16
tl.uint32
tl.uint64
tl.bool      # For masks
```
