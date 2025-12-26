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

For strided/2D access patterns:

```python
# Create a block pointer to a tile in a matrix
block_ptr = tl.make_block_ptr(
    base=matrix_ptr,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(row_offset, col_offset),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0)  # Memory layout order
)

# Load the entire block
tile = tl.load(block_ptr, boundary_check=(0, 1))

# Advance the pointer
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
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
