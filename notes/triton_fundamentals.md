# Triton Fundamentals

## The Core Mental Model Shift: From Threads to Programs

### CUDA Mental Model (What You Know)

In CUDA, you think about **individual threads**:

- Each thread has a unique ID (`threadIdx`, `blockIdx`)
- Each thread processes one (or a few) elements
- You manage explicit parallelism at thread granularity
- Shared memory synchronization with `__syncthreads()`

```cpp
// CUDA: One thread → One element
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Single index
    if (i < n) {
        c[i] = a[i] + b[i];  // One element per thread
    }
}
```

### Triton Mental Model (What You Need to Learn)

In Triton, you think about **programs operating on blocks of data**:

- Each "program" is like a CUDA thread block, but you program it as a single unit
- Programs process **blocks/tiles** of elements at once
- Operations are implicitly parallel (vectorized)
- The compiler handles thread-level details, shared memory, synchronization

```python
# Triton: One program → One block of elements
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                      # Which program (like blockIdx)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Vector of indices!
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)          # Load BLOCK_SIZE elements
    y = tl.load(y_ptr + offsets, mask=mask)          # Load BLOCK_SIZE elements
    tl.store(output_ptr + offsets, x + y, mask=mask) # Store BLOCK_SIZE elements
```

## Key Concept: Block-Level Thinking

```
CUDA Perspective:                    Triton Perspective:
┌─────────────────────────┐         ┌─────────────────────────┐
│ Grid of Thread Blocks   │         │ Grid of Programs        │
├─────────────────────────┤         ├─────────────────────────┤
│ Block 0 │ Block 1 │ ... │         │ Prog 0 │ Prog 1 │ ...   │
├─────────┼─────────┼─────┤         ├────────┼────────┼───────┤
│ 256     │ 256     │     │         │ Operates on             │
│ threads │ threads │     │         │ BLOCK_SIZE elements     │
│ each    │ each    │     │         │ as a unit               │
└─────────┴─────────┴─────┘         └────────┴────────┴───────┘
```

The Triton compiler decides:

- How many actual GPU threads to use
- How to map your block operations to SIMT execution
- When to use shared memory
- How to optimize memory access patterns

## The Program ID Hierarchy

Just like CUDA has `blockIdx.x/y/z`, Triton has:

```python
pid_x = tl.program_id(axis=0)  # Equivalent to blockIdx.x
pid_y = tl.program_id(axis=1)  # Equivalent to blockIdx.y
pid_z = tl.program_id(axis=2)  # Equivalent to blockIdx.z
```

And just like CUDA's grid dimensions, Triton's grid is specified at launch:

```python
# CUDA launch:    kernel<<<grid_dim, block_dim>>>()
# Triton launch:  kernel[grid]()

grid = (triton.cdiv(n, BLOCK_SIZE),)  # Number of programs
add_kernel[grid](x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE=1024)
```

## Critical Difference: No Thread-Level Indexing

In CUDA, you compute a single index per thread:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

In Triton, you create a **vector of indices** for the entire block:

```python
# tl.arange creates a vector: [0, 1, 2, ..., BLOCK_SIZE-1]
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# Result: [pid*BLOCK_SIZE, pid*BLOCK_SIZE+1, ..., pid*BLOCK_SIZE+BLOCK_SIZE-1]
```

This is why Triton code looks like NumPy - you're operating on arrays, not scalars.

## The constexpr Requirement

Notice `BLOCK_SIZE: tl.constexpr` - this is **mandatory** for shape values:

```python
def add_kernel(
    x_ptr: tl.pointer_type,
    n_elements: int,              # Runtime value - can vary per call
    BLOCK_SIZE: tl.constexpr,     # Compile-time constant - affects code generation
):
    # tl.arange needs compile-time known bounds
    offsets = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE must be constexpr
```

Why? The compiler needs to know tensor shapes at compile time to:

- Generate efficient vectorized code
- Determine register allocation
- Optimize memory access patterns

## Summary: Your Translation Dictionary

| CUDA Concept | Triton Equivalent |
|--------------|-------------------|
| `__global__` function | `@triton.jit` decorated function |
| `blockIdx.x` | `tl.program_id(axis=0)` |
| `threadIdx.x` | Implicit (handled by compiler) |
| `blockDim.x` | `BLOCK_SIZE` (constexpr parameter) |
| Single element access | Block of elements via `tl.arange` |
| `if (i < n)` | `mask = offsets < n` |
| `__shared__` | Automatic or explicit via `tl.static_alloc` |
| `__syncthreads()` | `tl.debug_barrier()` (rarely needed) |
| `<<<grid, block>>>` | `kernel[grid]()` |
