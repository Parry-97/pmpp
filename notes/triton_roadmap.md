# Triton Learning Roadmap

## Quick Start: Your First Week

### Day 1-2: Fundamentals

1. Read `triton_fundamentals.md` - understand the mental model shift
2. Study your `vector_addition.py` deeply - it's the "hello world"
3. Modify it: try different BLOCK_SIZE values, add timing

### Day 3-4: Operations

1. Read `triton_operations.md` - understand the API
2. Implement element-wise kernels:
   - ReLU: `output = tl.where(x > 0, x, 0.0)`
   - Sigmoid: `output = 1 / (1 + tl.exp(-x))`
   - GELU: `output = 0.5 * x * (1 + tl.tanh(0.797884 * (x + 0.044715 * x**3)))`

### Day 5-7: 2D Operations

1. Read `triton_patterns.md` - focus on Pattern 4 (2D grid)
2. Implement grayscale conversion in Triton
3. Implement simple 2D operations (transpose, element-wise on matrices)

## Intermediate: Weeks 2-3

### Matrix Multiplication Journey

1. Study Pattern 3 in `triton_patterns.md`
2. Implement naive matmul first (understand the indexing)
3. Add tiling (the optimized version)
4. Understand why `tl.dot` matters

### Fused Operations

1. Implement softmax (Pattern 6)
2. Understand why fusion saves memory bandwidth
3. Try fusing: LayerNorm, attention scores

### Autotuning

1. Read `triton_autotuning.md`
2. Add autotuning to your matmul
3. Compare against cuBLAS

## Advanced: Month 2+

### Flash Attention

- The crown jewel of Triton kernels
- Requires understanding: tiling, online softmax, memory management

### Custom Training Kernels

- Fused Adam optimizer
- Fused dropout + residual + layer norm

### Multi-GPU Considerations

- Understand how Triton interacts with PyTorch distributed

---

## Exercises (Progressive Difficulty)

### Level 1: Warm-up

```python
# Exercise 1.1: Implement vector scaling
# y = alpha * x

# Exercise 1.2: Implement element-wise max
# z = max(x, y)

# Exercise 1.3: Implement ReLU
# y = max(0, x)
```

### Level 2: Reductions

```python
# Exercise 2.1: Sum reduction
# result = sum(x)

# Exercise 2.2: Max reduction with index
# max_val, max_idx = max(x)

# Exercise 2.3: Softmax (row-wise)
# y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
```

### Level 3: 2D Operations

```python
# Exercise 3.1: Matrix transpose
# B = A^T

# Exercise 3.2: Row-wise normalization
# y[i, :] = x[i, :] / sum(x[i, :])

# Exercise 3.3: Matrix-vector multiplication
# y = A @ x
```

### Level 4: Tiled Operations

```python
# Exercise 4.1: Tiled matrix multiplication
# C = A @ B (with tiles!)

# Exercise 4.2: Fused matmul + bias + ReLU
# y = ReLU(A @ B + bias)

# Exercise 4.3: Batch matrix multiplication
# C[b] = A[b] @ B[b]
```

### Level 5: Advanced

```python
# Exercise 5.1: Layer normalization
# y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

# Exercise 5.2: Attention scores
# scores = softmax(Q @ K^T / sqrt(d))

# Exercise 5.3: Fused dropout
# y = dropout(x, p) * scale  # Without materializing mask
```

---

## Quick Reference Card

### Kernel Template

```python
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    input_ptr,                    # Pointer from torch tensor
    output_ptr,
    n_elements,                   # Runtime value
    BLOCK_SIZE: tl.constexpr,     # Compile-time constant
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute
    # Your operation
    
    # Store
    tl.store(output_ptr + offsets, y, mask=mask)

def my_op(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    my_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output
```

### Key Functions Cheat Sheet

```
tl.program_id(axis)     → Which program (like blockIdx)
tl.arange(start, end)   → Vector of indices
tl.load(ptr, mask)      → Load from memory
tl.store(ptr, val, mask)→ Store to memory
tl.dot(a, b)            → Matrix multiply
tl.sum(x, axis)         → Reduce sum
tl.max(x, axis)         → Reduce max
tl.where(cond, a, b)    → Element-wise conditional
tl.zeros(shape, dtype)  → Create zero tensor
x.to(dtype)             → Cast dtype
x[:, None]              → Add dimension (for broadcasting)
```

### Launch Pattern

```python
# 1D
grid = (triton.cdiv(n, BLOCK_SIZE),)
kernel[grid](args..., BLOCK_SIZE=BLOCK_SIZE)

# 2D
grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
kernel[grid](args..., BM=64, BN=64)

# With lambda (for autotuning)
grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']),)
kernel[grid](args...)
```

---

## Resources

### Official

- Triton Tutorials: <https://triton-lang.org/main/getting-started/tutorials/>
- Triton API: <https://triton-lang.org/main/python-api/triton.language.html>
- Triton Programming Guide: <https://triton-lang.org/main/programming-guide/>

### Production Code to Study

- Flash Attention: <https://github.com/Dao-AILab/flash-attention>
- Unsloth: <https://github.com/unslothai/unsloth>
- vLLM: <https://github.com/vllm-project/vllm>
- Liger Kernel: <https://github.com/linkedin/Liger-Kernel>
- Triton Puzzles: <https://github.com/srush/Triton-Puzzles>

### Essential Papers

- Flash Attention (Dao et al., 2022)
- Flash Attention 2 (Dao, 2023)
- Original Triton paper (Tillet et al., 2019)

### Your Learning Files

```
notes/
├── triton_fundamentals.md       ← Start here
├── triton_operations.md         ← API reference  
├── triton_patterns.md           ← CUDA→Triton translation
├── triton_autotuning.md         ← Performance tuning
├── triton_debugging.md          ← When things go wrong
├── triton_roadmap.md            ← This file
├── triton_advanced_internals.md ← Compiler, IR, memory (advanced)
├── triton_advanced_kernels.md   ← Flash Attention, etc. (advanced)
└── triton_mastery_resources.md  ← Full resource guide (advanced)
```
