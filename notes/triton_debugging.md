# Triton Debugging and Common Pitfalls

## Critical Gotchas for CUDA Developers

### 1. Block Sizes MUST Be Powers of Two

```python
# ❌ WRONG - Will fail or be inefficient
offsets = tl.arange(0, 100)

# ✅ CORRECT - Use power of 2, pad with mask
BLOCK_SIZE: tl.constexpr = 128  # Power of 2
offsets = tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
```

The Triton compiler generates vectorized code assuming power-of-2 sizes.

### 2. constexpr Is Mandatory for Shape Values

```python
# ❌ WRONG - Runtime value for shape
def kernel(ptr, n_elements, block_size):
    offsets = tl.arange(0, block_size)  # ERROR!

# ✅ CORRECT - Compile-time constant
def kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)  # OK
```

### 3. Always Use Masks for Loads/Stores

```python
# ❌ DANGEROUS - Out-of-bounds access
x = tl.load(ptr + offsets)  # May read garbage or crash

# ✅ SAFE - With mask
mask = offsets < n_elements
x = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### 4. `other` Parameter Matters for Reductions

```python
# ❌ PROBLEMATIC - Default `other` may be wrong
max_val = tl.max(tl.load(ptr + offsets, mask=mask))  # other=0 by default

# ✅ CORRECT - Use appropriate default
max_val = tl.max(tl.load(ptr + offsets, mask=mask, other=-float('inf')))
min_val = tl.min(tl.load(ptr + offsets, mask=mask, other=float('inf')))
sum_val = tl.sum(tl.load(ptr + offsets, mask=mask, other=0.0))
```

### 5. Broadcasting Rules Can Surprise You

```python
# Creating 2D indices - be explicit about dimensions
row_idx = tl.arange(0, BLOCK_M)  # Shape: (BLOCK_M,)
col_idx = tl.arange(0, BLOCK_N)  # Shape: (BLOCK_N,)

# ❌ WRONG - This doesn't create a 2D grid
offsets = row_idx * stride + col_idx  # Shape: (BLOCK_M,) or error

# ✅ CORRECT - Explicit broadcasting
offsets = row_idx[:, None] * stride + col_idx[None, :]  # Shape: (BLOCK_M, BLOCK_N)
```

### 6. Accumulate in Higher Precision

```python
# ❌ RISKY - Accumulating in fp16 loses precision
acc = tl.zeros((M, N), dtype=tl.float16)
for k in range(K):
    acc += tl.dot(a, b)  # fp16 accumulation

# ✅ CORRECT - Accumulate in fp32, convert at end
acc = tl.zeros((M, N), dtype=tl.float32)
for k in range(K):
    acc = tl.dot(a, b, acc)  # fp32 accumulation
output = acc.to(tl.float16)  # Convert for storage
```

### 7. Pointer Types Are Strict

```python
# Triton pointers are typed - must match tensor dtype
# torch.float32 tensor → reads as float32
# torch.float16 tensor → reads as float16

# If you need to reinterpret:
# Use .to() AFTER loading, not during
x = tl.load(ptr + offsets, mask=mask).to(tl.float32)
```

## Debugging Techniques

### 1. Print Debugging (Development Only)

```python
@triton.jit
def debug_kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Print from first program only
    if pid == 0:
        tl.device_print("Starting kernel")
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(ptr + offsets, mask=offsets < n)
        tl.device_print("Loaded values, first =", x[0])
```

**Warning:** `device_print` is slow and should be removed for production.

### 2. Dump Intermediate Values to Memory

```python
@triton.jit
def kernel_with_debug(
    input_ptr, output_ptr, debug_ptr,  # Add debug buffer
    n, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    x = tl.load(input_ptr + offsets, mask=offsets < n)
    
    # Dump intermediate value
    intermediate = x * 2
    tl.store(debug_ptr + offsets, intermediate, mask=offsets < n)
    
    output = intermediate + 1
    tl.store(output_ptr + offsets, output, mask=offsets < n)

# In Python
debug_buffer = torch.empty_like(input_tensor)
kernel_with_debug[grid](input, output, debug_buffer, n, BLOCK_SIZE=1024)
print(debug_buffer)  # Inspect intermediate values
```

### 3. Test with Small Inputs First

```python
# Start with tiny sizes where you can verify by hand
def test_kernel():
    # Use sizes small enough to manually verify
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    y = torch.tensor([4.0, 3.0, 2.0, 1.0], device='cuda')
    
    output = torch.empty_like(x)
    add_kernel[(1,)](x, y, output, 4, BLOCK_SIZE=4)
    
    expected = torch.tensor([5.0, 5.0, 5.0, 5.0], device='cuda')
    assert torch.allclose(output, expected)
```

### 4. Compare Against PyTorch Reference

```python
def test_matmul():
    M, N, K = 32, 32, 32  # Small sizes
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    
    # Reference
    expected = torch.matmul(a, b)
    
    # Your kernel
    actual = triton_matmul(a, b)
    
    # Compare with tolerance
    if not torch.allclose(actual, expected, rtol=1e-3, atol=1e-3):
        diff = (actual - expected).abs()
        print(f"Max diff: {diff.max().item()}")
        print(f"Mean diff: {diff.mean().item()}")
        print(f"Where: {(diff > 1e-3).nonzero()}")
```

### 5. Check Grid Configuration

```python
def launch_with_debug(x, y, output, n, BLOCK_SIZE):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    print(f"n={n}, BLOCK_SIZE={BLOCK_SIZE}, grid={grid}")
    print(f"Total elements covered: {grid[0] * BLOCK_SIZE}")
    
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
```

## Common Error Messages

### "incompatible shape"
```
Cause: Tensor shapes don't broadcast correctly
Fix: Check your [:, None] and [None, :] patterns
```

### "constexpr required"
```
Cause: Using runtime value where compile-time constant needed
Fix: Add tl.constexpr type hint to the parameter
```

### "CUDA error: invalid configuration"
```
Cause: Grid size too large or block config invalid
Fix: Check grid calculation, ensure it's > 0
```

### "out of resources"
```
Cause: Too many registers/shared memory requested
Fix: Reduce BLOCK_SIZE, reduce num_warps
```

## Validation Checklist

Before declaring your kernel "done":

- [ ] Test with n not divisible by BLOCK_SIZE
- [ ] Test with n smaller than BLOCK_SIZE
- [ ] Test with n = 0
- [ ] Test with large n (millions of elements)
- [ ] Compare numerical output with reference implementation
- [ ] Run multiple times to check for race conditions
- [ ] Test with different dtypes (float16, float32, bfloat16)
- [ ] Benchmark against PyTorch/cuBLAS baseline

## Triton Environment Variables for Debugging

```bash
# Show compiled PTX/SASS
export TRITON_PRINT_AUTOTUNING=1

# Disable cache (for debugging)
export TRITON_CACHE_DIR=""

# Verbose compilation
export MLIR_ENABLE_DUMP=1
```

## Asserting in Kernels

```python
@triton.jit
def safe_kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Runtime assertion
    tl.device_assert(n > 0, "n must be positive")
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.device_assert(tl.max(offsets) < 1000000, "offset overflow")
```
