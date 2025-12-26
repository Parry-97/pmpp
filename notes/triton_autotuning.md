# Triton Autotuning and Performance

## What is Autotuning?

Unlike CUDA where you manually tune block sizes, Triton can automatically find optimal configurations by testing multiple options and measuring performance.

## Basic Autotuning Setup

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],  # Re-tune when these values change
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Kernel implementation...
```

## triton.Config Parameters

```python
triton.Config(
    # Meta-parameters (become constexpr in kernel)
    kwargs={
        'BLOCK_SIZE_M': 128,
        'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64,
        'GROUP_SIZE_M': 8,
    },
    
    # Compilation options
    num_warps=8,      # Number of warps per SM (affects parallelism)
    num_stages=3,     # Pipeline stages (affects memory latency hiding)
    
    # Optional: Pre-hook for setup
    pre_hook=None,
)
```

### num_warps
- Controls how many warps (groups of 32 threads) execute the kernel
- More warps = more parallelism but more register pressure
- Typical values: 2, 4, 8, 16
- Rule of thumb: Larger block sizes → more warps

### num_stages
- Controls software pipelining depth for memory operations
- More stages = better latency hiding but more register/shared memory usage
- Typical values: 2, 3, 4, 5
- Rule of thumb: Memory-bound kernels benefit from more stages

## The `key` Parameter

```python
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K'],  # These are kernel arguments
)
```

The autotuner caches the best configuration for each unique combination of key values:
- First call with M=1024, N=1024, K=512 → runs all configs, caches best
- Second call with same values → uses cached config
- Call with M=2048, N=1024, K=512 → runs all configs again (different M)

## Configuration Strategies

### For Matrix Multiplication (Compute-Bound)
```python
def get_cuda_autotune_config():
    return [
        # Large blocks for large matrices
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, 
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, 
                      num_stages=3, num_warps=8),
        
        # Medium blocks
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
                      num_stages=4, num_warps=4),
        
        # Small blocks for small matrices
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
                      num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, 
                      num_stages=5, num_warps=2),
    ]
```

### For Memory-Bound Kernels (Element-wise, Reductions)
```python
def get_elementwise_config():
    return [
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
    ]
```

## L2 Cache Optimization: GROUP_SIZE_M

For matmul, programs computing nearby output tiles should run together to share L2 cache:

```python
@triton.jit
def matmul_kernel(..., GROUP_SIZE_M: tl.constexpr):
    # Map 1D program ID to 2D tile coordinates with grouping
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouped ordering for L2 cache reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

**Visual: Without vs With Grouping**
```
Without GROUP_SIZE_M (row-major):     With GROUP_SIZE_M=4:
Program order visits output tiles:    Programs visit tiles in groups:
┌───┬───┬───┬───┐                     ┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ → far apart        │ 0 │ 4 │ 8 │12 │
├───┼───┼───┼───┤   in B matrix      ├───┼───┼───┼───┤  
│ 4 │ 5 │ 6 │ 7 │                     │ 1 │ 5 │ 9 │13 │ → nearby tiles
├───┼───┼───┼───┤                     ├───┼───┼───┼───┤   share B data
│ 8 │ 9 │10 │11 │                     │ 2 │ 6 │10 │14 │   in L2 cache
├───┼───┼───┼───┤                     ├───┼───┼───┼───┤
│12 │13 │14 │15 │                     │ 3 │ 7 │11 │15 │
└───┴───┴───┴───┘                     └───┴───┴───┴───┘
```

## Benchmarking and Profiling

### Built-in Benchmarking

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as x-axis
        x_vals=[128 * i for i in range(2, 33)],  # Values for x-axis
        line_arg='provider',  # Argument name to use for different lines
        line_vals=['triton', 'torch'],  # Values for different lines
        line_names=['Triton', 'PyTorch'],  # Legend labels
        styles=[('green', '-'), ('blue', '-')],  # Line styles
        ylabel='TFLOPS',  # y-axis label
        plot_name='matmul-performance',  # Plot filename
        args={},  # Extra arguments
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b),
            quantiles=quantiles
        )
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b),
            quantiles=quantiles
        )
    
    # Compute TFLOPS
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

# Run benchmark
benchmark.run(show_plots=True, print_data=True)
```

### Manual Benchmarking

```python
import triton.testing

# Simple timing
ms = triton.testing.do_bench(lambda: my_kernel[grid](args...))
print(f"Kernel time: {ms:.3f} ms")

# With percentiles
ms, min_ms, max_ms = triton.testing.do_bench(
    lambda: my_kernel[grid](args...),
    quantiles=[0.5, 0.2, 0.8]
)
```

## Common Performance Issues

### 1. Block Size Too Large
**Symptom:** Low occupancy, register spilling
**Solution:** Reduce BLOCK_SIZE, reduce num_warps

### 2. Block Size Too Small
**Symptom:** Not enough work per program, poor arithmetic intensity
**Solution:** Increase BLOCK_SIZE

### 3. Non-Power-of-2 Block Sizes
**Symptom:** Suboptimal vectorization
**Solution:** Use powers of 2 (32, 64, 128, 256)

### 4. Memory Access Not Coalesced
**Symptom:** Low memory throughput
**Solution:** Ensure consecutive threads access consecutive memory

```python
# Bad: Strided access
offsets = tl.arange(0, BLOCK_SIZE) * stride  # Non-contiguous

# Good: Coalesced access
offsets = tl.arange(0, BLOCK_SIZE)  # Contiguous
data = tl.load(ptr + offsets)
```

### 5. Not Using Hints

```python
# Without hints
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

# With hints (helps compiler optimize)
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_am = tl.where(offs_am < M, offs_am, 0)
offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)

# Also add assumptions about strides
tl.assume(stride_am > 0)
tl.assume(stride_ak > 0)
```

## Quick Tuning Checklist

1. **Start with reasonable defaults:**
   - BLOCK_SIZE: 64-256 (power of 2)
   - num_warps: 4
   - num_stages: 2

2. **Test matrix sizes relevant to your use case**

3. **Add configurations progressively:**
   - Start with 3-5 configs
   - Add more based on benchmark results

4. **Profile to identify bottlenecks:**
   - Compute-bound: Increase BLOCK_SIZE, more warps
   - Memory-bound: More stages, optimize access patterns

5. **Use `key` parameter wisely:**
   - Include dimensions that significantly affect performance
   - Don't include dimensions that don't matter
