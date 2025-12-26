# Triton Advanced Internals

## Understanding the Compilation Pipeline

```
Python Code (@triton.jit)
        ↓
    Triton IR (ttir)
        ↓
    Triton GPU IR (ttgir)
        ↓
    LLVM IR
        ↓
    PTX (NVIDIA) / AMDGPU IR (AMD)
        ↓
    SASS / Machine Code
```

### Inspecting Intermediate Representations

```python
@triton.jit
def my_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, x * 2)

# Get compiled kernel info
compiled = my_kernel.warmup(
    torch.empty(1024, device='cuda'),
    torch.empty(1024, device='cuda'),
    BLOCK=1024,
    grid=(1,)
)

# Access different IR stages
print(compiled.asm['ttir'])   # Triton IR
print(compiled.asm['ttgir'])  # Triton GPU IR
print(compiled.asm['llir'])   # LLVM IR
print(compiled.asm['ptx'])    # PTX assembly
print(compiled.asm['cubin'])  # Binary (hex)
```

### Reading Triton IR (TTIR)

```mlir
// Example Triton IR for vector load-store
module {
  tt.func @my_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1024 = arith.constant 1024 : i32
    
    // Create range [0, 1024)
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    
    // Pointer arithmetic
    %1 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    
    // Load
    %3 = tt.load %2 : tensor<1024xf32>
    
    // Compute
    %cst = arith.constant dense<2.0> : tensor<1024xf32>
    %4 = arith.mulf %3, %cst : tensor<1024xf32>
    
    // Store
    %5 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %6, %4 : tensor<1024xf32>
    
    tt.return
  }
}
```

### Key IR Operations to Understand

| TTIR Op | Meaning |
|---------|---------|
| `tt.make_range` | Creates `tl.arange()` |
| `tt.splat` | Broadcasts scalar to tensor |
| `tt.addptr` | Pointer arithmetic |
| `tt.load` / `tt.store` | Memory operations |
| `tt.dot` | Matrix multiplication |
| `tt.reduce` | Reduction operations |
| `tt.broadcast` | Shape broadcasting |

## Memory Hierarchy Deep Dive

### GPU Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                      Global Memory (HBM)                     │
│                    ~80GB, ~2TB/s bandwidth                   │
│           Accessed via: tl.load(), tl.store()                │
└─────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                        L2 Cache                              │
│                    ~40MB, shared across SMs                  │
│              Automatic, but GROUP_SIZE_M helps               │
└─────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                     L1 Cache / Shared Memory                 │
│              ~128-228KB per SM, ~19TB/s bandwidth            │
│         Triton manages automatically (or tl.static_alloc)    │
└─────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                       Register File                          │
│               ~256KB per SM, ~78TB/s bandwidth               │
│           Where your tl.zeros(), accumulator live            │
└─────────────────────────────────────────────────────────────┘
```

### Explicit Shared Memory Control

```python
@triton.jit
def kernel_with_explicit_smem(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Explicit shared memory allocation (advanced)
    # Usually not needed - compiler handles this
    
    # For async copy (Ampere+), use tensor descriptors
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Load using TMA (Tensor Memory Accelerator) on Hopper
    a_tile = a_desc.load([pid_m * BLOCK_M, 0])
```

### Software Pipelining

```python
@triton.jit
def pipelined_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # num_stages controls pipelining depth
    # With num_stages=3:
    # - Stage 0: Load tile K=0
    # - Stage 1: Load tile K=1, Compute tile K=0
    # - Stage 2: Load tile K=2, Compute tile K=1
    # - Continue: overlap load and compute
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # tl.range enables software pipelining
    for k in tl.range(0, K // BLOCK_K, num_stages=3):
        # Compiler overlaps these loads with previous iteration's compute
        a = tl.load(a_ptr + ...)
        b = tl.load(b_ptr + ...)
        acc = tl.dot(a, b, acc)
    
    tl.store(c_ptr + ..., acc)
```

## Tensor Core Utilization

### When Does Triton Use Tensor Cores?

`tl.dot()` maps to tensor cores when:
1. Input types are fp16, bf16, fp8, or int8
2. Block dimensions are multiples of 16 (for fp16/bf16) or 32 (for int8)
3. Accumulator is fp32 or matching type

```python
# ✅ Uses Tensor Cores (Ampere/Hopper)
a = tl.load(a_ptr, ...).to(tl.float16)  # fp16 input
b = tl.load(b_ptr, ...).to(tl.float16)  # fp16 input
acc = tl.zeros((64, 64), dtype=tl.float32)  # fp32 accumulator
acc = tl.dot(a, b, acc)  # → Tensor Core MMA instructions

# ❌ Falls back to CUDA cores
a = tl.load(a_ptr, ...)  # fp32 input
b = tl.load(b_ptr, ...)  # fp32 input
c = tl.dot(a, b)  # → Regular FMA instructions
```

### Tensor Core Shapes

| GPU Generation | Supported Shapes (M×N×K) | Types |
|----------------|--------------------------|-------|
| Volta (V100) | 16×16×16 | fp16 |
| Turing | 16×16×16, 8×8×16 | fp16, int8 |
| Ampere (A100) | 16×16×16, 16×8×16 | fp16, bf16, tf32, int8 |
| Hopper (H100) | 16×16×16 + wgmma | fp16, bf16, fp8, int8 |

### FP8 for Maximum Throughput (Hopper)

```python
@triton.jit
def fp8_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # FP8 gives 2x throughput over FP16 on Hopper
    # E4M3 for forward, E5M2 for gradients (more range)
    
    pid = tl.program_id(0)
    # ... offset calculations ...
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load as fp8
        a = tl.load(a_ptr + ..., ...)  # Stored as fp8_e4m3
        b = tl.load(b_ptr + ..., ...)  # Stored as fp8_e4m3
        
        # Dot product with scaling
        acc = tl.dot(a, b, acc)
    
    # Scale and store
    c = (acc * scale).to(tl.float16)
    tl.store(c_ptr + ..., c)
```

## Warp-Level Programming

### Understanding Warp Execution in Triton

```python
@triton.jit
def warp_aware_kernel(
    ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    With BLOCK_SIZE=256 and num_warps=8:
    - 256 elements / 8 warps = 32 elements per warp
    - Each warp has 32 threads
    - Each thread handles 1 element
    
    With BLOCK_SIZE=256 and num_warps=4:
    - 256 elements / 4 warps = 64 elements per warp
    - Each warp has 32 threads
    - Each thread handles 2 elements
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # The compiler maps this to warps automatically
    x = tl.load(ptr + offs, mask=offs < N)
    tl.store(ptr + offs, x * 2, mask=offs < N)
```

### Warp Specialization (Hopper)

```python
# Advanced: Different warps do different work
@triton.jit
def warp_specialized_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # On Hopper, can have producer warps (load) and consumer warps (compute)
    # This is handled through num_stages and compiler optimizations
    
    # Producer warps: async load from global → shared
    # Consumer warps: compute from shared → registers
    # Barrier synchronization between them
    
    # Triton handles this through software pipelining
    for k in tl.range(0, K, BLOCK_K, num_stages=4):
        # Compiler splits this into producer/consumer pattern
        a = tl.load(...)
        b = tl.load(...)
        acc = tl.dot(a, b, acc)
```

## Advanced Memory Patterns

### Async Global to Shared Copy (Ampere+)

```python
@triton.jit
def async_copy_kernel(
    src_ptr, dst_ptr, N,
    BLOCK: tl.constexpr,
):
    """
    On Ampere+, loads can be asynchronous:
    1. Issue async load (cp.async)
    2. Do other work
    3. Wait for completion (cp.async.wait_group)
    
    Triton handles this through num_stages in tl.range()
    """
    pid = tl.program_id(0)
    
    # With num_stages > 1, compiler generates async copies
    for i in tl.range(0, N, BLOCK, num_stages=3):
        offs = pid * BLOCK + tl.arange(0, BLOCK) + i
        x = tl.load(src_ptr + offs)  # Async on Ampere+
        tl.store(dst_ptr + offs, x)
```

### Swizzling for Bank Conflict Avoidance

```python
@triton.jit
def swizzled_load(
    ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Shared memory has 32 banks. Accessing same bank from multiple
    threads in a warp causes serialization.
    
    Triton automatically applies swizzling, but you can hint:
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Standard 2D offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Using swizzle2d for custom patterns
    # (This is usually not needed - compiler handles it)
    swizzled_rn = tl.swizzle2d(rn, BLOCK_N, BLOCK_N, 8)
```

### Coalescing Patterns

```python
# ✅ COALESCED: Threads access consecutive addresses
# Thread 0 → addr 0, Thread 1 → addr 1, ...
offs = tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr + offs)  # Single memory transaction

# ❌ STRIDED: Threads access non-consecutive addresses  
# Thread 0 → addr 0, Thread 1 → addr 128, ...
offs = tl.arange(0, BLOCK_SIZE) * stride
x = tl.load(ptr + offs)  # Multiple transactions

# ✅ FIX strided access with transposition or tiling
# Load contiguous, then transpose in registers
```

## Occupancy and Resource Management

### Register Pressure

```python
@triton.jit
def high_register_kernel(
    ptr, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Large BLOCK sizes = more registers per thread
    More registers = fewer concurrent warps = lower occupancy
    
    Balance: Enough work to hide latency, not so much that occupancy drops
    """
    # This creates BLOCK_M * BLOCK_N floats in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # With BLOCK_M=128, BLOCK_N=128:
    # - 128*128 = 16384 elements
    # - With num_warps=8, that's 16384/8/32 = 64 elements/thread
    # - 64 fp32 = 256 bytes = 64 registers per thread
    # - Max is 255 registers/thread, so this is close to limit
```

### Checking Resource Usage

```python
# After compilation, check resource usage
compiled = kernel.warmup(args..., grid=(1,))

print(f"Registers per thread: {compiled.n_regs}")
print(f"Shared memory: {compiled.n_shared} bytes")
print(f"Spilled registers: {compiled.n_spills}")  # Bad if > 0
```

### Occupancy Calculation

```python
def estimate_occupancy(n_regs, n_shared, num_warps):
    """
    Rough occupancy estimation for Ampere (A100)
    """
    MAX_REGS_PER_SM = 65536
    MAX_SHARED_PER_SM = 164 * 1024  # 164KB
    MAX_WARPS_PER_SM = 64
    MAX_BLOCKS_PER_SM = 32
    
    # Warps limited by registers
    warps_by_regs = MAX_REGS_PER_SM // (n_regs * 32)
    
    # Warps limited by shared memory
    blocks_by_shared = MAX_SHARED_PER_SM // n_shared if n_shared > 0 else MAX_BLOCKS_PER_SM
    warps_by_shared = blocks_by_shared * num_warps
    
    # Actual limit
    active_warps = min(warps_by_regs, warps_by_shared, MAX_WARPS_PER_SM)
    occupancy = active_warps / MAX_WARPS_PER_SM
    
    return occupancy
```

## Persistent Kernels

### What Are Persistent Kernels?

Traditional: Launch many short-lived programs, each processes one tile
Persistent: Launch few long-lived programs, each processes many tiles

```python
@triton.jit
def persistent_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Instead of launching M/BLOCK_M * N/BLOCK_N programs,
    launch NUM_SMS programs that loop over all tiles.
    
    Benefits:
    - Lower launch overhead
    - Better L2 cache utilization
    - Can implement custom scheduling
    """
    pid = tl.program_id(0)
    
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    
    # Each program processes multiple tiles
    for tile_id in tl.range(pid, total_tiles, NUM_SMS):
        # Decode tile coordinates
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        
        # Process this tile
        offs_am = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            # ... load and compute ...
            a = tl.load(...)
            b = tl.load(...)
            acc = tl.dot(a, b, acc)
        
        # Store result
        tl.store(...)

# Launch with exactly NUM_SMS programs
grid = (NUM_SMS,)
persistent_matmul[grid](
    a, b, c, M, N, K,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    NUM_SMS=108,  # Number of SMs on A100
)
```

### Tile Scheduling Strategies

```python
@triton.jit
def scheduled_persistent_matmul(
    ...,
    GROUP_M: tl.constexpr,
):
    """
    Custom tile scheduling for better L2 locality
    """
    pid = tl.program_id(0)
    
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    
    for linear_tile_id in tl.range(pid, num_tiles_m * num_tiles_n, NUM_SMS):
        # Grouped scheduling (same as non-persistent)
        num_pid_in_group = GROUP_M * num_tiles_n
        group_id = linear_tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_tiles_m - first_pid_m, GROUP_M)
        
        tile_m = first_pid_m + (linear_tile_id % num_pid_in_group) % group_size_m
        tile_n = (linear_tile_id % num_pid_in_group) // group_size_m
        
        # ... process tile ...
```

## Debugging at the IR Level

### Using TRITON_INTERPRET=1

```bash
# Run kernel in interpreter mode (slow, but debuggable)
TRITON_INTERPRET=1 python my_script.py
```

```python
@triton.jit
def debuggable_kernel(ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    
    # In interpret mode, you can set breakpoints here
    x = tl.load(ptr + offs, mask=offs < N)
    
    # Print works in interpret mode
    if pid == 0:
        tl.device_print("x[0] =", x[0])
    
    tl.store(ptr + offs, x * 2, mask=offs < N)
```

### Analyzing PTX Output

```python
compiled = kernel.warmup(args..., grid=(1,))
ptx = compiled.asm['ptx']

# Look for:
# - ld.global.v4 → vectorized loads (good)
# - ld.global.b32 → scalar loads (might be bad)
# - bar.sync → synchronization points
# - mma.sync → tensor core usage
# - spill → register spilling (bad)
```

### Common PTX Patterns

```ptx
// Good: Vectorized load (128 bits = 4 floats)
ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];

// Good: Tensor core MMA
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...

// Bad: Scalar loads in a loop
ld.global.b32 %f1, [%rd1];
ld.global.b32 %f2, [%rd1+4];

// Bad: Register spilling to local memory
st.local.b32 [%rd_spill], %r1;
```
