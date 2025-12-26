# Triton Mastery Resources

## Official Resources

### Documentation
- **Triton Language Reference**: https://triton-lang.org/main/python-api/triton.language.html
- **Triton Tutorials**: https://triton-lang.org/main/getting-started/tutorials/
- **Triton Programming Guide**: https://triton-lang.org/main/programming-guide/

### Source Code (Essential for Deep Understanding)
- **Triton Compiler**: https://github.com/triton-lang/triton
  - `python/triton/` - Python frontend
  - `lib/Dialect/` - MLIR dialects (TTIR, TTGIR)
  - `lib/Conversion/` - IR transformations
  - `third_party/nvidia/` - NVIDIA-specific backends

## Production-Quality Reference Implementations

### 1. Flash Attention
**Repository**: https://github.com/Dao-AILab/flash-attention
**Why study it**:
- Gold standard for Triton attention kernels
- Shows backward pass implementation
- Memory-efficient attention patterns
- Causal and non-causal variants

```bash
# Key files to study:
flash_attn/flash_attn_triton.py
flash_attn/flash_attn_triton_kernel.py
```

### 2. Unsloth
**Repository**: https://github.com/unslothai/unsloth
**Why study it**:
- Production LLM training kernels
- Fused backward passes
- RoPE, cross-entropy, SwiGLU implementations
- Real-world optimization patterns

```bash
# Key files:
unsloth/kernels/
├── cross_entropy_loss.py
├── layernorm.py
├── rope_embedding.py
├── swiglu.py
└── utils.py
```

### 3. vLLM
**Repository**: https://github.com/vllm-project/vllm
**Why study it**:
- Production inference server kernels
- PagedAttention implementation
- Quantization kernels (AWQ, GPTQ, SqueezeLLM)
- Speculative decoding

```bash
# Key files:
vllm/attention/ops/
csrc/quantization/
```

### 4. xFormers
**Repository**: https://github.com/facebookresearch/xformers
**Why study it**:
- Meta's attention library
- Multiple attention variants
- Memory-efficient implementations
- Comprehensive testing

### 5. FlagGems
**Repository**: https://github.com/FlagOpen/FlagGems
**Why study it**:
- Comprehensive operator library
- Clean implementations of common ops
- Good for learning patterns

### 6. Liger Kernel
**Repository**: https://github.com/linkedin/Liger-Kernel
**Why study it**:
- LinkedIn's production kernels
- Training-focused optimizations
- Fused operations

## Research Papers (Essential Reading)

### Foundational
1. **"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"** (Tillet et al., 2019)
   - Original Triton paper
   - Understand the design philosophy

2. **"Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022)
   - The algorithm that revolutionized attention
   - Online softmax technique
   - IO complexity analysis

3. **"Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning"** (Dao, 2023)
   - Improved tiling strategies
   - Better GPU utilization

### Advanced Topics
4. **"FlashDecoding for Long-Context LLM Inference"** (Hong et al., 2023)
   - Attention for long sequences
   - Parallel reduction patterns

5. **"PagedAttention"** (Kwon et al., 2023)
   - Memory management for inference
   - Dynamic batching

6. **"FP8 Formats for Deep Learning"** (NVIDIA, 2022)
   - FP8 training and inference
   - Scaling strategies

## GPU Architecture Resources

### NVIDIA Documentation
- **CUDA C Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **PTX ISA Reference**: https://docs.nvidia.com/cuda/parallel-thread-execution/
- **Ampere Tuning Guide**: https://docs.nvidia.com/cuda/ampere-tuning-guide/
- **Hopper Tuning Guide**: https://docs.nvidia.com/cuda/hopper-tuning-guide/

### Architecture Deep Dives
- **Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking**: https://arxiv.org/abs/1804.06826
- **Dissecting the Ampere GPU Architecture through Microbenchmarking**: https://arxiv.org/abs/2208.11174

## Learning Exercises (Mastery Path)

### Stage 1: Fundamentals (Week 1-2)
```python
# Exercise 1.1: Vector operations
- Vector add, multiply, subtract
- Fused multiply-add (FMA)
- Element-wise activation functions

# Exercise 1.2: Reductions
- Sum, max, min reduction
- Argmax, argmin
- Variance (two-pass and Welford's online)

# Exercise 1.3: 2D operations
- Matrix transpose
- Element-wise 2D operations
```

### Stage 2: Memory Patterns (Week 3-4)
```python
# Exercise 2.1: Tiled matrix multiply
- Naive tiled matmul
- L2-cache-aware scheduling (GROUP_SIZE_M)
- Autotuning setup

# Exercise 2.2: Memory coalescing
- Implement strided access kernel
- Measure bandwidth
- Fix with tiling/transposition

# Exercise 2.3: Softmax variations
- Row-wise softmax
- Column-wise softmax
- Online softmax (Flash style)
```

### Stage 3: Fusion Patterns (Week 5-6)
```python
# Exercise 3.1: LayerNorm
- Forward pass (fused mean, var, normalize, scale)
- Backward pass (complex gradient flow)

# Exercise 3.2: Attention building blocks
- QK^T computation
- Softmax
- Weighted sum with V

# Exercise 3.3: Transformer FFN
- Fused matmul + activation
- SwiGLU/GeGLU
```

### Stage 4: Advanced (Week 7-8)
```python
# Exercise 4.1: Flash Attention
- Study the algorithm deeply
- Implement forward pass
- Add causal masking
- (Advanced) Backward pass

# Exercise 4.2: Persistent kernels
- Implement persistent matmul
- Custom tile scheduling

# Exercise 4.3: Quantization
- INT8 matmul
- Dequantization fused with compute
```

### Stage 5: Mastery (Ongoing)
```python
# Exercise 5.1: Read and understand production code
- Flash Attention source
- Unsloth kernels
- vLLM ops

# Exercise 5.2: Contribute
- Find performance issue in existing library
- Profile and identify bottleneck
- Submit optimization PR

# Exercise 5.3: Novel kernel
- Implement a kernel for new architecture feature
- Write about your findings
```

## Profiling and Debugging Tools

### NVIDIA Nsight Systems
```bash
# Profile Triton kernels
nsys profile -o output python my_script.py

# View in GUI
nsys-ui output.nsys-rep
```

### NVIDIA Nsight Compute
```bash
# Detailed kernel analysis
ncu --set full -o output python my_script.py

# Specific metrics
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum python my_script.py
```

### Triton-Specific
```bash
# Interpreter mode (slow, but debuggable)
TRITON_INTERPRET=1 python my_script.py

# Print autotuning results
TRITON_PRINT_AUTOTUNING=1 python my_script.py

# Dump IR at various stages
MLIR_ENABLE_DUMP=1 python my_script.py
```

### PyTorch Profiler
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    output = my_triton_kernel(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Community Resources

### Discord/Slack
- **Triton Discord**: Active community, quick answers
- **PyTorch Discord**: #triton channel

### GitHub Discussions
- https://github.com/triton-lang/triton/discussions
- Search for similar problems before asking

### Blog Posts and Tutorials
1. **"Understanding Flash Attention"** - Aleksa Gordić
2. **"GPU Performance Optimization"** - Lei Mao's blog
3. **"Triton Puzzles"** - Sasha Rush (excellent learning exercises)
   - https://github.com/srush/Triton-Puzzles

## Hardware Considerations

### GPU Tiers for Learning

| GPU | Memory | Tensor Cores | Good For |
|-----|--------|--------------|----------|
| RTX 3060 | 12GB | Yes (3rd gen) | Learning, small models |
| RTX 4090 | 24GB | Yes (4th gen) | Serious development |
| A100 | 40/80GB | Yes (3rd gen) | Production workloads |
| H100 | 80GB | Yes (4th gen) | Cutting edge (FP8, TMA) |

### Architecture-Specific Features

| Feature | Ampere (A100) | Hopper (H100) |
|---------|---------------|---------------|
| TF32 Tensor Cores | ✅ | ✅ |
| BF16 Tensor Cores | ✅ | ✅ |
| FP8 Tensor Cores | ❌ | ✅ |
| TMA (Tensor Memory Accelerator) | ❌ | ✅ |
| Warp Specialization | Limited | Full |
| Async copy (cp.async) | ✅ | ✅ |
| Distributed shared memory | ❌ | ✅ |

## Recommended Study Order

```
Week 1-2: Fundamentals
├── Read triton_fundamentals.md
├── Implement vector_add variations
├── Implement element-wise activations
└── Understand constexpr and masks

Week 3-4: Memory
├── Read triton_advanced_internals.md (memory section)
├── Implement tiled matmul
├── Add autotuning
└── Profile and understand memory behavior

Week 5-6: Fusion
├── Read triton_advanced_kernels.md
├── Implement softmax
├── Implement LayerNorm
└── Understand kernel fusion benefits

Week 7-8: Attention
├── Study Flash Attention paper
├── Implement online softmax
├── Implement Flash Attention forward
└── Understand memory vs compute tradeoffs

Month 3+: Mastery
├── Read production code (Flash Attention, Unsloth)
├── Implement backward passes
├── Explore new GPU features (FP8, TMA)
├── Contribute to open source
└── Write your own optimizations
```

## Your Files Summary

```
notes/
├── triton_fundamentals.md      ← Mental model, CUDA→Triton mapping
├── triton_operations.md        ← API reference
├── triton_patterns.md          ← Common kernel patterns
├── triton_autotuning.md        ← Performance tuning
├── triton_debugging.md         ← Debugging and pitfalls
├── triton_roadmap.md           ← Learning path and exercises
├── triton_advanced_internals.md ← Compiler, IR, memory hierarchy
├── triton_advanced_kernels.md  ← Flash Attention, LayerNorm, etc.
└── triton_mastery_resources.md ← This file
```

## Final Advice

1. **Read code, not just tutorials**: The best Triton developers read production implementations

2. **Profile everything**: Don't guess about performance - measure it

3. **Understand the hardware**: Know your GPU's memory bandwidth, compute throughput, and cache sizes

4. **Start simple, optimize incrementally**: Get correctness first, then optimize

5. **Contribute to open source**: Best way to learn is to solve real problems

6. **Stay current**: Triton evolves rapidly - follow the GitHub repo

Good luck on your journey to mastering Triton! 🚀
