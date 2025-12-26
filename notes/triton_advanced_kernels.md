# Advanced Triton Kernel Implementations

## Flash Attention - The Crown Jewel

Flash Attention is a memory-efficient attention implementation that:
1. Never materializes the full N×N attention matrix
2. Uses tiling and online softmax
3. Achieves O(N) memory instead of O(N²)

### The Algorithm

```
Standard Attention:
Q, K, V ∈ R^(N×d)
S = QK^T           ← O(N²) memory!
P = softmax(S)     ← O(N²) memory!
O = PV

Flash Attention:
For each block of Q:
    For each block of K, V:
        Compute local S = Q_block @ K_block^T
        Update running softmax statistics
        Accumulate O_block incrementally
    Rescale O_block by final softmax denominator
```

### Online Softmax: The Key Insight

```python
# Standard softmax (needs full row):
# softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

# Online softmax (incremental):
# Process chunks, track running max and sum

def online_softmax_update(m_prev, l_prev, o_prev, m_new, l_new, o_new):
    """
    m: running max
    l: running sum of exp(x - m)
    o: running weighted output
    """
    m = max(m_prev, m_new)
    
    # Rescale previous statistics
    l = l_prev * exp(m_prev - m) + l_new * exp(m_new - m)
    o = o_prev * exp(m_prev - m) + o_new * exp(m_new - m)
    
    return m, l, o
```

### Flash Attention Forward Kernel

```python
@triton.jit
def flash_attention_fwd(
    Q, K, V, O,
    Lse,  # Log-sum-exp for backward pass
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program indices
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Base pointers for this batch/head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load Q block - stays in registers for entire K,V loop
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # Running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # Running sum
    o_i = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)   # Running output
    
    # Scale factor
    qk_scale = 1.0 / tl.sqrt(tl.cast(BLOCK_DMODEL, tl.float32))
    q = q * qk_scale
    
    # Determine loop bounds (causal masking)
    if IS_CAUSAL:
        hi = min((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX
    
    # Loop over K, V blocks
    for start_n in range(0, hi, BLOCK_N):
        # Load K block
        k_ptrs = K + k_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        k_ptrs = k_ptrs + start_n * stride_kn
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        
        # Compute QK^T for this block
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        
        # Causal masking
        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (start_n + offs_n[None, :]),
                qk,
                float('-inf')
            )
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)  # Row-wise max
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Rescale old values
        alpha = tl.exp(m_i - m_i_new)
        
        # Compute exp(qk - m_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, axis=1)
        
        # Update running sum
        l_i = l_i * alpha + l_ij
        
        # Load V block
        v_ptrs = V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        v_ptrs = v_ptrs + start_n * stride_vn
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        
        # Update output accumulator
        o_i = o_i * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        # Update max
        m_i = m_i_new
    
    # Final normalization
    o_i = o_i / l_i[:, None]
    
    # Store output
    o_ptrs = O + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, o_i.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    
    # Store log-sum-exp for backward
    lse_ptrs = Lse + off_hz * N_CTX + offs_m
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)
```

### Flash Attention Memory Analysis

```
Standard Attention (N=4096, d=64):
- S matrix: 4096 × 4096 × 4 bytes = 64 MB
- P matrix: 4096 × 4096 × 4 bytes = 64 MB
- Total: 128 MB per attention head

Flash Attention:
- Q block: BLOCK_M × d × 4 bytes = 128 × 64 × 4 = 32 KB
- K block: BLOCK_N × d × 4 bytes = 128 × 64 × 4 = 32 KB
- V block: BLOCK_N × d × 4 bytes = 128 × 64 × 4 = 32 KB
- Accumulators: BLOCK_M × d × 4 = 32 KB
- Total: ~128 KB per program (fits in shared memory!)
```

## Fused Layer Normalization

### Standard LayerNorm (Multiple Kernels)

```python
# PyTorch implementation - 5 kernel launches!
def layer_norm_pytorch(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)      # Kernel 1
    var = x.var(dim=-1, keepdim=True)        # Kernel 2
    x_norm = (x - mean) / torch.sqrt(var + eps)  # Kernel 3
    return weight * x_norm + bias            # Kernel 4 (elementwise)
```

### Fused LayerNorm (Single Kernel)

```python
@triton.jit
def layer_norm_fwd_kernel(
    X,      # Input
    Y,      # Output
    W,      # Weight (gamma)
    B,      # Bias (beta)
    Mean,   # Output mean (for backward)
    Rstd,   # Output 1/std (for backward)
    stride,
    N,      # Number of elements per row
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program normalizes one row
    row = tl.program_id(0)
    
    # Pointers to this row
    X_row = X + row * stride
    Y_row = Y + row * stride
    
    # Load row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    x_norm = x_centered * rstd
    
    # Scale and shift
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = x_norm * w + b
    
    # Store output
    tl.store(Y_row + cols, y.to(Y.dtype.element_ty), mask=mask)
    
    # Store statistics for backward
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


@triton.jit
def layer_norm_bwd_kernel(
    DY,     # Gradient of output
    X,      # Original input
    W,      # Weight
    Mean,   # Saved mean
    Rstd,   # Saved 1/std
    DX,     # Gradient of input (output)
    DW,     # Gradient of weight (output, partial)
    DB,     # Gradient of bias (output, partial)
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load saved values
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    
    # Load inputs
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    
    # Compute normalized x
    x_hat = (x - mean) * rstd
    
    # Gradient of weight and bias (will need atomic add across rows)
    dw = dy * x_hat
    db = dy
    
    # Gradient of x (complex due to mean/var dependencies)
    # dx = rstd * (dy * w - (1/N) * sum(dy * w) - x_hat * (1/N) * sum(dy * w * x_hat))
    wdy = w * dy
    mean_wdy = tl.sum(wdy, axis=0) / N
    mean_wdy_xhat = tl.sum(wdy * x_hat, axis=0) / N
    dx = rstd * (wdy - mean_wdy - x_hat * mean_wdy_xhat)
    
    # Store
    tl.store(DX + row * stride + cols, dx, mask=mask)
    
    # Partial sums for DW, DB (need reduction across rows)
    tl.atomic_add(DW + cols, dw, mask=mask)
    tl.atomic_add(DB + cols, db, mask=mask)
```

## Fused Softmax with Temperature

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Apply temperature
    row = row / temperature
    
    # Numerically stable softmax
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

## Fused Dropout + Residual + LayerNorm

Common in transformers: `LayerNorm(x + Dropout(sublayer(x)))`

```python
@triton.jit
def dropout_residual_layernorm_kernel(
    X,          # Sublayer output (to be dropped)
    Residual,   # Residual connection
    W, B,       # LayerNorm params
    Y,          # Output
    Mean, Rstd, # Stats for backward
    seed,       # Dropout seed
    p,          # Dropout probability
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load sublayer output and residual
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    res = tl.load(Residual + row * stride + cols, mask=mask, other=0.0)
    
    # Generate dropout mask (seeded random)
    # Each element gets unique seed based on row and col
    random = tl.rand(seed, row * N + cols)
    keep_mask = random > p
    scale = 1.0 / (1.0 - p)
    
    # Apply dropout and add residual
    x_dropped = tl.where(keep_mask & mask, x * scale, 0.0)
    x_residual = x_dropped + res
    
    # Layer normalization
    mean = tl.sum(x_residual, axis=0) / N
    x_centered = x_residual - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    x_norm = x_centered * rstd
    
    # Scale and shift
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = x_norm * w + b
    
    # Store
    tl.store(Y + row * stride + cols, y, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
```

## RoPE (Rotary Position Embeddings)

Used in LLaMA, Mistral, and most modern LLMs:

```python
@triton.jit
def rope_kernel(
    Q, K,           # Query and Key tensors
    Cos, Sin,       # Precomputed cos/sin tables
    seq_len,
    head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RoPE formula:
    q_rot[2i] = q[2i] * cos[i] - q[2i+1] * sin[i]
    q_rot[2i+1] = q[2i] * sin[i] + q[2i+1] * cos[i]
    """
    # Program handles one (batch, head, position) triple
    pid = tl.program_id(0)
    batch = pid // (seq_len * num_heads)
    remainder = pid % (seq_len * num_heads)
    head = remainder // seq_len
    pos = remainder % seq_len
    
    # Pointers
    q_ptr = Q + batch * stride_qb + head * stride_qh + pos * stride_qs
    k_ptr = K + batch * stride_kb + head * stride_kh + pos * stride_ks
    cos_ptr = Cos + pos * (head_dim // 2)
    sin_ptr = Sin + pos * (head_dim // 2)
    
    # Process pairs of elements
    for i in range(0, head_dim // 2, BLOCK_SIZE):
        idx = tl.arange(0, BLOCK_SIZE)
        mask = i + idx < head_dim // 2
        
        # Load cos/sin for this position
        cos = tl.load(cos_ptr + i + idx, mask=mask)
        sin = tl.load(sin_ptr + i + idx, mask=mask)
        
        # Load Q pairs
        q_even = tl.load(q_ptr + (i + idx) * 2 * stride_qd, mask=mask)
        q_odd = tl.load(q_ptr + ((i + idx) * 2 + 1) * stride_qd, mask=mask)
        
        # Rotate Q
        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd = q_even * sin + q_odd * cos
        
        # Store Q
        tl.store(q_ptr + (i + idx) * 2 * stride_qd, q_rot_even, mask=mask)
        tl.store(q_ptr + ((i + idx) * 2 + 1) * stride_qd, q_rot_odd, mask=mask)
        
        # Same for K
        k_even = tl.load(k_ptr + (i + idx) * 2 * stride_kd, mask=mask)
        k_odd = tl.load(k_ptr + ((i + idx) * 2 + 1) * stride_kd, mask=mask)
        
        k_rot_even = k_even * cos - k_odd * sin
        k_rot_odd = k_even * sin + k_odd * cos
        
        tl.store(k_ptr + (i + idx) * 2 * stride_kd, k_rot_even, mask=mask)
        tl.store(k_ptr + ((i + idx) * 2 + 1) * stride_kd, k_rot_odd, mask=mask)
```

## Cross-Entropy Loss (Fused)

```python
@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,     # (batch, vocab_size)
    targets_ptr,    # (batch,)
    losses_ptr,     # (batch,)
    stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: log_softmax + nll_loss
    loss = -log(softmax(logits)[target])
         = -logits[target] + log(sum(exp(logits)))
         = -logits[target] + max(logits) + log(sum(exp(logits - max)))
    """
    row = tl.program_id(0)
    
    # Load target class
    target = tl.load(targets_ptr + row)
    
    # Compute log-sum-exp in chunks
    row_ptr = logits_ptr + row * stride
    
    m = tl.full([1], float('-inf'), dtype=tl.float32)  # Running max
    s = tl.zeros([1], dtype=tl.float32)                 # Running sum
    
    for start in range(0, vocab_size, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        
        # Online log-sum-exp
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    
    # log-sum-exp = m + log(s)
    lse = m + tl.log(s)
    
    # Load logit for target class
    target_logit = tl.load(row_ptr + target)
    
    # Loss = -target_logit + lse
    loss = -target_logit + lse
    
    tl.store(losses_ptr + row, loss)
```

## SwiGLU Activation (LLaMA FFN)

```python
@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SwiGLU(x, y) = Swish(x) * y
                 = (x * sigmoid(x)) * y
    
    Used in LLaMA: FFN(x) = SwiGLU(W_gate @ x, W_up @ x) @ W_down
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    gate = tl.load(gate_ptr + offs, mask=mask)
    up = tl.load(up_ptr + offs, mask=mask)
    
    # Swish activation: x * sigmoid(x)
    swish = gate * tl.sigmoid(gate)
    
    # Gated output
    output = swish * up
    
    tl.store(output_ptr + offs, output, mask=mask)
```
