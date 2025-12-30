# Fused Kernels & Triton's "Automatic Fusion"

## What is a Fused Kernel?

A **Fused Kernel** is simply a single GPU kernel that performs a sequence of mathematical operations that would otherwise require multiple separate kernel launches.

### The Core Concept
*   **Unfused:** `Load -> Op1 -> Store (RAM) -> Load (RAM) -> Op2 -> Store`
*   **Fused:** `Load -> Op1 -> Op2 -> Store`

The performance gain comes from **eliminating Global Memory (HBM) accesses**. The intermediate data between `Op1` and `Op2` lives in fast **Registers** or **SRAM (L1/Shared Memory)**.

---

## "Wait, isn't this just normal programming?"

Yes. If you are writing raw CUDA C++, writing `float d = (a * b) + c;` is trivial. The intermediate value `(a * b)` naturally lives in a register.

So why is "Fusion" such a big deal in the AI world?

1.  **Framework Limitations (The "PyTorch Problem"):**
    Deep Learning frameworks execute eagerly (line-by-line). Doing `x = x + y` and then `z = x * w` launches two separate compiled binaries. This forces a write to global memory in between.
    
2.  **Complexity of Shared Memory Fusion:**
    Fusion is easy for element-wise math (`+`, `-`, `*`). It becomes **extremely difficult** when fusing **reductions** (like `Sum`, `Max`) with other operations (like `Softmax` or `LayerNorm`).

---

## The Real Value: Fusing Reductions (CUDA vs. Triton)

The term "Automatic Fusion" in Triton refers to its ability to handle **Shared Memory management and Synchronization** automatically, which is the hardest part of writing fused kernels in CUDA.

### Example: Fused Softmax (`x = exp(x) / sum(exp(x))`)

This requires communicating data between threads to calculate the `sum`.

| Feature | Manual CUDA C++ | Triton |
| :--- | :--- | :--- |
| **Memory** | You must manually allocate `__shared__ float[]` arrays. | Compiler manages memory automatically. |
| **Sync** | You must manually place `__syncthreads()` barriers. ONE mistake = Race Condition or Deadlock. | Compiler places barriers automatically. |
| **Algorithm** | You must write the tree-reduction loop (e.g., `for (int s=blockDim/2; ...)`). | You just write `tl.sum(x, axis=0)`. |
| **Code** | ~50+ lines of complex, fragile C++ code. | ~5 lines of Python-like code. |

### Comparison Summary

*   **Element-wise Fusion (Easy):** 
    *   **CUDA:** `float t = a + b; float c = t * d;` (Same as Triton)
    *   **Triton:** `t = a + b; c = t * d;` (Same as CUDA)

*   **Reduction Fusion (Hard):**
    *   **CUDA:** Requires explicit `__shared__` memory, parallel reduction algorithms, and careful barrier synchronization.
    *   **Triton:** You write `tl.sum()`, and the compiler **automatically fuses** the reduction logic into your kernel, generating the optimal shared memory and synchronization instructions for you.
