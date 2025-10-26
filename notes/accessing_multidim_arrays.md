### Why `A[j][i]` doesn’t work for dynamically allocated arrays in C (and CUDA C)

#### 1. Static vs. Dynamic arrays

| Case                                     | Example                                       | Works with `A[j][i]`? | Why                                                                   |
| ---------------------------------------- | --------------------------------------------- | --------------------- | --------------------------------------------------------------------- |
| **Static** (size known at compile time)  | `float A[4][5];`                              | ✅ Yes                | Compiler knows each row has 5 elements → can compute offset `j*5 + i` |
| **Dynamic** (size known only at runtime) | `float *A = malloc(nx * ny * sizeof(float));` | ❌ No                 | Compiler only sees `float*` → doesn’t know row stride (`nx`)          |

---

#### 2. The core rule

In C, multidimensional indexing `A[j][i]` only works if the **number of columns per row is encoded in the type** and thus known at compile time.
Dynamic allocations (via `malloc`, `cudaMalloc`, etc.) lack that info — so the compiler can’t do the address arithmetic for you.

---

#### 3. What happens under the hood

All arrays in C are stored linearly in memory.
The compiler automatically linearizes:

```
A[j][i] → *(A + j * num_cols + i)
```

…but only when `num_cols` is known at compile time.

---

#### 4. The error you’ll see

Example that fails:

```c
float *A = malloc(nx * ny * sizeof(float));
A[2][3] = 1.0f;  // ❌
```

Error:

```
error: subscripted value is neither array nor pointer nor vector
```

---

#### 5. The correct CUDA-style way

Manually compute the flat index:

```c
A[j * nx + i] = value;
```

In kernels:

```c
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;
if (i < nx && j < ny)
    A[j * nx + i] = ...;
```

---

#### 6. Quick summary

| Concept                        | Static Array | Dynamic Array |
| ------------------------------ | ------------ | ------------- |
| Known at compile time          | ✅           | ❌            |
| Compiler can compute 2D offset | ✅           | ❌            |
| Needs manual flattening        | ❌           | ✅            |
| Typical in CUDA                | Rare         | Common        |

---
