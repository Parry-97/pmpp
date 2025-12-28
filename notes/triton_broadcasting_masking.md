# Triton 2D Grids: Broadcasting & Masking Explained

In Triton, handling 2D data (like images or matrices) relies heavily on **broadcasting**. This is the mechanism that allows you to combine 1D vectors (like row indices and column indices) into 2D grids of coordinates or masks.

## The Goal: 2D Coordinates

When processing a block of pixels (e.g., 32x32), every thread/element in that block needs to know its specific coordinate `(row, col)`.

Instead of iterating with loops, we define:
1.  A **Vertical Vector** of row indices.
2.  A **Horizontal Vector** of column indices.
3.  We "add" or "AND" them together to create a full 2D grid.

---

## 1. Broadcasting for Coordinates (Offsets)

Imagine a small 3x4 block.

### Step A: Create the 1D Vectors
We calculate the global row and column indices for the block.

```python
# Assume pid_row=0, pid_col=0 for simplicity
rows = tl.arange(0, 3)  # [0, 1, 2]
cols = tl.arange(0, 4)  # [0, 1, 2, 3]
```

### Step B: Orient Them (The `None` Trick)
This is the key step. We use `None` (or `np.newaxis`) to change the *shape* of the vectors so they are perpendicular.

*   `rows[:, None]` makes it a **Column Vector** (Shape: `3x1`).
*   `cols[None, :]` makes it a **Row Vector** (Shape: `1x4`).

```text
rows[:, None]       cols[None, :]
  ┌   ┐             ┌            ┐
  │ 0 │             │ 0  1  2  3 │
  │ 1 │             └            ┘
  │ 2 │
  └   ┘
```

### Step C: Broadcast & Combine
When you perform an operation (like `+` or `*`), Triton stretches these vectors to match the full `3x4` block size.

**Calculating Linear Offsets:** `offset = row * WIDTH + col`
Assume `WIDTH = 10`.

1.  **Term 1:** `rows[:, None] * 10`
    The column vector `[0, 1, 2]` becomes `[0, 10, 20]` and then stretches across:
    ```text
    ┌            ┐
    │  0   0   0   0 │  (Rows 0 start at 0)
    │ 10  10  10  10 │  (Rows 1 start at 10)
    │ 20  20  20  20 │  (Rows 2 start at 20)
    └            ┘
    ```

2.  **Term 2:** `cols[None, :]`
    The row vector `[0, 1, 2, 3]` stretches down:
    ```text
    ┌            ┐
    │ 0  1  2  3 │
    │ 0  1  2  3 │
    │ 0  1  2  3 │
    └            ┘
    ```

3.  **Result:** Term 1 + Term 2
    ```text
    ┌               ┐
    │  0   1   2   3 │  (Linear indices for Row 0)
    │ 10  11  12  13 │  (Linear indices for Row 1)
    │ 20  21  22  23 │  (Linear indices for Row 2)
    └               ┘
    ```

---

## 2. Broadcasting for Masking

We use the exact same logic to create a 2D `mask` that tells us which pixels are valid (inside the image) and which are padding (outside).

### Step A: 1D Checks
Check bounds for rows and columns separately.
```python
# Assume Image Height=2, Width=3
# Block is 3x4 (so we are processing some out-of-bounds pixels)

mask_r = rows < 2  # [True, True, False]
mask_c = cols < 3  # [True, True, True, False]
```

### Step B: Orient & Broadcast
We combine them with a logical `&` (AND).
`mask_r[:, None] & mask_c[None, :]`

1.  **Vertical Mask (Rows):** Stretched sideways
    ```text
    ┌                  ┐
    │ True  True  True  True │ (Row 0 is valid)
    │ True  True  True  True │ (Row 1 is valid)
    │ False False False False│ (Row 2 is invalid)
    └                  ┘
    ```

2.  **Horizontal Mask (Cols):** Stretched downwards
    ```text
    ┌                  ┐
    │ True  True  True  False│ (Cols 0-2 valid)
    │ True  True  True  False│ (Cols 0-2 valid)
    │ True  True  True  False│ (Cols 0-2 valid)
    └                  ┘
    ```

3.  **Result (AND):** The Intersection
    ```text
    ┌                  ┐
    │ True  True  True  False│ -> Valid Pixels
    │ True  True  True  False│ -> Valid Pixels
    │ False False False False│ -> Invalid (Row 2)
    └                  ┘
                 ^
             Invalid (Col 3)
    ```

This 2D mask is then passed to `tl.load` and `tl.store` to ensure we don't read/write invalid memory.
