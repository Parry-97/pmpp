"""
Parallel matrix multiplication with Triton

Author: Param Pal Singh
"""

import torch
import triton
import triton.language as tl
from pprint import pprint


@triton.jit
def naive_matmul_kernel(
    # Pointers to tensors
    x_ptr,
    y_ptr,
    output_ptr,
    width: int,
    # Block size (compile-time constant)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for parallel matrix multiplication with triton.

    Args:
        x_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Number of elements to process
        BLOCK_SIZE: Number of elements each program processes
    """
    # Get the program ID (equivalent to CUDA blockIdx.x)
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    # Compute the offsets for this program's block
    # tl.arange creates a vector: [0, 1, 2, ..., BLOCK_SIZE-1]
    out_offsets_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_offsets_col = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create mask to handle edge cases (when n_elements % BLOCK_SIZE != 0)
    out_mask_row = out_offsets_row < width
    out_mask_col = out_offsets_col < width

    out_mask = out_mask_row[:, None] & out_mask_col[None, :]
    out_offsets = out_offsets_row[:, None] * width + out_offsets_col[None, :]
    k = 0
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    while k < width:
        # Load x input data (BLOCK_SIZE elements at once)
        x_offsets_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_offsets_col = k + tl.arange(0, BLOCK_SIZE)
        x_mask = (x_offsets_row[:, None] < width) & (x_offsets_col[None, :] < width)
        x_offsets = x_offsets_row[:, None] * width + x_offsets_col[None, :]

        # Load y input data (BLOCK_SIZE elements at once)
        y_offsets_row = k + tl.arange(0, BLOCK_SIZE)
        y_offsets_col = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        y_mask = (y_offsets_row[:, None] < width) & (y_offsets_col[None, :] < width)
        y_offsets = y_offsets_row[:, None] * width + y_offsets_col[None, :]

        x = tl.load(x_ptr + x_offsets, mask=x_mask)
        y = tl.load(y_ptr + y_offsets, mask=y_mask)
        output += tl.dot(x, y)
        k += 32

    tl.store(output_ptr + out_offsets, output, mask=out_mask)


def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton kernel.

    Args:
        x: Input tensor
        y: Input tensor

    Returns:
        Output tensor after kernel computation
    """
    # Ensure input is contiguous
    x = x.contiguous()
    y = y.contiguous()

    # Allocate output tensor
    output = torch.zeros(x.size())

    # Define block size
    BLOCK_SIZE = tl.constexpr(32)

    # Calculate grid size (number of programs to launch)
    grid = (triton.cdiv(x.size(0), BLOCK_SIZE), triton.cdiv(x.size(1), BLOCK_SIZE))

    # Launch the kernel
    # NOTE: We assume the block size to be the same
    naive_matmul_kernel[grid](x, y, output, x.size(0), BLOCK_SIZE=BLOCK_SIZE)

    return output


if __name__ == "__main__":
    # Test the kernel
    torch.manual_seed(0)
    # NOTE: Let's consider for the initial example that they are square matrices
    # of the same size
    x = torch.rand((3, 3), dtype=torch.float32)
    y = torch.rand((3, 3), dtype=torch.float32)

    # Run the kernel
    output = matmul_kernel(x, y)

    # Verify correctness (replace with actual expected computation)
    expected = x @ y
    print("Expected Output:")
    pprint(expected)
    print("Actual Output:")
    pprint(output)
    assert torch.allclose(output, expected), "Kernel output does not match expected!"
    print("✓ Kernel test passed!")
