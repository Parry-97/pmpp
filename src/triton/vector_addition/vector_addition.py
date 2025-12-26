"""
Parallel vector addition implementation in Triton

Author: Param Pal Singh
"""

import torch
import triton
from pprint import pprint
import triton.language as tl


@triton.jit
def vector_addition_kernel(
    # Pointers to tensors
    x_ptr,
    y_ptr,
    output_ptr,
    # Size of the input
    n_elements,
    # Block size (compile-time constant)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for parallel vector addition implementation in triton.

    Args:
        x_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Number of elements to process
        BLOCK_SIZE: Number of elements each program processes
    """
    # Get the program ID (equivalent to CUDA blockIdx.x)
    pid = tl.program_id(axis=0)

    # Compute the offsets for this program's block
    # tl.arange creates a vector: [0, 1, 2, ..., BLOCK_SIZE-1]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask to handle edge cases (when n_elements % BLOCK_SIZE != 0)
    mask = offsets < n_elements

    # Load input data (BLOCK_SIZE elements at once)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # ============================================
    # TODO: Implement your kernel computation here
    # ============================================
    output = x + y  # Replace with actual computation

    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_addition(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton kernel.

    Args:
        x: Input tensor
        y: Input tensor

    Returns:
        Output tensor after kernel computation
    """
    # Ensure input is on CUDA and contiguous
    # assert x.is_cuda, "Input tensor must be on CUDA device"

    assert x.size() == y.size(), "The input tensors must have the same size"

    x = x.contiguous()
    y = y.contiguous()

    # Allocate output tensor
    output = torch.empty_like(x)

    # Get total number of elements
    n_elements = x.numel()

    # Define block size
    BLOCK_SIZE = tl.constexpr(1024)

    # Calculate grid size (number of programs to launch)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the kernel
    vector_addition_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output


if __name__ == "__main__":
    # Test the kernel
    torch.manual_seed(0)
    x = torch.zeros(1024, dtype=torch.float32)
    y = torch.ones(1024, dtype=torch.float32)

    # Run the kernel
    output = vector_addition(x, y)

    # Verify correctness (replace with actual expected computation)
    pprint(output)
    expected = y
    assert torch.allclose(output, expected), "Kernel output does not match expected!"
    print("✓ Kernel test passed!")
