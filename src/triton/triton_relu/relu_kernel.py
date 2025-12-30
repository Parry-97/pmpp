"""
ReLU Activation with Triton

Author: Param Pal Singh
"""

import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    # Pointers to tensors
    x_ptr,
    output_ptr,
    # Size of the input
    n_elements,
    # Block size (compile-time constant)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for relu activation with triton.

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

    # ============================================
    # NOTE: Implement your kernel computation here
    # ============================================
    output = tl.maximum(0, x)  # Replace with actual computation

    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton kernel.

    Args:
        x: Input tensor

    Returns:
        Output tensor after kernel computation
    """
    # Ensure input is contiguous
    x = x.contiguous()

    # Allocate output tensor
    output = torch.empty_like(x)

    # Get total number of elements
    n_elements = x.numel()

    # Define block size
    BLOCK_SIZE = tl.constexpr(1024)

    # Calculate grid size (number of programs to launch)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the kernel
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output


if __name__ == "__main__":
    # Test the kernel
    torch.manual_seed(0)
    x = torch.rand(1000, dtype=torch.float32)

    # Run the kernel
    output = relu(x)

    # Verify correctness (replace with actual expected computation)
    expected = torch.relu(x)  # TODO: Replace with expected result
    assert torch.allclose(output, expected), "Kernel output does not match expected!"
    print("✓ Kernel test passed!")
