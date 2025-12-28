"""
Parallel implementation of grayscale filter in Triton

Author: Param Pal Singh
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel_kernel(
    # Pointers to tensors
    x_ptr,
    output_ptr,
    # Size of the input
    height,
    width,
    # Block size (compile-time constant)
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Triton kernel for parallel implementation of grayscale filter in triton.

    Args:
        x_ptr: Pointer to input 2D tensor (H, W, 3)
        output_ptr: Pointer to output 2D tensor (H, W)
        height: height of the input tensor
        width: width of the input tensor
        BLOCK_SIZE_H: number of elements in BLOCK HEIGHT
        BLOCK_SIZE_W: number of elements in BLOCK WIDTH
    """
    # Get the program ID (equivalent to CUDA blockIdx.x)
    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    # Compute the offsets for this program's block
    block_start_h = pid_h * BLOCK_SIZE_H
    block_start_w = pid_w * BLOCK_SIZE_W

    offsets_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)
    offsets_w = block_start_w + tl.arange(0, BLOCK_SIZE_W)

    # Create mask to handle edge cases
    mask_row = offsets_h < height
    mask_col = offsets_w < width
    mask = mask_row[:, None] & mask_col[None, :]

    img_offsets = offsets_h[:, None] * width + offsets_w[None, :]
    rgb_offset = img_offsets * 3

    # Load input data
    r = tl.load(x_ptr + rgb_offset, mask=mask)
    g = tl.load(x_ptr + rgb_offset + 1, mask=mask)
    b = tl.load(x_ptr + rgb_offset + 2, mask=mask)

    # Compute grayscale value
    output = 0.21 * r + 0.71 * g + 0.07 * b

    # Store the result
    tl.store(output_ptr + img_offsets, output, mask=mask)


def grayscale_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton kernel.

    Args:
        x: Input tensor of shape (H, W, 3)

    Returns:
        Output tensor after kernel computation
    """
    x = x.contiguous()

    height, width, channels = x.shape
    assert channels == 3, "Input tensor must have 3 channels (RGB)"

    # Allocate output tensor (single channel)
    output = torch.empty((height, width), device=x.device, dtype=x.dtype)

    # Define block size
    BLOCK_SIZE_H = tl.constexpr(16)
    BLOCK_SIZE_W = tl.constexpr(16)

    # Calculate grid size (2D grid)
    grid = (
        triton.cdiv(height, BLOCK_SIZE_H),
        triton.cdiv(width, BLOCK_SIZE_W),
    )

    # Launch the kernel
    grayscale_kernel_kernel[grid](
        x, output, height, width, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    return output


if __name__ == "__main__":
    # Test the kernel
    torch.manual_seed(0)
    H, W = 1024, 2048  # Test with non-square dimensions
    x = torch.rand((H, W, 3), dtype=torch.float32)

    # Run the kernel
    output_triton = grayscale_kernel(x)

    # Reference computation in PyTorch
    expected = 0.21 * x[..., 0] + 0.71 * x[..., 1] + 0.07 * x[..., 2]

    # Verify correctness
    if torch.allclose(output_triton, expected, atol=1e-5):
        print(f"✓ Kernel test passed for {H}x{W} image!")
    else:
        print("✗ Kernel test failed!")
        diff = torch.abs(output_triton - expected).max()
        print(f"Max difference: {diff.item()}")
