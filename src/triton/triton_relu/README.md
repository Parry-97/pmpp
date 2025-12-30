# triton_relu

ReLU Activation with Triton

## Installation

```bash
uv sync
```

## Usage

```python
import torch
from relu_kernel import relu_kernel

x = torch.rand(1024, device="cuda", dtype=torch.float32)
output = relu_kernel(x)
```

## Running Tests

```bash
uv run python relu_kernel.py
```



## Kernel Parameters

- **BLOCK_SIZE**: 1024 (elements per program)

## Author

Param Pal Singh
