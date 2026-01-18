# reduction_kernel

Parallel Reduction kernel with Triton

## Installation

```bash
uv sync
```

## Usage

```python
import torch
from reduction_kernel import reduction_kernel

x = torch.rand(1024, device="cuda", dtype=torch.float32)
output = reduction_kernel(x)
```

## Running Tests

```bash
uv run python reduction_kernel.py
```



## Kernel Parameters

- **BLOCK_SIZE**: 1024 (elements per program)

## Author

Param Pal Singh
