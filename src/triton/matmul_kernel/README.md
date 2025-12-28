# matmul_kernel

Parallel matrix multiplication with Triton

## Installation

```bash
uv sync
```

## Usage

```python
import torch
from matmul_kernel import matmul_kernel

x = torch.rand(1024, device="cuda", dtype=torch.float32)
output = matmul_kernel(x)
```

## Running Tests

```bash
uv run python matmul_kernel.py
```



## Kernel Parameters

- **BLOCK_SIZE**: 1024 (elements per program)

## Author

Param Pal Singh
