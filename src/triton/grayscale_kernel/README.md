# grayscale_kernel

Parallel implementation of grayscale filter in Triton

## Installation

```bash
uv sync
```

## Usage

```python
import torch
from grayscale_kernel import grayscale_kernel

x = torch.rand(1024, device="cuda", dtype=torch.float32)
output = grayscale_kernel(x)
```

## Running Tests

```bash
uv run python grayscale_kernel.py
```



## Kernel Parameters

- **BLOCK_SIZE**: 1024 (elements per program)

## Author

Param Pal Singh
