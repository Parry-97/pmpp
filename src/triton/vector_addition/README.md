# Vector Addition

Parallel vector addition implementation in Triton

## Installation

```bash
uv sync
```

## Usage

```python
import torch
from vector_addition import vector_addition

x = torch.zeros(1024, dtype=torch.float32)
y = torch.ones(1024, dtype=torch.float32)

# Run the kernel
output = vector_addition(x, y)
```

## Running Tests

```bash
uv run python vector_addition.py
# If you don't have a GPU and want to run it on CPU
TRITON_INTERPRET=1 uv run python vector_addition.py
```

## Kernel Parameters

- **BLOCK_SIZE**: 1024 (elements per program)

## Author

Param Pal Singh
