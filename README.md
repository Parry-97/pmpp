# PMPP: Programming Massively Parallel Processors

[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-3.18%2B-064F8C?logo=cmake)](https://cmake.org/)
[![Progress](https://img.shields.io/badge/Progress-Chapter%203-blue)](https://github.com/yourusername/pmpp)

A learning repository following the book **"Programming Massively Parallel Processors: A Hands-on Approach"** by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. This project explores parallel programming through both CUDA C/C++ and Python/Triton implementations.

## 📚 About

This repository documents my journey learning GPU programming and parallel computing. I'm experimenting with:

- **CUDA C/C++** for low-level GPU programming
- **Triton** for high-level, Pythonic GPU kernels
- **CMake** for C/C++ build management
- **uv** for Python dependency management
- **Doxygen** for C/C++ code documentation
- **jj (Jujutsu)** for version control

**Current Progress:** Chapter 3

> **Note:** This is an experimental learning repository. Code may not be production-ready and is intended for educational purposes.

## 📂 Project Structure

```
pmpp/
├── src/
│   ├── cuda/              # CUDA C/C++ implementations
│   │   └── vector_add/    # Chapter 2-3: Vector addition example
│   └── triton/            # Python/Triton implementations (coming soon)
├── notes/                 # Chapter summaries and learning notes
├── html/                  # Doxygen-generated documentation
├── CMakeLists.txt         # CMake configuration (if using top-level build)
├── pyproject.toml         # Python project configuration
├── uv.lock                # Locked Python dependencies
├── Doxyfile               # Doxygen configuration
└── README.md              # This file
```

### Directory Purposes

- **`src/cuda/`**: Contains CUDA C/C++ kernel implementations organized by chapter/topic
- **`src/triton/`**: Will contain Python/Triton kernel implementations for comparison
- **`notes/`**: Personal notes, chapter summaries, and key concepts
- **`html/`**: Auto-generated API documentation (gitignored, generated locally)

## 🔧 Prerequisites

### Hardware

- NVIDIA GPU with CUDA support (Compute Capability 3.5+)
- Check your GPU: `nvidia-smi`

### Software

- **CUDA Toolkit** (≥11.0 recommended) - [Installation Guide](https://developer.nvidia.com/cuda-downloads)
- **CMake** (≥3.18) - For building C/C++ projects
- **Python** (≥3.11) - For Triton implementations
- **uv** - Modern Python package manager ([Installation](https://github.com/astral-sh/uv))
- **Doxygen** (optional) - For generating C/C++ documentation
- **jj (Jujutsu)** (optional) - Version control ([Installation](https://github.com/martinvonz/jj))

### Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

## 🚀 Getting Started

### 1. Clone the Repository

Using jj:

```bash
jj git clone <repository-url>
cd pmpp
```

Or with git:

```bash
git clone <repository-url>
cd pmpp
```

### 2. Python Setup with uv

```bash
# Install dependencies (Triton ≥3.5.0)
uv sync

# Verify installation
uv run python -c "import triton; print(triton.__version__)"
```

### 3. Building CUDA C/C++ Projects

Each CUDA project has its own CMakeLists.txt. Navigate to the project directory:

```bash
# Example: Building vector_add
cd src/cuda/vector_add
cmake -B build
cmake --build build

# Run the executable
./build/vector_add.out
```

For a cleaner workflow, you can also use:

```bash
cd src/cuda/vector_add
cmake .
make
./vector_add.out
```

## 🏃 Running Examples

### CUDA C/C++ Examples

```bash
cd src/cuda/vector_add
cmake -B build && cmake --build build
./build/vector_add.out
```

### Python/Triton Examples (Coming Soon)

```bash
uv run python src/triton/example.py
```

## 📖 Documentation

### Generating C/C++ API Documentation

```bash
# Generate HTML documentation
doxygen Doxyfile

# View in browser
firefox html/index.html
# or
xdg-open html/index.html
```

The Doxygen configuration parses inline comments in CUDA source files to generate comprehensive API documentation.

## 🌿 Version Control with jj

This project uses **jj (Jujutsu)** instead of traditional git. Basic commands:

```bash
# Create a new change
jj describe          # Add commit description

# View history
jj log               # View commit graph

# Create new change
jj commit            # Finalize current change

# Sync with remote
jj git push          # Push to git remote
jj git fetch         # Fetch from git remote
```

**New to jj?** Check out the [Jujutsu Tutorial](https://martinvonz.github.io/jj/latest/tutorial/)

## 📚 Learning Resources

### Primary Resource

- **Book:** [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) (4th Edition recommended)

### Documentation

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Triton Documentation](https://triton-lang.org/)
- [OpenAI Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Supplementary Materials

- [CUDA by Example](https://developer.nvidia.com/cuda-example)
- [GPU Gems Series](https://developer.nvidia.com/gpugems/gpugems/contributors)
- Chapter notes available in the `notes/` directory

## 🎯 Learning Goals

- ✅ Understand GPU architecture and memory hierarchies
- ✅ Master CUDA programming fundamentals (kernels, threads, blocks, grids)
- 🔄 Learn advanced optimization techniques (memory coalescing, shared memory)
- 🔄 Explore Triton for high-level GPU programming
- 🔜 Compare CUDA and Triton approaches
- 🔜 Implement real-world parallel algorithms

**Legend:** ✅ Completed | 🔄 In Progress | 🔜 Upcoming

## 🗺️ Chapter Progress

| Chapter | Topic                                 | CUDA C/C++ | Triton | Notes |
| ------- | ------------------------------------- | ---------- | ------ | ----- |
| 1       | Introduction                          | ✅         | -      | ✅    |
| 2       | Heterogeneous Data Parallel Computing | ✅         | 🔜     | ✅    |
| 3       | Multidimensional Grids and Data       | 🔄         | 🔜     | 🔄    |
| 4       | Compute Architecture and Scheduling   | 🔜         | 🔜     | 🔜    |
| ...     | ...                                   | ...        | ...    | ...   |

## 🤝 Contributing

This is a personal learning repository, but suggestions and corrections are welcome! Feel free to:

- Open issues for questions or clarifications
- Submit pull requests for bug fixes
- Share alternative implementations

## 📝 License

This project is for educational purposes. Code implementations are based on exercises and examples from "Programming Massively Parallel Processors."

For academic use, please cite the original book:

```
Hwu, W., Kirk, D., & El Hajj, I. (2022).
Programming Massively Parallel Processors: A Hands-on Approach (4th ed.).
Morgan Kaufmann.
```

---

**Built with:** 🚀 CUDA • 🐍 Python • ⚡ Triton • 🛠️ CMake • 📦 uv • 📚 Doxygen • 🌿 jj

_Happy parallel programming! 🎉_
