# CMake Guide for CUDA Projects

## What is CMake?

CMake is a **meta-build system** that generates native build files (Makefiles, Ninja files, Visual Studio projects) from a high-level, platform-independent configuration.

```
CMakeLists.txt → CMake → Build System (Make/Ninja) → Compiler → Executable
```

### Why CMake?
- **Cross-platform**: Same CMakeLists.txt works on Linux, Windows, macOS
- **Language support**: Native C/C++/CUDA support
- **Dependency management**: Built-in tools (FetchContent, find_package)
- **IDE integration**: CLion, VS Code, Visual Studio support
- **Industry standard**: Used by most C++ projects

## CMake Workflow

### 1. Configure Phase
```bash
cmake -B build -S .
```
- Reads `CMakeLists.txt`
- Finds compilers (nvcc, g++, clang)
- Resolves dependencies
- Generates build system files in `build/`

### 2. Build Phase
```bash
cmake --build build
```
- Invokes the generated build system
- Compiles source files
- Links executables/libraries

### 3. Test Phase
```bash
ctest --test-dir build
```
- Runs registered tests
- Reports results

### 4. Install Phase (Optional)
```bash
cmake --install build --prefix /usr/local
```
- Copies binaries/headers to system directories

## Basic CMakeLists.txt Structure

```cmake
# Minimum CMake version
cmake_minimum_required(VERSION 3.18)

# Project declaration (enables CUDA)
project(ProjectName LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Create library from CUDA sources
add_library(my_lib src/kernels.cu src/utils.cpp)

# Create executable
add_executable(my_app src/main.cu)
target_link_libraries(my_app my_lib)

# Testing
enable_testing()
add_executable(tests tests/test.cu)
target_link_libraries(tests my_lib)
add_test(NAME MyTests COMMAND tests)
```

## CUDA-Specific Configuration

### 1. Enable CUDA Language
```cmake
project(MyProject LANGUAGES CXX CUDA)
```

### 2. Set Compute Capabilities
```cmake
# Target specific GPU architectures
set(CMAKE_CUDA_ARCHITECTURES 75 86)  # Turing, Ampere
# Or auto-detect
set(CMAKE_CUDA_ARCHITECTURES native)
```

### 3. Separable Compilation (for device linking)
```cmake
set_target_properties(my_lib PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON)
```

### 4. Position Independent Code
```cmake
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
```

## Dependency Management

### Method 1: FetchContent (Recommended for Learning)

**Advantages**: Self-contained, no external tools, reproducible

```cmake
include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  # Or use URL for faster download
  # URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# Make gtest available
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Now use it
add_executable(tests test.cu)
target_link_libraries(tests GTest::gtest_main)

# Auto-discover tests
include(GoogleTest)
gtest_discover_tests(tests)
```

**How it works**:
1. Downloads dependency at configure time
2. Builds it as part of your project
3. Makes targets available (e.g., `GTest::gtest_main`)

### Method 2: find_package (System-installed)

```cmake
# Find system-installed package
find_package(GTest REQUIRED)

add_executable(tests test.cu)
target_link_libraries(tests GTest::gtest_main)
```

**Installation first**:
```bash
# Arch/EndeavourOS
sudo pacman -S gtest

# Ubuntu/Debian
sudo apt install libgtest-dev

# Build from source
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib
```

### Method 3: Git Submodules

```bash
# Add as submodule
git submodule add https://github.com/google/googletest.git extern/gtest
git submodule update --init --recursive
```

```cmake
# Add subdirectory
add_subdirectory(extern/gtest)

add_executable(tests test.cu)
target_link_libraries(tests gtest_main)
```

### Method 4: Conan (Advanced)

**Install Conan**:
```bash
pip install conan
```

**conanfile.txt**:
```ini
[requires]
gtest/1.14.0

[generators]
CMakeDeps
CMakeToolchain
```

**CMakeLists.txt**:
```cmake
find_package(GTest REQUIRED)
target_link_libraries(tests GTest::gtest)
```

**Build**:
```bash
conan install . --build=missing
cmake -B build -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build build
```

## Complete Example for This Project

```cmake
cmake_minimum_required(VERSION 3.18)
project(pmpp LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)  # Auto-detect GPU

# Compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")  # Debug info

# Library with CUDA kernels
add_library(pmpp_kernels
    src/ch02/vector_add.cu
    src/ch03/matrix_multiply.cu
)

# Executable
add_executable(pmpp_main src/main.cu)
target_link_libraries(pmpp_main pmpp_kernels)

# Testing with FetchContent
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test executable
add_executable(pmpp_tests
    tests/test_vector_add.cu
    tests/test_matrix_multiply.cu
)
target_link_libraries(pmpp_tests 
    pmpp_kernels
    GTest::gtest_main
)

# Auto-discover and register tests
include(GoogleTest)
gtest_discover_tests(pmpp_tests)

# CUDA error checking wrapper
target_compile_definitions(pmpp_tests PRIVATE 
    CUDA_ERROR_CHECK=1
)
```

## Build Commands

```bash
# Configure (generate build files)
cmake -B build -S .

# Build all targets
cmake --build build

# Build specific target
cmake --build build --target pmpp_tests

# Run tests
ctest --test-dir build --output-on-failure

# Verbose build
cmake --build build --verbose

# Clean build
cmake --build build --target clean

# Rebuild from scratch
rm -rf build && cmake -B build && cmake --build build
```

## Directory Structure

```
pmpp/
├── CMakeLists.txt           # Main build configuration
├── src/
│   ├── ch02/
│   │   └── vector_add.cu
│   └── main.cu
├── tests/
│   └── test_vector_add.cu
├── include/                 # Public headers
│   └── pmpp/
│       └── kernels.cuh
└── build/                   # Generated (don't commit)
    ├── CMakeCache.txt
    ├── Makefile
    └── ...
```

## Advanced Features

### 1. Organizing Multi-File Projects

```cmake
# Create interface library for headers
add_library(pmpp_interface INTERFACE)
target_include_directories(pmpp_interface INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against interface
target_link_libraries(pmpp_kernels PUBLIC pmpp_interface)
```

### 2. Conditional Compilation

```cmake
option(BUILD_TESTS "Build test suite" ON)
option(ENABLE_PROFILING "Enable CUDA profiling" OFF)

if(ENABLE_PROFILING)
    target_compile_definitions(pmpp_kernels PRIVATE ENABLE_PROFILING)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### 3. Build Types

```cmake
# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Per-config flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
```

**Usage**:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

### 4. Installing Headers and Libraries

```cmake
# Install targets
install(TARGETS pmpp_kernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY include/pmpp
    DESTINATION include
)
```

## Common Issues and Solutions

### Issue: CUDA not found
```bash
# Explicitly set CUDA path
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Issue: Wrong GPU architecture
```bash
# Set manually
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75
```

### Issue: FetchContent slow
Use `URL` instead of `GIT_REPOSITORY` for faster downloads.

### Issue: Rebuild everything on small change
Use Ninja instead of Make:
```bash
cmake -B build -G Ninja
ninja -C build
```

## CMake vs Make

| Feature | Make | CMake |
|---------|------|-------|
| **Portability** | Unix only | Cross-platform |
| **Syntax** | Complex, manual | High-level, declarative |
| **Dependencies** | Manual tracking | Automatic |
| **CUDA** | Manual nvcc flags | Native support |
| **IDE Support** | Limited | Excellent |

## Best Practices

1. **Out-of-source builds**: Always use `-B build`
2. **Version control**: Add `build/` to `.gitignore`
3. **Minimum version**: Use latest stable (3.18+ for CUDA)
4. **Target-based**: Use `target_*` commands, not global settings
5. **Dependencies**: Prefer FetchContent for learning, Conan for production
6. **Testing**: Always enable and auto-discover tests

## Resources

- [Official CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- [Modern CMake](https://cliutils.gitlab.io/modern-cmake/)
- [CMake CUDA Documentation](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)
- [Effective Modern CMake](https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1)

## Quick Reference

```bash
# Configure
cmake -B build                          # Default generator
cmake -B build -G Ninja                # Use Ninja
cmake -B build -DCMAKE_BUILD_TYPE=Debug # Debug build

# Build
cmake --build build                    # Build all
cmake --build build --target tests     # Build specific
cmake --build build -j$(nproc)        # Parallel build

# Test
ctest --test-dir build                # Run all tests
ctest --test-dir build -R vector      # Run matching tests
ctest --test-dir build --verbose      # Verbose output

# Clean
cmake --build build --target clean    # Clean build artifacts
rm -rf build                          # Nuclear option

# Install
cmake --install build --prefix ~/.local
```
