# Simple CUDA CMake Project

This is a simple example of a CUDA project that can be built using CMake.

## Prerequisites

Before you begin, ensure you have the following installed:

- A C++ compiler (like g++)
- The NVIDIA CUDA Toolkit (which includes the `nvcc` compiler)
  - **On Arch-based systems:**

    ```bash
    sudo pacman -S cuda
    ```

- CMake (version 3.8 or higher)

If you do not have CMake installed, you can typically install it using your system's package manager.

For example:

- **On Debian-based systems (like Ubuntu):**

  ```bash
  sudo apt-get update
  sudo apt-get install cmake
  ```

- **On Fedora/CentOS/RHEL:**

  ```bash
  sudo dnf install cmake
  ```

- **On Arch-based systems:**

  ```bash
  sudo pacman -S cmake
  ```

## Building the Project

To build the project, follow these steps:

1. **Create a build directory:**

    ```bash
    mkdir build
    ```

2. **Navigate into the build directory:**

    ```bash
    cd build
    ```

3. **Run CMake to generate the build files:**

    ```bash
    cmake ..
    ```

4. **Compile the project using make:**

    ```bash
    make
    ```

## Running the Project

After the build is complete, an executable named `cuda_hello` will be located in the `build` directory. You can run it with:

```bash
./cuda_hello
```

You should see the following output:

```
Hello from the CPU!
Hello from the GPU!
```
