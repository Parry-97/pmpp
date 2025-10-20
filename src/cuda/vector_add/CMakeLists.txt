cmake_minimum_required(VERSION 3.8)

project(cuda_hello LANGUAGES CUDA C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cuda_hello main.cu)
