cmake_minimum_required(VERSION 3.8)

project(tiled_matmul LANGUAGES CUDA C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(tiled_matmul.out main.cu)
