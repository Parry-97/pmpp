cmake_minimum_required(VERSION 3.8)

project(matmul LANGUAGES CUDA C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(matmul.out main.cu)
