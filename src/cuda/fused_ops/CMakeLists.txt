cmake_minimum_required(VERSION 3.8)

project(fused_ops LANGUAGES CUDA C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(fused_ops.out main.cu)
