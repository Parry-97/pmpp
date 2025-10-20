cmake_minimum_required(VERSION 3.8)

project(vector_add LANGUAGES CUDA C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(vector_add.out main.cu)
