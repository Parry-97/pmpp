cmake_minimum_required(VERSION 3.8)

project(reduction LANGUAGES CUDA C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(reduction.out main.cu)
