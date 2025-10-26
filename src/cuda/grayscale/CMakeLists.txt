cmake_minimum_required(VERSION 3.8)

project(grayscale LANGUAGES CUDA C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(grayscale.out main.cu)
