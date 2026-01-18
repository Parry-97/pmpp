cmake_minimum_required(VERSION 3.8)

project(parallel_scan LANGUAGES CUDA C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(parallel_scan.out main.cu)
