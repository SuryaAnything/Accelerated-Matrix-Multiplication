cmake_minimum_required(VERSION 3.22)
project(Matrix_Mult C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_COMPILER gcc)
add_compile_options(-mavx2)
add_executable(Matrix_Mult main.c
        tensor.h
        tensor.c
        avx.h)
target_compile_options(Matrix_Mult PRIVATE -O3)