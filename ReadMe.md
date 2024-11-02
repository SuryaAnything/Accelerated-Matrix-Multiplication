# Accelerated Matrix Multiplication for x86_64

This library offers optimized matrix multiplication routines specifically designed for high-performance computing on x86_64 architecture. Utilizing AVX (Advanced Vector Extensions) intrinsics, the library leverages SIMD (Single Instruction Multiple Data) parallelism to accelerate matrix operations. It includes memory-aligned structures, multithreading support, and AVX-specific functions to handle large matrices efficiently, making it well-suited for computationally intensive tasks.

## Key Optimization
### SIMD (Single Instruction Multiple Data): 
By using AVX intrinsics, the library performs multiple calculations in a single CPU instruction, processing up to 8 floating-point operations simultaneously in 256-bit registers. This approach is particularly effective for matrix operations where the same operation needs to be applied to large blocks of data.

### Multithreading: 
The library supports parallel execution by dividing the matrix multiplication workload among multiple threads. Each thread computes a portion of the output matrix, reducing computation time and effectively utilizing multicore processors. Thread management is handled via engx_attr_t attributes, allowing specific row ranges to be processed in parallel, minimizing contention and synchronizing at the completion of computation.

### Caching and Tiling: 
Matrix multiplication involves accessing matrix elements repeatedly, so the library uses a tiling strategy to enhance cache utilization. By dividing matrices into smaller tiles, it reduces cache misses by keeping each tile small enough to fit into the CPU cache, minimizing memory access times and ensuring that data remains close to the CPU during calculations.

### Memory Alignment: 
AVX instructions benefit from aligned memory, so tensors are allocated with alignment in mind. This improves access speeds for AVX operations and ensures that the library meets alignment requirements for SIMD calculations, providing further performance gains.

## Table of Contents
- [Files](#files)
- [Constants and Macros](#constants-and-macros)
    - [Helper Macros](#helper-macros)
    - [API Macros](#api-macros)
- [Structures](#structures)
- [Function Descriptions](#function-descriptions)
    - [Tensor Management](#tensor-management)
    - [Initialization and Filling Functions](#initialization-and-filling-functions)
    - [Matrix and Vector Operations](#matrix-and-vector-operations)

---

## Files

### `tensor.h`
Defines the `tensor_t` structure and provides function declarations for managing tensors, as well as macros for error handling and tensor operations.

### `avx.h`
Defines AVX-based functions for vector and matrix operations, including addition, multiplication, and matrix transposition.

### `tensor.c`
Contains the implementation of tensor.h

---

## Constants and Macros

### Helper Macros

- **Error Codes**
    - `MEMORY_ALLOCATION_ERROR`: Error for memory allocation failure.
    - `DIMENSION_MISMATCH_ERROR`: Error for dimension mismatch between tensors.

- **Memory and Buffer Configurations**
    - `MEM_ALIGNED_SIZE`: Memory alignment size (64 bytes).

- **Attribute Macros**
    - `__tensor_aligned(_a)`: Aligns memory to `_a` bytes.
    - `_AVX_mem_aligned(_value_)`: Aligns memory for AVX operations.

- **Utility Macros**
    - `__max(A, B)`: Returns the maximum of `A` and `B`.
    - `__min(A, B)`: Returns the minimum of `A` and `B`.

- **Error Handling**
    - `__throw`: Logs an exception with the line number.
    - `__exception(msg)`: Logs a custom exception message.

### API Macros

These macros provide an easy-to-use API for common tensor operations.

- **Tensor Creation and Deletion**
    - `tensor2D_allocate(_Row, _Col)`: Allocates a tensor of size `_Row x _Col`.
    - `tensor2D_deallocate(A)`: Deallocates memory associated with tensor `A`.

- **Initialization and Filling**
    - `tensor2D_randomfill(A, S)`: Fills tensor `A` with random values, seeded by `S`.
    - `tensor2D_fill(A, value)`: Fills tensor `A` with a specified value.

- **Matrix Operations**
    - `tensor2D_dotproduct(A, B)`: Multiplies tensor `A` by tensor `B`.
    - `tensor2D_transpose(A)`: Transposes tensor `A`.

- **Output**
    - `tensor2D_stdout(A)`: Prints tensor `A` to the standard output.
    - `tensor2D_fileout(A, filename)`: Writes tensor `A` to a file specified by `filename`.

---

## Structures

### `tensor_t`
Defines a 2D tensor structure.
- **Fields**:
    - `float* arr`: Pointer to tensor data.
    - `size_t x_dim`: Number of rows.
    - `size_t y_dim`: Number of columns.

### `engx_attr_t`
Attributes structure for multithreaded matrix multiplication.
- **Fields**:
    - `tensor_t* A`, `tensor_t* B`, `tensor_t* C`: Tensors for multiplication.
    - `int start_row`, `int end_row`: Row range for thread operation.

---

## Function Descriptions

### Tensor Management

#### `void __destroy_matrix(tensor_t* A)`
Destroys a tensor by freeing its allocated memory.

#### `tensor_t* __aligned_creation_of_matrix(int _Row, int _Col)`
Allocates an aligned 2D tensor with the specified dimensions.

#### `tensor_t* __aligned_matrix_transpose(const tensor_t* __restrict__ A)`
Performs an aligned matrix transpose on the given tensor `A`.

### Initialization and Filling Functions

#### `void __random_initializer(const tensor_t* A, const int seed)`
Randomly initializes the tensor `A` with a given seed value.

#### `void __specific_initializer(const tensor_t* A, const int value)`
Fills the tensor `A` with a specified value.

#### `void __avx256_immintrinsic_random_fill(float* __restrict__ _Src, const int _Size, int _Seed)`
Fills an array with random values using AVX, given a seed for random generation.

#### `void __avx256_immintrinsic_specific_fill(float* __restrict__ _Src, const int _Size, const int _Value)`
Fills an array with a specific value using AVX.

### Matrix and Vector Operations

#### `float __avx256_immintrinsic_accumulate(const __m256 _vector1)`
Accumulates all elements in a 256-bit AVX register and returns the result.

#### `__m256 __avx256_immintrinsic_add(const float* __restrict__ _vector1, const float* __restrict__ _vector2)`
Adds two 256-bit vectors (8 floating points each) and returns the result.

#### `__m256 __avx256_immintrinsic_mul(const float* __restrict__ _vector1, const float* __restrict__ _vector2)`
Multiplies two 256-bit vectors (8 floating points each) and returns the result.

#### `float __avx256_immintrinsic_fused_mul_add(const float* __restrict__ _vector1, const float* __restrict__ _vector2)`
Performs element-wise multiplication of two vectors followed by accumulation.

#### `float __avx256_immintrinsic_fused_vector_mul_add(const float* __restrict__ _Vector1, const float* __restrict__ _Vector2, const int _Size)`
Multiplies elements of two vectors and accumulates the result, optimized with AVX for aligned data.

#### `float* __avx256_immintrinsic_vector_multiplication(const float* __restrict__ _Vector1, const float* __restrict__ _Vector2, const int _Size)`
Multiplies two vectors and returns the resulting vector, with aligned memory allocation.

#### `float __avx256_immintrinsic_vector_accumulate(float* __restrict__ _Vector, const int _Size)`
Accumulates all elements in a vector.

#### `void __avx256_immintrinsic_matrix_multiplication(const float* __restrict__ _Tensor1, const float* __restrict__ _Tensor2, float* __restrict__ _Dest, int _Row, int _Col, int _ComDim)`
Performs matrix multiplication between `_Tensor1` and `_Tensor2`, storing the result in `_Dest`.

#### `void __avx256_immintrinsic_matrix_transpose(const float* __restrict__ _Src, float* __restrict__ _Dest, int _Row, int _Col)`
Transposes a matrix, with optimization for cache efficiency using block transposition.

---
