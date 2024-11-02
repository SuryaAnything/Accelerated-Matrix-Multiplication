//
// Created by surya on 2/11/2024.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>
#include <semaphore.h>
#include <stdint.h>

#define MEMORY_ALLOCATION_ERROR     1
#define DIMENSION_MISMATCH_ERROR    2
#define MEM_ALIGNED_SIZE            64
#define __THREAD_INFO_defined       0
#define _State_INFO                 0
#define __tensor_aligned(_a)        __attribute__((aligned(_a)))
#define MAX_THREAD_EXTENSION        0
#define __max(A, B)                 (A > B) ? A : B
#define __min(A, B)                 (A < B) ? A : B

// exception handling
#define __throw             fprintf(stderr, "Exception in Line %d\n",__LINE__);
#define __exception(msg)    fprintf(stderr, "%s\n", msg);

typedef struct _tensor {
    float* arr;
    size_t x_dim;
    size_t y_dim;
} __tensor_aligned(64) tensor_t;

typedef struct __engx_attr {
    tensor_t* A;
    tensor_t* B;
    tensor_t* C;
    int start_row;
    int end_row;
} engx_attr_t;

void __random_initializer(const tensor_t* A, const int seed);
void __specific_initializer(const tensor_t* A, const int value);
void __destroy_matrix(tensor_t* A);
void __print_matrix_single(const tensor_t* A, const int precision);
void* __tensor_product_core_engine(const void* __args);
tensor_t* __intrinsic_tensor_product(tensor_t* A, const tensor_t* B, bool multi_processing);
void* thread_run(void* arg);
void console_matrix(int num_memb, ...);
void matrix_multiply(tensor_t* A, const tensor_t* B, tensor_t* C, int num_threads);
void matrix_create(int _Row, int _Col, tensor_t* _Ptr);
tensor_t* __aligned_creation_of_matrix(int _Row, int _Col);
tensor_t* __aligned_matrix_transpose(const tensor_t* __restrict__ A);
void __tensor_file_print(const tensor_t* A, const char* _FileName);
void __tensor_stdout(const tensor_t* A);
void print_matrix_to_file(const float* A, int rows, int cols, int precision, const char* filename);
int sched_getcpu(void);

#ifndef _MAT_MUL_
#define _MAT_MUL_

#define tensor2D_deallocate(A)                        __destroy_matrix(A)
#define tensor2D_randomfill(A, S)                    __random_initializer(A, S)
#define tensor2D_dotproduct(A, B)                     __intrinsic_tensor_product(A, B, true)
#define tensor2D_allocate(_Row, _Col)                 __aligned_creation_of_matrix(_Row, _Col)
#define tensor2D_fileout(A, filename)                 __tensor_file_print(A, filename)
#define tensor2D_stdout(A)                            __tensor_stdout(A)
#define tensor2D_transpose(A)                         __aligned_matrix_transpose(A)
#define tensor2D_fill(A, value)                       __specific_initializer(A, value)

#endif // _MAT_MUL_

#endif //TENSOR_H
