//
// Created by surya on 2/11/2024.
//
#include "tensor.h"
#include "avx.h"

void matrix_create(const int _Row, const int _Col, tensor_t* _Ptr) {
    if(_Ptr == NULL) return;
    (_Ptr)->arr = malloc(sizeof(float) * _Row * _Col);
    (_Ptr)->x_dim = _Row;
    (_Ptr)->y_dim = _Col;
}

tensor_t* __aligned_creation_of_matrix(const int _Row, const int _Col) {
    tensor_t* A = malloc(sizeof(tensor_t));
    if (A == NULL) return NULL;
    if (posix_memalign((void**)&A->arr, MEM_ALIGNED_SIZE, _Row * _Col * sizeof(float)) != 0) {
        free(A);
        return NULL;
    }
    A->x_dim = _Row;
    A->y_dim = _Col;
    return A;
}

void __random_initializer(const tensor_t* A, const int seed) {
    if(A == NULL) return;
    __avx256_immintrinsic_random_fill(A->arr, A->x_dim*A->y_dim, seed);
}

void __specific_initializer(const tensor_t* A, const int value) {
    if(A == NULL) return;
    __avx256_immintrinsic_specific_fill(A->arr, A->x_dim*A->y_dim, value);
}

void __destroy_matrix(tensor_t* A) {
    if(A == NULL) return;
    if(A->arr) free(A->arr);
    free(A);
}

void __print_matrix_single(const tensor_t* A, const int precision) {
    const int row = A->x_dim;
    const int col = A->y_dim;
    const float* ptr = A->arr;
    char __format[10];
    snprintf(__format, sizeof(__format), "%%.%df ", precision);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf(__format, ptr[i * col + j]);
        }
        printf("\n");
    }
}

void print_matrix_to_file(const float* A, const int rows, const int cols, const int precision, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    char format[10];
    snprintf(format, sizeof(format), "%%.%df ", precision);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fprintf(file, format, A[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void console_matrix(const int num_memb, ...) {
    __builtin_va_list vargs;
    __builtin_va_start(vargs, num_memb);
    for(int _ii = 0; _ii < num_memb; ++_ii) {
        __print_matrix_single(__builtin_va_arg(vargs, tensor_t*), 1);
        printf("\n");
    }
    __builtin_va_end(vargs);
}

tensor_t* __aligned_matrix_transpose(const tensor_t* __restrict__ A) {
    tensor_t* C = __aligned_creation_of_matrix(A->y_dim, A->x_dim);
    __avx256_immintrinsic_matrix_transpose(A->arr, C->arr, A->x_dim, A->y_dim);
    return C;
}

void* __tensor_product_core_engine(const void* __args) {
    const engx_attr_t* vargs = __args;
    const tensor_t* A = vargs->A;
    const tensor_t* B = vargs->B;
    const tensor_t* C = vargs->C;
    const int row = vargs->end_row - vargs->start_row;
    const int col = B->x_dim;
    const int comm_dim = A->y_dim;

    const int TILE_SIZE = 1024;

    for (int i = 0; i < row; i += TILE_SIZE) {
        for (int j = 0; j < col; j += TILE_SIZE) {
            for (int k = 0; k < comm_dim; k += TILE_SIZE) {
                const int row_end = i + TILE_SIZE < row ? i + TILE_SIZE : row;
                const int col_end = j + TILE_SIZE < col ? j + TILE_SIZE : col;
                const int comm_end = k + TILE_SIZE < comm_dim ? k + TILE_SIZE : comm_dim;

                __avx256_immintrinsic_matrix_multiplication(
                    A->arr + (vargs->start_row + i) * comm_dim + k,
                    B->arr + k * col + j,
                    C->arr + (vargs->start_row + i) * C->y_dim + j,
                    row_end - i, col_end - j, comm_end - k
                );
            }
        }
    }
    return NULL;
}

void* thread_run(void* arg) {
    const engx_attr_t* data = arg;
#if (__THREAD_INFO_defined)
    printf("Running thread %lu on cpu %d\n", syscall(SYS_gettid), sched_getcpu());
#endif
    __tensor_product_core_engine(data);
    return NULL;
}


void matrix_multiply(tensor_t* A, const tensor_t* B, tensor_t* C, const int num_threads) {
    if(C == NULL || A == NULL || B == NULL) { __throw __exception("NULL Pointer Exception"); return; }
    if(A->x_dim != C->x_dim || A->y_dim != B->x_dim || B->y_dim != C->y_dim) { __throw __exception("Dimension mismatch Exception"); exit(DIMENSION_MISMATCH_ERROR); }
    tensor_t* T = __aligned_matrix_transpose(B);
    for(int i=0; i<C->x_dim*C->y_dim; ++i) {
        C->arr[i] = 0;
    }
    pthread_t threads[num_threads];
    engx_attr_t vargs[num_threads];
    const int rows_per_thread = A->x_dim / num_threads;
    for(int t=0; t<num_threads; ++t) {
        vargs[t].A = A;
        vargs[t].B = T;
        vargs[t].C = C;
        vargs[t].start_row = t * rows_per_thread;
        vargs[t].end_row = (t == num_threads - 1) ? (int)C->x_dim : (t + 1) * rows_per_thread;
        pthread_create(&threads[t], NULL, thread_run, &vargs[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
}

tensor_t* __intrinsic_tensor_product(tensor_t* A, const tensor_t* B, bool multi_processing) {
    tensor_t* res = tensor2D_allocate(A->x_dim, B->y_dim);
    const long num_cores = (multi_processing) ? sysconf(_SC_NPROCESSORS_ONLN) : 1;
    matrix_multiply(A, B, res, num_cores);
    return res;
}

void __tensor_stdout(const tensor_t* A) {
    __print_matrix_single(A, 2);
    printf("\n");
}

void __tensor_file_print(const tensor_t* A, const char* _FileName) {
    if(A == NULL) return;
    print_matrix_to_file(A->arr, A->x_dim, A->y_dim, 2, _FileName);
}
