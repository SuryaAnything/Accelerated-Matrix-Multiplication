#include <stdio.h>
#include "tensor.h"

int main(void) {
    size_t x_dim = 1024;
    size_t comm_dim = 1024;
    size_t y_dim = 1024;
    struct timespec start, end;
    clock_gettime(1, &start);
    tensor_t* A = tensor2D_allocate(x_dim, comm_dim);
    tensor_t* B = tensor2D_allocate(comm_dim, y_dim);
    tensor2D_randomfill(A, 10);
    tensor2D_randomfill(B, 10);
    tensor_t* C = tensor2D_dotproduct(A, B);
    clock_gettime(1, &end);
    const double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Multiplied matrix of dimensions (%zu x %zu) and (%zu x %zu) in %lf seconds", x_dim, comm_dim, comm_dim, y_dim, time_taken);
}
