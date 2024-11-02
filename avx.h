//
// Created by surya on 2/11/2024.
//

#pragma once
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>

#pragma once
#ifndef __x86_64__
    #error "This code requires a 64-bit architecture"
#endif // __x86_64__

#ifndef AVX_H
#define AVX_H
#define _AVX_mem_aligned(_value_)    __attribute__((aligned(_value_)))

// addition of two 8 floating points vector
__m256 __attribute__((flatten)) __attribute__((nonnull(1,2)))
__avx256_immintrinsic_add(const float* __restrict__ _vector1, const float* __restrict__ _vector2) {
    const __m256 vec1 = _mm256_load_ps(_vector1);
    const __m256 vec2 = _mm256_load_ps(_vector2);
    return _mm256_add_ps(vec1, vec2);
}

// multiplication of two 8 floating points vector
__m256 __attribute__((flatten)) __attribute__((nonnull(1,2)))
__avx256_immintrinsic_mul(const float* __restrict__ _vector1, const float* __restrict__ _vector2) {
    const __m256 vec1 = _mm256_load_ps(_vector1);
    const __m256 vec2 = _mm256_load_ps(_vector2);
    return _mm256_mul_ps(vec1, vec2);
}

// summation of two 8 floating points vector
float __attribute__((flatten))
__avx256_immintrinsic_accumulate(const __m256 _vector1) {
    __m256 vec1 = _mm256_hadd_ps(_vector1, _vector1);
    vec1 = _mm256_hadd_ps(vec1, vec1);
    float temp[8] _AVX_mem_aligned(32);
    _mm256_store_ps(temp, vec1);
    return temp[0] + temp[7];
}

// multiplication + addition of two 8 floating points vector
float __attribute__((flatten)) __attribute__((nonnull(1,2)))
__avx256_immintrinsic_fused_mul_add(const float* __restrict__ _vector1, const float* __restrict__ _vector2) {
    const __m256 vec1 = _mm256_load_ps(_vector1);
    const __m256 vec2 = _mm256_load_ps(_vector2);
    const __m256 vec3 = _mm256_mul_ps(vec1, vec2);
    return __avx256_immintrinsic_accumulate(vec3);
}

float __attribute__((flatten)) __attribute__((nonnull(1,2)))
__avx256_immintrinsic_fused_vector_mul_add(const float* __restrict__ _Vector1, const float* __restrict__ _Vector2, const int _Size) {
    float result = 0;
    if(!_Vector1 || !_Vector2 || _Size <= 0) return 0.0f;
    const int end = (_Size / 8) * 8;
    for(int i = 0; i < end; i += 8) {
        if (((uintptr_t)(_Vector1 + i) % 32 == 0) && ((uintptr_t)(_Vector2 + i) % 32 == 0)) {
            result += __avx256_immintrinsic_fused_mul_add(_Vector1 + i, _Vector2 + i);
        } else {
            for (int j = 0; j < 8; j++) {
                result += _Vector1[i + j] * _Vector2[i + j];
            }
        }
    }
    for(int i = end; i < _Size; ++i) {
        result += _Vector1[i] * _Vector2[i];
    }
    return result;
}

float* __attribute__((flatten)) __attribute__((nonnull(1,2)))
__avx256_immintrinsic_vector_multiplication(const float* __restrict__ _Vector1, const float* __restrict__ _Vector2, const int _Size) {
    if(!_Vector1 || !_Vector2 || _Size <= 0) return NULL;
    float* result = NULL;
    if(posix_memalign((void**)&result, 32, sizeof(float) * _Size) != 0) {
        perror("Aligned memory allocation failed");
        return NULL;
    }
    const int end = (_Size / 8) * 8;
    for(int i = 0; i < end; i += 8) {
        if (((uintptr_t)(_Vector1 + i) % 32 == 0) && ((uintptr_t)(_Vector2 + i) % 32 == 0)) {
            const __m256 mul = __avx256_immintrinsic_mul(_Vector1 + i, _Vector2 + i);
            _mm256_store_ps(result + i, mul);
        } else {
            for (int j = 0; j < 8; j++) {
                result[i + j] = _Vector1[i + j] * _Vector2[i + j];
            }
        }
    }
    for(int i = end; i < _Size; ++i) {
        result[i] = _Vector1[i] * _Vector2[i];
    }
    return result;
}

float __attribute__((flatten)) __attribute__((nonnull(1)))
__avx256_immintrinsic_vector_accumulate(float* __restrict__ _Vector, const int _Size) {
    if(!_Vector) { perror("NULL Pointer Exception"); exit(1); }
    float sum = 0;
    const int end = (_Size / 8) * 8;
    const int rem = _Size % 8;
    for(int i = 0; i < end; i += 8) {
        const __m256 temp = _mm256_load_ps(_Vector + i);
        sum += __avx256_immintrinsic_accumulate(temp);
    }
    for(int i = end; i < end + rem; ++i) {
        sum += _Vector[i];
    }
    return sum;
}

void __attribute__((flatten)) __attribute__((nonnull(1,2,3)))
__avx256_immintrinsic_matrix_multiplication(const float* __restrict__ _Tensor1, const float* __restrict__ _Tensor2, float* __restrict__ _Dest,
    const int _Row, const int _Col, const int _ComDim) {
    for(int i = 0; i < _Row; ++i) {
        for(int j = 0; j < _Col; ++j) {
            _Dest[(i * _Col) + j] = __avx256_immintrinsic_fused_vector_mul_add(_Tensor1 + i * _ComDim, _Tensor2 + j * _ComDim, _ComDim);
        }
    }
}

void __attribute__((flatten))
__avx256_immintrinsic_matrix_transpose(const float* __restrict__ _Src, float* __restrict__ _Dest, const int _Row, const int _Col) {
    const int block_size = 16;
    for (int i = 0; i < _Row; i += block_size) {
        for (int j = 0; j < _Col; j += block_size) {
            for (int ii = i; ii < i + block_size && ii < _Row; ++ii) {
                for (int jj = j; jj < j + block_size && jj < _Col; ++jj) {
                    _Dest[jj * _Row + ii] = _Src[ii * _Col + jj];
                }
            }
        }
    }
}

void __attribute__((flatten))
__avx256_immintrinsic_random_fill(float* __restrict__ _Src, const int _Size, int _Seed) {
    _Seed++;
    for(int i = 0; i < _Size; i += 8) {
        if (((uintptr_t)(_Src + i) % 32) == 0) {
            const __m256 rand_vals = _mm256_set_ps(
                (float)(rand() % _Seed + 1),
                (float)(rand() % _Seed + 1),
                (float)(rand() % _Seed + 1),
                (float)(rand() % _Seed + 1),
                (float)(rand() % _Seed + 1),
                (float)(rand() % _Seed + 1),
                (float)(rand() % _Seed + 1),
                (float)(rand() % _Seed + 1)
            );
            _mm256_store_ps(_Src + i, rand_vals);
        } else {
            for (int j = i; j < i + 8 && j < _Size; j++) {
                _Src[j] = (float)(rand() % _Seed + 1);
            }
        }
    }
    for (int i = (_Size / 8) * 8; i < _Size; ++i) {
        _Src[i] = (float)(rand() % _Seed + 1);
    }
}

void __attribute__((flatten))
__avx256_immintrinsic_specific_fill(float* __restrict__ _Src, const int _Size, const int _Value) {
    const __m256 fill_val = _mm256_set1_ps((float)_Value);
    int i;
    for (i = 0; i <= _Size - 8; i += 8) {
        if (((uintptr_t)(_Src + i) % 32) == 0) {
            _mm256_store_ps(_Src + i, fill_val);
        } else {
            for (int j = i; j < i + 8; ++j) {
                _Src[j] = _Value;
            }
        }
    }
    for (; i < _Size; ++i) {
        _Src[i] = (float)_Value;
    }
}


#endif //AVX_H
