#pragma once

#include <cstdint>

#include <cuda_runtime.h>


void cudaCheckImpl(
    cudaError_t error,
    const char *expression,
    const char *fileName,
    uint32_t line
);


#define CUDA_CHECK(expr) \
    cudaCheckImpl((expr), #expr, __FILE__, __LINE__)
