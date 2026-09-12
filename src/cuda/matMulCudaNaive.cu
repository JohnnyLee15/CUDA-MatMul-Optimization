#include <cstdint>

#include <cuda_runtime.h>

#include "matrix.h"
#include "cuda/matMulCudaNaive.h"


constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t BLOCK_X = 16;


namespace{
__global__ void matMulCudaNaiveKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= m || x >= n) return;

    float acc = 0.0f;
    for (int i = 0; i < k; i++) {
        acc += a[y * k + i] * b[i * n + x];
    }

    c[y * n + x] = acc;
}
}


void matMulCudaNaiveLaunch(const Matrix &a, const Matrix &b, Matrix &c) {
    uint32_t m = a.nrows();
    uint32_t k = a.ncols();
    uint32_t n = b.ncols();

    uint32_t gridY = (m + BLOCK_Y - 1) / BLOCK_Y;
    uint32_t gridX = (n + BLOCK_X - 1) / BLOCK_X;
    dim3 gridDims(gridX, gridY);

    dim3 blockDims(BLOCK_X, BLOCK_Y);

    matMulCudaNaiveKernel<<<gridDims, blockDims>>>(a.data(), b.data(), c.data(), m, n, k);
}
