#include <cstdint>

#include <cuda_runtime.h>

#include "cuda/matMulCudaSharedMem.h"
#include "matrix.h"
#include "cudaCheck.h"


constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t TILE_K = 16;


namespace {
__global__ void matMulCudaSharedMemKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ float aTile[BLOCK_Y][TILE_K];
    __shared__ float bTile[TILE_K][BLOCK_X];

    uint32_t ty = threadIdx.y;
    uint32_t tx = threadIdx.x;

    uint32_t cy = blockIdx.y * blockDim.y + ty;
    uint32_t cx = blockIdx.x * blockDim.x + tx;

    float acc = 0.0f;
    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {

        if (tx < TILE_K) {
            uint32_t ax = tileStart + tx;
            aTile[ty][tx] = (cy < m && ax < k) ? a[cy * k + ax] : 0.0f;
        }

        if (ty < TILE_K) {
            uint32_t by = tileStart + ty;
            bTile[ty][tx] = (by < k && cx < n) ? b[by * n + cx] : 0.0f;
        }

        __syncthreads();

        for (uint32_t i = 0; i < TILE_K; i++) {
            acc += aTile[ty][i] * bTile[i][tx];
        }

        __syncthreads();
    }

    if (cy < m && cx < n) {
        c[cy * n + cx] = acc;
    }
}
}


void matMulCudaSharedMemLaunch(const Matrix &a, const Matrix &b, Matrix &c) {
    uint32_t m = c.nrows();
    uint32_t n = c.ncols();
    uint32_t k = a.ncols();

    uint32_t gridY = (m + BLOCK_Y - 1) / BLOCK_Y;
    uint32_t gridX = (n + BLOCK_X - 1) / BLOCK_X;
    dim3 gridDim(gridX, gridY);

    dim3 blockDim(BLOCK_X, BLOCK_Y);

    matMulCudaSharedMemKernel<<<gridDim, blockDim>>>(a.data(), b.data(), c.data(), m, n, k);
    CUDA_CHECK(cudaGetLastError());
}
