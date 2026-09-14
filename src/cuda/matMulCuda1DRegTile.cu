#include <cstdint>

#include <cuda_runtime.h>

#include "cuda/matMulCuda1DRegTile.h"
#include "matrix.h"
#include "cudaCheck.h"


constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t ROWS_PER_THREAD = 8;

constexpr uint32_t TILE_M = BLOCK_Y * ROWS_PER_THREAD;
constexpr uint32_t TILE_K = 16;


namespace {
__global__ void matMulCuda1DRegTileKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ float aTile[TILE_M][TILE_K];
    __shared__ float bTile[TILE_K][BLOCK_X];

    uint32_t ty = threadIdx.y;
    uint32_t tx = threadIdx.x;

    uint32_t cyStart = blockIdx.y * TILE_M + ty * ROWS_PER_THREAD;
    uint32_t cx = blockIdx.x * blockDim.x + tx;

    float acc[ROWS_PER_THREAD] = {0.0f};

    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {

        // load aTile
        if (tx < TILE_K) {
            uint32_t aTileYStart = ty * ROWS_PER_THREAD;
            uint32_t ax = tileStart + tx;

            for (uint32_t r = 0; r < ROWS_PER_THREAD; r++) {
                uint32_t aTileY = aTileYStart + r;
                uint32_t ay = cyStart + r;
                aTile[aTileY][tx] = (ay < m && ax < k) ? a[ay * k + ax] : 0.0f;
            }
        }

        // load bTile
        if (ty < TILE_K) {
            uint32_t by = tileStart + ty;
            bTile[ty][tx] = (by < k && cx < n) ? b[by * n + cx] : 0.0f;
        }

        __syncthreads();

        // compute partial dot product
        #pragma unroll
        for (uint32_t i = 0; i < TILE_K; ++i) {
            float bVal = bTile[i][tx];

            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_THREAD; ++r) {
                acc[r] += aTile[ty * ROWS_PER_THREAD + r][i] * bVal;
            }
        }

        __syncthreads();
    }

    // write output values
    #pragma unroll
    for (uint32_t r = 0; r < ROWS_PER_THREAD; r++) {
        uint32_t cy = cyStart + r;
        if (cy < m && cx < n) {
            c[cy * n + cx] = acc[r];
        }
    }
}
}

void matMulCuda1DRegTileLaunch(const Matrix &a, const Matrix &b, Matrix &c) {
    uint32_t m = c.nrows();
    uint32_t n = c.ncols();
    uint32_t k = a.ncols();

    uint32_t gridY = (m + TILE_M - 1) / TILE_M;
    uint32_t gridX = (n + BLOCK_X - 1) / BLOCK_X;
    dim3 gridDim(gridX, gridY);

    dim3 blockDim(BLOCK_X, BLOCK_Y);

    matMulCuda1DRegTileKernel<<<gridDim, blockDim>>>(a.data(), b.data(), c.data(), m, n, k);
    CUDA_CHECK(cudaGetLastError());
}
