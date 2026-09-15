#include <cstdint>

#include <cuda_runtime.h>

#include "cuda/matMulCuda2DRegTile.h"
#include "matrix.h"
#include "cudaCheck.h"


constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t NUM_THREADS = BLOCK_Y * BLOCK_X;
constexpr uint32_t ROWS_PER_THREAD = 4;
constexpr uint32_t COLS_PER_THREAD = 4;

constexpr uint32_t TILE_M = BLOCK_Y * ROWS_PER_THREAD;
constexpr uint32_t TILE_N = BLOCK_X * COLS_PER_THREAD;
constexpr uint32_t TILE_K = 16;
constexpr uint32_t A_TILE_ROW_STRIDE = NUM_THREADS / TILE_K;
constexpr uint32_t B_TILE_ROW_STRIDE = NUM_THREADS / TILE_N;


namespace {
__global__ void matMulCuda2DRegTileKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ float aTile[TILE_M][TILE_K];
    __shared__ float bTile[TILE_K][TILE_N];

    uint32_t ty = threadIdx.y;
    uint32_t tx = threadIdx.x;
    uint32_t tid = ty * BLOCK_X + tx;

    uint32_t aTileYStart = tid / TILE_K;
    uint32_t aTileX = tid % TILE_K;

    uint32_t bTileYStart = tid / TILE_N;
    uint32_t bTileX = tid % TILE_N;

    uint32_t cyStart = blockIdx.y * TILE_M + ty * ROWS_PER_THREAD;
    uint32_t cxStart = blockIdx.x * TILE_N + tx * COLS_PER_THREAD;

    float aReg[ROWS_PER_THREAD];
    float bReg[COLS_PER_THREAD];
    float acc[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};

    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {

        uint32_t ayBlockStart = blockIdx.y * TILE_M;
        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_M; tileRowOffset += A_TILE_ROW_STRIDE) {
            uint32_t aTileY = tileRowOffset + aTileYStart;
            uint32_t ay = ayBlockStart + aTileY;
            uint32_t ax = tileStart + aTileX;
            aTile[aTileY][aTileX] = (ay < m && ax < k) ? a[ay * k + ax] : 0.0f;
        }

        uint32_t bxBlockStart = blockIdx.x * TILE_N;
        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_K; tileRowOffset += B_TILE_ROW_STRIDE) {
            uint32_t bTileY = tileRowOffset + bTileYStart;
            uint32_t by = tileStart + bTileY;
            uint32_t bx = bxBlockStart + bTileX;
            bTile[bTileY][bTileX] = (by < k && bx < n) ? b[by * n + bx] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t dotIdx = 0; dotIdx < TILE_K; dotIdx++) {

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
                aReg[i] = aTile[ty * ROWS_PER_THREAD + i][dotIdx];
            }

            #pragma unroll
            for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
                bReg[j] = bTile[dotIdx][tx * COLS_PER_THREAD + j];
            }

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {

                #pragma unroll
                for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }

        }

        __syncthreads();
    }

    #pragma unroll
    for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
        uint32_t cy = cyStart + i;

        #pragma unroll
        for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
            uint32_t cx = cxStart + j;

            if (cy < m && cx < n) {
                c[cy * n + cx] = acc[i][j];
            }
        }
    }
}
}


void matMulCuda2DRegTileLaunch(const Matrix &a, const Matrix &b, Matrix &c) {
    uint32_t m = c.nrows();
    uint32_t n = c.ncols();
    uint32_t k = a.ncols();

    uint32_t gridY = (m + TILE_M - 1) / TILE_M;
    uint32_t gridX = (n + TILE_N - 1) / TILE_N;
    dim3 gridDim(gridX, gridY);

    dim3 blockDim(BLOCK_X, BLOCK_Y);

    matMulCuda2DRegTileKernel<<<gridDim, blockDim>>>(a.data(), b.data(), c.data(), m, n, k);
    CUDA_CHECK(cudaGetLastError());
}