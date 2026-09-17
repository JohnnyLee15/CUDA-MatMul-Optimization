#include <cstdint>

#include <cuda_runtime.h>

#include "cuda/matMulCudaVectorizedGeneral.h"
#include "matrix.h"
#include "cudaCheck.h"


constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t NUM_THREADS = BLOCK_X * BLOCK_Y;

constexpr uint32_t ROWS_PER_THREAD = 4;
constexpr uint32_t COLS_PER_THREAD = 4;

constexpr uint32_t TILE_M = BLOCK_Y * ROWS_PER_THREAD;
constexpr uint32_t TILE_N = BLOCK_X * COLS_PER_THREAD;
constexpr uint32_t TILE_K = 16;

constexpr uint32_t FLOATS_PER_FLOAT4 = 4;
constexpr uint32_t FLOATS_PER_LOAD_PASS = NUM_THREADS * FLOATS_PER_FLOAT4;

constexpr uint32_t A_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_K;
constexpr uint32_t B_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_N;

constexpr uint32_t BYTES_PER_FLOAT4 = 16;


namespace {
__global__ void matMulCudaVectorizedGeneralKernel (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ __align__(BYTES_PER_FLOAT4) float aTile[TILE_K][TILE_M];
    __shared__ __align__(BYTES_PER_FLOAT4) float bTile[TILE_K][TILE_N];

    const uint32_t ty = threadIdx.y;
    const uint32_t tx = threadIdx.x;
    const uint32_t tid = ty * BLOCK_X + tx;
    const uint32_t threadStartIdx = tid * FLOATS_PER_FLOAT4;

    const uint32_t aTileYStart = threadStartIdx / TILE_K;
    const uint32_t bTileYStart = threadStartIdx / TILE_N;

    const uint32_t aTileX = threadStartIdx % TILE_K;
    const uint32_t bTileX = threadStartIdx % TILE_N;

    const uint32_t ayBlockStart = blockIdx.y * TILE_M;
    const uint32_t bxBlockStart = blockIdx.x * TILE_N;

    const uint32_t cyStart = ayBlockStart + ty * ROWS_PER_THREAD;
    const uint32_t cxStart = bxBlockStart + tx * COLS_PER_THREAD;

    __align__(BYTES_PER_FLOAT4) float aReg[ROWS_PER_THREAD];
    __align__(BYTES_PER_FLOAT4) float bReg[COLS_PER_THREAD];
    __align__(BYTES_PER_FLOAT4) float acc[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};

    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_M; tileRowOffset += A_TILE_ROW_STRIDE) {
            uint32_t aTileY = tileRowOffset + aTileYStart;
            uint32_t ay = ayBlockStart + aTileY;
            uint32_t ax = tileStart + aTileX;

            if (ay < m && (ax + FLOATS_PER_FLOAT4) <= k && (ay * k + ax) % FLOATS_PER_FLOAT4 == 0)  {
                const float4 toLoad = *reinterpret_cast<const float4*>(&a[ay * k + ax]);
                aTile[aTileX + 0][aTileY] = toLoad.x;
                aTile[aTileX + 1][aTileY] = toLoad.y;
                aTile[aTileX + 2][aTileY] = toLoad.z;
                aTile[aTileX + 3][aTileY] = toLoad.w;

            } else {
                #pragma unroll
                for (uint32_t p = 0; p < FLOATS_PER_FLOAT4; p++) {
                    aTile[aTileX + p][aTileY] = (ay < m && (ax + p) < k) ? a[ay * k + (ax + p)] : 0.0f;
                }
            }
        }

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_K; tileRowOffset += B_TILE_ROW_STRIDE) {
            uint32_t bTileY = tileRowOffset + bTileYStart;
            uint32_t by = tileStart + bTileY;
            uint32_t bx = bxBlockStart + bTileX;

            if (by < k && (bx + FLOATS_PER_FLOAT4) <= n && (by * n + bx) % FLOATS_PER_FLOAT4 == 0)  {
                *reinterpret_cast<float4*>(&bTile[bTileY][bTileX]) = *reinterpret_cast<const float4*>(&b[by * n + bx]);
            } else {
                #pragma unroll
                for (uint32_t p = 0; p < FLOATS_PER_FLOAT4; p++) {
                    bTile[bTileY][bTileX + p] = (by < k && (bx + p) < n) ? b[by * n + (bx + p)] : 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t dotIdx = 0; dotIdx < TILE_K; dotIdx++) {

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i += FLOATS_PER_FLOAT4) {
                *reinterpret_cast<float4*>(&aReg[i]) = *reinterpret_cast<float4*>(&aTile[dotIdx][ty* ROWS_PER_THREAD + i]);
            }

            #pragma unroll
            for (uint32_t j = 0; j < COLS_PER_THREAD; j += FLOATS_PER_FLOAT4) {
                *reinterpret_cast<float4*>(&bReg[j]) = *reinterpret_cast<float4*>(&bTile[dotIdx][tx * COLS_PER_THREAD + j]);
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
        for (uint32_t j = 0; j < COLS_PER_THREAD; j += FLOATS_PER_FLOAT4) {
            uint32_t cx = cxStart + j;

            if (cy < m && (cx + FLOATS_PER_FLOAT4) <= n && (cy * n + cx) % FLOATS_PER_FLOAT4 == 0) {
                *reinterpret_cast<float4*>(&c[cy * n + cx]) = *reinterpret_cast<float4*>(&acc[i][j]);
            } else {
                #pragma unroll
                for (uint32_t p = 0; p < FLOATS_PER_FLOAT4; p++) {
                    if (cy < m && (cx + p) < n) c[cy * n + (cx + p)] = acc[i][j + p];
                }
            }
        }
    }
}
}


void matMulCudaVectorizedGeneralLaunch(const Matrix &a, const Matrix &b, Matrix &c) {
    uint32_t m = c.nrows();
    uint32_t n = c.ncols();
    uint32_t k = a.ncols();

    uint32_t gridY = (m + TILE_M - 1) / TILE_M;
    uint32_t gridX = (n + TILE_N - 1) / TILE_N;
    dim3 gridDim(gridX, gridY);

    dim3 blockDim(BLOCK_X, BLOCK_Y);

    matMulCudaVectorizedGeneralKernel<<<gridDim, blockDim>>>(a.data(), b.data(), c.data(), m, n, k);
    CUDA_CHECK(cudaGetLastError());
}

