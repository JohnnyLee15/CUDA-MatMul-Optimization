#include <cstdint>
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuda/matCudaUtils.h"
#include "validation.h"
#include "matrix.h"


constexpr uint32_t BLOCK_SIZE = 256;
constexpr unsigned long long RANDOM_SEED = 42ULL;


namespace {
__global__ void matCompareKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint32_t n,
    uint32_t *mismatch
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    if (fabsf(a[i] - b[i]) > ABS_ERROR) {
        atomicExch(mismatch, 1);
    }
}


__global__ void matInitRandomKernel(
    float* __restrict__ a,
    uint32_t n,
    float min,
    float max,
    unsigned long long seed
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);

    float u = curand_uniform(&state);
    a[i] = min + u * (max - min);
}
}


bool matCompareLaunch(const Matrix &a, const Matrix &b) {
    uint32_t mismatch;
    uint32_t *d_mismatch;
    cudaMalloc(&d_mismatch, sizeof(uint32_t));
    cudaMemset(d_mismatch, 0, sizeof(uint32_t));

    float *aData = a.data();
    float *bData = b.data();
    uint32_t n = a.getSize();

    uint32_t numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matCompareKernel<<<numBlocks, BLOCK_SIZE>>>(aData, bData, n, d_mismatch);

    cudaMemcpy(&mismatch, d_mismatch, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_mismatch);

    return mismatch == 0;
}


void matInitRandomLaunch(Matrix &a, float min, float max) {
    float *aData = a.data();
    uint32_t n = a.getSize();

    uint32_t numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matInitRandomKernel<<<numBlocks, BLOCK_SIZE>>>(aData, n, min, max, RANDOM_SEED);
}
