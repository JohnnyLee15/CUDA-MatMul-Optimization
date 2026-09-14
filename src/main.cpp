#include <cstdint>
#include <cstdio>

#include "cpu/matMulCpuNaive.h"
#include "cuda/matMulCudaNaive.h"
#include "cuda/matMulCudaSharedMem.h"
#include "cuda/matMulCuda1DRegTile.h"
#include "cuda/matMulCuda2DRegTile.h"
#include "matrix.h"
#include "device.h"
#include "benchmark.h"


constexpr uint32_t M = 1024;
constexpr uint32_t N = 1024;
constexpr uint32_t K = 1024;

constexpr int MIN_RANDOM_NUMBER = 0;
constexpr int MAX_RANDOM_NUMBER = 1;

constexpr uint32_t NUM_WARM_UPS_CPU = 0;
constexpr uint32_t NUM_WARM_UPS_CUDA = 3;

constexpr uint32_t NUM_RUNS_CPU = 1;
constexpr uint32_t NUM_RUNS_CUDA = 20;


namespace {
constexpr char CLEAR_SCREEN_CHARS[] = "\033[2J\033[H";
void clearScreen() {
    printf(CLEAR_SCREEN_CHARS);
}


void printBenchmarkSize() {
    printf(
        "Matrix Size: A(%d x %d) x "
        "B(%d x %d) "
        "-> "
        "C(%d x %d)\n\n",
        M, K, K, N, M, N
    );
}
}

int main() {
    clearScreen();
    printBenchmarkSize();

    Matrix aCpu(M, K, Device::CPU);
    Matrix bCpu(K, N, Device::CPU);
    Matrix cCpu(M, N, Device::CPU);
    Matrix gtCpu(M, N, Device::CPU);

    aCpu.initRandom(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
    bCpu.initRandom(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
    matMulCpuNaiveKernel(aCpu, bCpu, gtCpu);

    Matrix aGpu = aCpu.to(Device::CUDA);
    Matrix bGpu = bCpu.to(Device::CUDA);
    Matrix gtGpu = gtCpu.to(Device::CUDA);
    Matrix cGpu = Matrix(M, N, Device::CUDA);

    float cpuNaiveTime = Benchmark::benchmarkMatMul(
        aCpu, bCpu, cCpu, gtCpu,
        matMulCpuNaiveKernel, NUM_RUNS_CPU, NUM_WARM_UPS_CPU, "CPU Naive"
    );

    float gpuNaiveTime = Benchmark::benchmarkMatMul(
        aGpu, bGpu, cGpu, gtGpu,
        matMulCudaNaiveLaunch, NUM_RUNS_CUDA, NUM_WARM_UPS_CUDA,
        "GPU Naive", cpuNaiveTime, "CPU Naive"
    );

    float gpuSharedMemTime = Benchmark::benchmarkMatMul(
        aGpu, bGpu, cGpu, gtGpu,
        matMulCudaSharedMemLaunch, NUM_RUNS_CUDA, NUM_WARM_UPS_CUDA,
        "GPU Shared Memory", gpuNaiveTime, "GPU Naive"
    );

    float gpu1DRegTileTime = Benchmark::benchmarkMatMul(
        aGpu, bGpu, cGpu, gtGpu,
        matMulCuda1DRegTileLaunch, NUM_RUNS_CUDA, NUM_WARM_UPS_CUDA,
        "GPU 1D Register Tile", gpuSharedMemTime, "GPU Shared Memory"
    );


    float gpu2DRegTileTime = Benchmark::benchmarkMatMul(
        aGpu, bGpu, cGpu, gtGpu,
        matMulCuda2DRegTileLaunch, NUM_RUNS_CUDA, NUM_WARM_UPS_CUDA,
        "GPU 2D Register Tile", gpu1DRegTileTime, "GPU 1D Register Tile"
    );
}
