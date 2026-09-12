#include <cstdint>

#include "cpu/matMulCpuNaive.h"
#include "cuda/matMulCudaNaive.h"
#include "matrix.h"
#include "device.h"
#include "benchmark.h"


constexpr uint32_t M = 1024;
constexpr uint32_t N = 1024;
constexpr uint32_t K = 1024;

constexpr int MIN_RANDOM_NUMBER = 0;
constexpr int MAX_RANDOM_NUMBER = 1;

constexpr uint32_t NUM_WARM_UPS = 3;
constexpr uint32_t NUM_RUNS = 20;


int main() {
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
        matMulCpuNaiveKernel, NUM_RUNS, NUM_WARM_UPS, "CPU Naive"
    );

    float gpuNaiveTime = Benchmark::benchmarkMatMul(
        aGpu, bGpu, cGpu, gtGpu,
        matMulCudaNaiveLaunch, NUM_RUNS, NUM_WARM_UPS,
        "GPU Naive", cpuNaiveTime
    );
}
