#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <ratio>

#include <cuda_runtime.h>

#include "validation.h"
#include "device.h"
#include "matrix.h"


template <typename F>
void Benchmark::warmup(const Matrix &a, const Matrix &b, Matrix &c, F matMul, uint32_t numWarmups) {
    for (uint32_t i = 0; i < numWarmups; i++) {
        matMul(a, b, c);
    }
}


template <typename F>
float Benchmark::runCpu(
    const Matrix &a,
    const Matrix &b,
    Matrix &c,
    F matMul,
    uint32_t numRuns
) {
    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < numRuns; i++) {
        matMul(a, b, c);
    }
    auto end = std::chrono::steady_clock::now();
    float durationMs = std::chrono::duration<float, std::milli>(end - start).count();
    return durationMs;
}


template <typename F>
float Benchmark::runCuda(
    const Matrix &a,
    const Matrix &b,
    Matrix &c,
    F matMul,
    uint32_t numRuns
) {
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (uint32_t i = 0; i < numRuns; i++) {
        matMul(a, b, c);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float durationMs;
    cudaEventElapsedTime(&durationMs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return durationMs;
}

template <typename F>
float Benchmark::benchmarkMatMul(
    const Matrix &a,
    const Matrix &b,
    Matrix &c,
    const Matrix &gt,
    F matMul,
    uint32_t numRuns,
    uint32_t numWarmups,
    const char *testName,
    float compareTime
) {
    validateMatMul(a, b, c);
    std::printf("Warming up with %u runs...\n", numWarmups);
    warmup(a, b, c, matMul, numWarmups);

    std::printf("Benchmarking with %u runs...\n", numRuns);

    float totalDurationMs;
    if (a.getDevice() == Device::CUDA) {
        totalDurationMs = runCuda(a, b, c, matMul, numRuns);
    } else if (a.getDevice() == Device::CPU) {
        totalDurationMs = runCpu(a, b, c, matMul, numRuns);
    } else {
        std::abort();
    }

    float avgDurationMs = totalDurationMs / numRuns;
    std::printf("%s avg duration: %.3fms\n", testName, avgDurationMs);
    std::printf("Validation of output vs ground truth: %s\n", (c == gt) ? "PASS" : "FAIL");
    if (compareTime != NO_COMPARISON) printSpeedUp(compareTime, avgDurationMs);
    printSep();
    return avgDurationMs;
}
