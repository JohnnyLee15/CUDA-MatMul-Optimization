#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <benchmark.h>
#include <matrix.h>
#include <device.h>
#include <cuda/matMulCudaNaive.h>
#include <matMulAutotune.h>

constexpr uint32_t NUM_ARGS = 4;

constexpr uint32_t NUM_WARM_UPS = 3;
constexpr uint32_t NUM_RUNS = 20;

constexpr int MIN_RANDOM_NUMBER = 0;
constexpr int MAX_RANDOM_NUMBER = 1;


bool parseAndValidateDim(const char *input, uint32_t &dim) {
    const char *end = input + std::strlen(input);
    auto result = std::from_chars(input, end, dim);

    return result.ec == std::errc{} &&
           result.ptr == end &&
           dim > 0;
}


int main(int argc, char *argv[]) {
    if (argc != NUM_ARGS) {
        std::fprintf(stderr, "Usage: %s M N K\n", argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t m, n, k;
    if (
        !parseAndValidateDim(argv[1], m) ||
        !parseAndValidateDim(argv[2], n) ||
        !parseAndValidateDim(argv[3], k)
    ) {
        std::fprintf(stderr, "M, N, and K must be positive uint32_t integers.\n");
        return EXIT_FAILURE;
    }

    Matrix a(m, k, Device::CUDA);
    Matrix b(k, n, Device::CUDA);
    Matrix c(m, n, Device::CUDA);
    Matrix gt(m, n, Device::CUDA);

    a.initRandom(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
    b.initRandom(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
    matMulCudaNaiveLaunch(a, b, gt);

    float avgDurationMs = Benchmark::measureMatMul(
        a, b, c, gt, matMulAutotuneLaunch, NUM_RUNS, NUM_WARM_UPS
    );

    if (avgDurationMs == Benchmark::VALIDATION_FAILED_DURATION) return EXIT_FAILURE;

    printf("%f", avgDurationMs);

    return EXIT_SUCCESS;
}
