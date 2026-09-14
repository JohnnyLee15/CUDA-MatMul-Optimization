#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include <cudaCheck.h>


void cudaCheckImpl(
    cudaError_t error,
    const char *expression,
    const char *fileName,
    uint32_t line
) {
    if (error == cudaSuccess) return;

    std::fprintf(
        stderr,
        "CUDA Error: %s\n"
        "Expression: %s\n"
        "Location: %s:%d\n",
        cudaGetErrorString(error),
        expression,
        fileName,
        line
    );

    std::abort();
}
