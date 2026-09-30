#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>

#include "cuda/matMulCublas.h"
#include "matrix.h"


namespace{
cublasHandle_t handle = nullptr;

void checkCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(
            stderr,
            "%s failed (cuBLAS status %d)\n",
            operation,
            static_cast<int>(status)
        );
        std::abort();
    }
}
}


void matMulCublasInit() {
    if (handle != nullptr) return;
    checkCublas(cublasCreate(&handle), "cublasCreate");
}


void matMulCublasLaunch(const Matrix& a, const Matrix& b, Matrix& c) {
    if (handle == nullptr) {
        std::fprintf(stderr, "Call matMulCublasInit before matMulCublasLaunch.\n");
        std::abort();
    }

    const int m = static_cast<int>(a.nrows());
    const int n = static_cast<int>(b.ncols());
    const int k = static_cast<int>(a.ncols());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    checkCublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            b.data(), n,
            a.data(), k,
            &beta,
            c.data(), n
        ),
        "cublasSgemm"
    );
}


void matMulCublasShutdown() {
    if (handle == nullptr) return;
    checkCublas(cublasDestroy(handle), "cublasDestroy");
    handle = nullptr;
}