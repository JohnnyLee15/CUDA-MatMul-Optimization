#include <cstdint>

#include "matrix.h"
#include "cpu/matMulCpuNaive.h"


void matMulCpuNaiveKernel(const Matrix &a, const Matrix &b, Matrix &c) {
    uint32_t m = a.nrows();
    uint32_t k = a.ncols();
    uint32_t n = b.ncols();

    float *aBuf = a.data();
    float *bBuf = b.data();
    float *cBuf = c.data();
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {

            float acc = 0.0f;
            for (uint32_t p = 0; p < k; p++) {
                acc += aBuf[i * k + p] * bBuf[p * n + j];
            }

            cBuf[i * n  + j] = acc;
        }
    }
}
