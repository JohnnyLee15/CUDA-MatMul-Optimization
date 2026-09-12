#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <random>
#include <cstring>

#include <cuda_runtime.h>

#include "matrix.h"
#include "validation.h"
#include "cuda/matCudaUtils.h"
#include "device.h"


Matrix::Matrix(uint32_t numRows, uint32_t numCols, Device device) :
    numRows(numRows),
    numCols(numCols),
    size(numRows * numCols),
    device(device)
{
    switch (device) {
        case Device::CPU:
            buffer = (float*) malloc(size * sizeof(float));
            break;
        case Device::CUDA:
            cudaMalloc(&buffer, size * sizeof(float));
            break;
        default:
            std::abort();
    }
}


Matrix::Matrix(Matrix &&other) noexcept :
    numRows(other.numRows),
    numCols(other.numCols),
    size(other.size),
    device(other.device),
    buffer(other.buffer)
{
    other.buffer = nullptr;
}


bool Matrix::operator==(const Matrix &other) const {
    if (numRows != other.numRows) return false;

    if (numCols != other.numCols) return false;

    if (device != other.device) return false;

    if (device == Device::CUDA) return matCompareLaunch(*this, other);

    if (device == Device::CPU) {
        for (uint32_t i = 0; i < size; i++) {
            if (std::abs(other.buffer[i] - buffer[i]) > ABS_ERROR) return false;
        }

        return true;
    }

    std::abort();
}


void Matrix::initRandom(float min, float max) {
    if (device == Device::CUDA) {
        matInitRandomLaunch(*this, min, max);
        return;
    }

    if (device == Device::CPU) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        for (uint32_t i = 0; i < size; i++) {
            buffer[i] = dist(gen);
        }
        return;
    }

    std::abort();
}


float* Matrix::data() const {
    return buffer;
}


uint32_t Matrix::nrows() const {
    return numRows;
}


uint32_t Matrix::ncols() const {
    return numCols;
}


uint32_t Matrix::getSize() const {
    return size;
}


Device Matrix::getDevice() const {
    return device;
}


Matrix Matrix::to(Device toDevice) const {
    Matrix copy(numRows, numCols, toDevice);

    if (device == Device::CUDA && toDevice == Device::CUDA) {
        cudaMemcpy(copy.buffer, buffer, size * sizeof(float), cudaMemcpyDeviceToDevice);
    } else if (device == Device::CUDA && toDevice == Device::CPU) {
        cudaMemcpy(copy.buffer, buffer, size * sizeof(float), cudaMemcpyDeviceToHost);
    } else if (device == Device::CPU && toDevice == Device::CUDA) {
        cudaMemcpy(copy.buffer, buffer, size * sizeof(float), cudaMemcpyHostToDevice);
    } else if (device == Device::CPU && toDevice == Device::CPU) {
        std::memcpy(copy.buffer, buffer, size * sizeof(float));
    } else {
        std::abort();
    }

    return copy;
}


Matrix::~Matrix() {
    switch (device) {
        case Device::CPU:
            free(buffer);
            break;
        case Device::CUDA:
            cudaFree(buffer);
            break;
        default:
            std::abort();
    }
}
