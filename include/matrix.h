#pragma once

#include <cstdint>


enum class Device;


class Matrix {
    private:
        uint32_t numRows;
        uint32_t numCols;
        uint32_t size;
        Device device;
        float *buffer;

    public:
        Matrix(uint32_t numRows, uint32_t numCols, Device device);
        Matrix(Matrix&& other) noexcept;

        bool operator==(const Matrix &other) const;

        void initRandom(float min, float max);

        Matrix to(Device device) const;

        float* data() const;

        uint32_t nrows() const;
        uint32_t ncols() const;
        uint32_t getSize() const;
        Device getDevice() const;

        ~Matrix();
};
