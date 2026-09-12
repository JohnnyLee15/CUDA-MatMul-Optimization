#include <cstdlib>

#include "validation.h"
#include "matrix.h"


void validateMatMul(const Matrix &a, const Matrix &b, const Matrix &c) {
    if (a.getDevice() != b.getDevice()) std::abort();
    if (a.ncols() != b.nrows()) std::abort();

    if (a.getDevice() != c.getDevice()) std::abort();
    if (a.nrows() != c.nrows()) std::abort();
    if (b.ncols() != c.ncols()) std::abort();
}
