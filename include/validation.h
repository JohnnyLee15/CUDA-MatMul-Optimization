#pragma once


class Matrix;


constexpr float ABS_ERROR = 1e-4f;


void validateMatMul(const Matrix &a, const Matrix &b, const Matrix &c);
