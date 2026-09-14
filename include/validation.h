#pragma once


class Matrix;


constexpr float ABS_TOL = 1e-4f;
constexpr float REL_TOL = 1e-5f;


void validateMatMul(const Matrix &a, const Matrix &b, const Matrix &c);
