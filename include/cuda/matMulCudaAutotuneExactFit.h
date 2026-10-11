#pragma once


class Matrix;

void matMulCudaAutotuneExactFitLaunch(const Matrix &a, const Matrix &b, Matrix &c);
