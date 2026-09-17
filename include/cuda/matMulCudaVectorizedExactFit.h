#pragma once


class Matrix;


void matMulCudaVectorizedExactFitLaunch(const Matrix &a, const Matrix &b, Matrix &c);
