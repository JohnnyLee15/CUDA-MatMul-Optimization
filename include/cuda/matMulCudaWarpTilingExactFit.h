#pragma once


class Matrix;


void matMulCudaWarpTilingExactFitLaunch(const Matrix &a, const Matrix &b, Matrix &c);
