#pragma once


class Matrix;


void matMulCudaSubWarpTilingExactFitLaunch(const Matrix &a, const Matrix &b, Matrix &c);
