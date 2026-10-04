#pragma once


class Matrix;


void matMulCudaWarpTilingGeneralLaunch(const Matrix &a, const Matrix &b, Matrix &c);
