#pragma once


class Matrix;


void matMulCudaWarpSubtilingExactFitLaunch(const Matrix &a, const Matrix &b, Matrix &c);
