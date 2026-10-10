#pragma once


class Matrix;


void matMulCudaWarpSubtilingGeneralLaunch(const Matrix &a, const Matrix &b, Matrix &c);
