#pragma once


class Matrix;


void matMulCudaVectorizedGeneralLaunch(const Matrix &a, const Matrix &b, Matrix &c);
