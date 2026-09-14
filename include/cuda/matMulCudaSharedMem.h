#pragma once


class Matrix;


void matMulCudaSharedMemLaunch(const Matrix &a, const Matrix &b, Matrix &c);
