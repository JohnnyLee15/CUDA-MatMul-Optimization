#pragma once


class Matrix;


void matMulCudaResolveBankConflictsExactFitLaunch(const Matrix &a, const Matrix &b, Matrix &c);
