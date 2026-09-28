#pragma once


class Matrix;


void matMulCudaResolveBankConflictsGeneralLaunch(const Matrix &a, const Matrix &b, Matrix &c);
