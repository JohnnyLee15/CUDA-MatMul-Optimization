#pragma once


class Matrix;


void matMulCublasInit();
void matMulCublasLaunch(const Matrix& a, const Matrix& b, Matrix& c);
void matMulCublasShutdown();
