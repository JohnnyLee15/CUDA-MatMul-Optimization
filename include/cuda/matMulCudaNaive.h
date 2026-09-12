#pragma once


class Matrix;


void matMulCudaNaiveLaunch(const Matrix &a, const Matrix &b, Matrix &c);
