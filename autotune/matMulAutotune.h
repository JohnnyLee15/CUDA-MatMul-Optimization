#pragma once


class Matrix;

void matMulAutotuneLaunch(const Matrix &a, const Matrix &b, Matrix &c);
bool matMulAutotuneCanLaunch();
