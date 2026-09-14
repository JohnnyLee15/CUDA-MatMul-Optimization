#pragma once


class Matrix;


void matMulCuda1DRegTileLaunch(const Matrix &a, const Matrix &b, Matrix &c);
